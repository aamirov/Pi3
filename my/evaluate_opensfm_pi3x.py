"""Compare OpenSfM-conditioned Pi3X runs with Blender cameras and GCPs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare_priors import load_opensfm, pixel_in_image, undistorted_pixels


def median_camera_error(estimated: np.ndarray, reference: np.ndarray) -> float:
    estimated = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    x = estimated - estimated.mean(axis=0)
    y = reference - reference.mean(axis=0)
    spread = np.square(x).sum()
    if spread <= 0:
        raise ValueError("Camera alignment requires distinct predicted camera centers")
    left, singular_values, right = np.linalg.svd(y.T @ x)
    signs = [1, 1, np.linalg.det(left @ right)]
    rotation = (left * signs) @ right
    scale = np.dot(singular_values, signs) / spread
    return float(np.median(np.linalg.norm(scale * (x @ rotation.T) - y, axis=1)))


def load_reference(dataset: Path, blender: Path):
    from opensfm.features import normalized_image_coordinates

    _, undistorted, _, pairs = load_opensfm(dataset)
    with np.load(blender / "camera_priors.npz", allow_pickle=False) as archive:
        positions = dict(zip(archive["image_names"], archive["poses"][:, :3, 3], strict=True))
    with np.load(blender / "gcp_sparse_depth.npz", allow_pickle=False) as archive:
        names, xy, depths = (archive[key] for key in ("image_names", "xy", "depths"))

    centers, samples = {}, {}
    for original, shot in pairs:
        centers[shot.id] = positions[original.id]
        selected = names == original.id
        if not selected.any():
            continue
        height, width = undistorted.undistorted_image_size(shot.id)
        normalized = normalized_image_coordinates(xy[selected], original.camera.width, original.camera.height)
        coordinates = undistorted_pixels(original, shot, normalized, width, height)
        inside = pixel_in_image(coordinates, width, height)
        samples[shot.id] = coordinates[inside], depths[selected][inside]
    return centers, samples


def score_prediction(prediction, centers: dict, samples: dict) -> dict:
    actual_depths, target_depths = [], []
    names = prediction["image_names"]
    estimate = prediction["camera_to_worlds"][:, :3, 3]
    position_error = median_camera_error(estimate, np.asarray([centers[name] for name in names]))
    transforms = prediction["image_pixel_transforms"]
    depths = prediction["depths"]
    valid = prediction["valid_pixels"]
    for index, name in enumerate(names):
        if name not in samples:
            continue
        xy, target = samples[name]
        transform = transforms[index]
        mapped = (xy + 0.5) @ transform[:2, :2].T + transform[:2, 2]
        x, y = np.floor(mapped).astype(np.intp).T
        height, width = valid[index].shape
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not inside.all() or not valid[index, y, x].all():
            raise ValueError(f"Mapped GCP outside the prediction area in {name}")
        actual_depths.extend(depths[index, y, x])
        target_depths.extend(target)
    if not target_depths:
        raise ValueError("No GCP observations overlap the prediction images")
    actual, target = np.asarray(actual_depths), np.asarray(target_depths)
    if not (np.isfinite(actual).all() and (actual > 0).all() and np.isfinite(target).all() and (target > 0).all()):
        raise ValueError("Expected finite positive predicted and reference depths at GCP pixels")
    scale = float(np.median(target / actual))
    return {
        "cameras": len(names),
        "aligned_median_camera_center_error_blender_units": position_error,
        "gcp_observations": len(target),
        "median_absolute_relative_depth_error_percent": float(100 * np.median(np.abs(actual / target - 1))),
        "fitted_depth_scale": scale,
        "scaled_median_absolute_relative_depth_error_percent": float(100 * np.median(np.abs(scale * actual / target - 1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--blender-priors", type=Path, required=True)
    args = parser.parse_args()
    centers, samples = load_reference(args.dataset.expanduser(), args.blender_priors.expanduser())
    predictions = args.predictions.expanduser()
    metrics = {}
    camera_inputs = None
    for variant in ("cameras", "sparse", "dense"):
        with np.load(predictions / variant / "pi3x_predictions.npz", allow_pickle=False) as prediction:
            inputs = {key: prediction[key] for key in (
                "image_names", "image_pixel_transforms", "input_camera_to_worlds", "input_intrinsics"
            )}
            if camera_inputs is None:
                camera_inputs = inputs
            elif any(not np.array_equal(inputs[key], value) for key, value in camera_inputs.items()):
                raise ValueError("Evaluation requires the same images, resolution, and input cameras in all variants")
            metrics[variant] = score_prediction(prediction, centers, samples)
    serialized = json.dumps(metrics, indent=2) + "\n"
    (predictions / "evaluation.json").write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
