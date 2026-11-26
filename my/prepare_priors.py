"""Convert Blender or OpenSfM camera and depth data for the Pi3X runner."""

import argparse
import json
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import numpy as np


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging:
        temp = Path(staging) / path.name
        np.savez_compressed(temp, **arrays)
        temp.replace(path)


def copy_original(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError(f"Different camera source already exists at {destination}")
    if source != destination.resolve():
        copy2(source, destination)


def blender_camera(document: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    camera, render = document["camera"], document["render"]
    if (width, height) != (render["effective_width"], render["effective_height"]):
        raise ValueError("Blender render dimensions do not match the image")
    fit = camera["sensor_fit"]
    if camera["type"] != "PERSP" or fit not in {"AUTO", "HORIZONTAL", "VERTICAL"}:
        raise ValueError("Expected a perspective Blender camera with a supported sensor fit")
    x, y, z = (document["rotation_euler"][axis] for axis in "xyz")
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = (rz @ ry @ rx) * [1, -1, -1]
    pose[:3, 3] = [document["location"][axis] for axis in "xyz"]
    sensor = camera["sensor_height" if fit == "VERTICAL" else "sensor_width"]
    extent = height if fit == "VERTICAL" or (fit == "AUTO" and height > width) else width
    focal = camera["focal_length"] * extent / sensor
    intrinsics = np.array([
        [focal, 0, width / 2 - extent * camera.get("shift_x", 0)],
        [0, focal, height / 2 + extent * camera.get("shift_y", 0)],
        [0, 0, 1],
    ])
    return pose, intrinsics


def export_blender(images: Path, blender_dir: Path, output: Path, gcp: Path | None) -> None:
    from PIL import Image

    names = sorted(path.name for path in images.iterdir()
                   if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not names:
        raise ValueError(f"No images in {images}")
    poses, intrinsics, sizes = {}, {}, {}
    for name in names:
        camera_path = blender_dir / Path(name).with_suffix(".json")
        with Image.open(images / name) as image:
            sizes[name] = image.size
        poses[name], intrinsics[name] = blender_camera(json.loads(camera_path.read_text()), *sizes[name])
        copy_original(camera_path, output / "blender" / camera_path.name)
    save_npz(output / "camera_priors.npz", image_names=np.asarray(names),
             poses=np.asarray(list(poses.values()), dtype=np.float32),
             intrinsics=np.asarray(list(intrinsics.values()), dtype=np.float32))

    if gcp is not None:
        entries = []
        for point in json.loads(gcp.read_text()):
            world = np.asarray(point["coordinates"])
            for name, pixel in point["images"].items():
                if name not in poses or pixel is None:
                    continue
                pose = poses[name]
                z = pose[:3, 2] @ (world - pose[:3, 3])
                if not np.isfinite(z) or z <= 0 or not pixel_in_image(np.atleast_2d(pixel), *sizes[name])[0]:
                    raise ValueError(f"Invalid GCP {point['name']} observation in {name}")
                entries.append((name, pixel, z, point["name"]))
        if not entries:
            raise ValueError("No GCP observations match the selected images")
        observed_names, xy, depths, point_names = zip(*entries, strict=True)
        copy_original(gcp, output / "gcp.json")
        save_npz(output / "gcp_sparse_depth.npz", image_names=np.asarray(observed_names),
                 xy=np.asarray(xy, dtype=np.float32), depths=np.asarray(depths, dtype=np.float32),
                 gcp_names=np.asarray(point_names))
        print(f"Exported {len(entries):,} GCP depth observations")
    print(f"Exported {len(names)} Blender cameras to {output}")


def pixel_in_image(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.isfinite(xy).all(axis=1) & ((xy >= -0.5) & (xy < [width - 0.5, height - 0.5])).all(axis=1)


def load_opensfm(work: Path):
    from opensfm.dataset import DataSet

    data = DataSet(str(work))
    undistorted = data.undistorted_dataset()
    mapping = undistorted.load_undistorted_shot_ids()
    cameras, = undistorted.load_undistorted_reconstruction()
    names = set(data.images())
    original = next((model for model in data.load_reconstruction() if set(model.shots) == names), None)
    if original is None or names != mapping.keys() or any(len(ids) != 1 for ids in mapping.values()):
        raise ValueError("Expected one registered reconstruction and one undistorted image per input")
    pairs = [(original.shots[name], cameras.shots[ids[0]]) for name, ids in sorted(mapping.items())]
    if len({shot.id for _, shot in pairs}) != len(pairs):
        raise ValueError("Undistorted image names must be unique")
    return data, undistorted, original, pairs


def undistorted_pixels(original, shot, observations: np.ndarray, width: int, height: int) -> np.ndarray:
    from opensfm.features import denormalized_image_coordinates

    rotation = shot.pose.get_R_world_to_cam() @ original.pose.get_R_cam_to_world()
    rays = original.camera.pixel_bearing_many(observations) @ rotation.T
    return denormalized_image_coordinates(shot.camera.project_many(rays), width, height)


def aligned_depth_map(undistorted, shot, width: int, height: int) -> np.ndarray:
    import cv2

    if not undistorted.clean_depthmap_exists(shot.id):
        return np.zeros((height, width), dtype=np.float32)
    depth, _, _ = undistorted.load_clean_depthmap(shot.id)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32, copy=False)
    if depth.shape == (height, width):
        return depth
    source_k = shot.camera.get_K_in_pixel_coordinates(depth.shape[1], depth.shape[0])
    target_k = shot.camera.get_K_in_pixel_coordinates(width, height)
    inverse = (source_k @ np.linalg.inv(target_k))[:2]
    # Convert camera coordinates at pixel edges to OpenCV sample centers.
    inverse[:, 2] += inverse[:, :2] @ [0.5, 0.5] - 0.5
    return cv2.warpAffine(depth, inverse, (width, height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)


def export_opensfm(work: Path, output: Path) -> None:
    from opensfm.features import denormalized_image_coordinates

    data, undistorted, reconstruction, pairs = load_opensfm(work)
    tracks = data.load_tracks_manager()
    names, poses, intrinsics = [], [], []
    sparse_names, sparse_xy, sparse_z = [], [], []
    coverage = {}
    for original, shot in pairs:
        height, width = undistorted.undistorted_image_size(shot.id)
        names.append(shot.id)
        poses.append(shot.pose.get_cam_to_world())
        intrinsics.append(shot.camera.get_K_in_pixel_coordinates(width, height))
        observations = [(reconstruction.points[key].coordinates, observation.point)
                        for key, observation in tracks.get_shot_observations(original.id).items()
                        if key in reconstruction.points]
        sparse_count = 0
        if observations:
            world, normalized = map(np.asarray, zip(*observations, strict=True))
            local = shot.pose.transform_many(world)
            captured = undistorted_pixels(original, shot, normalized, width, height)
            projected = denormalized_image_coordinates(shot.camera.project_many(local), width, height)
            keep = (pixel_in_image(captured, width, height) & (local[:, 2] > 0)
                    & np.isfinite(local[:, 2]) & (np.linalg.norm(projected - captured, axis=1) <= 3))
            sparse_count = int(keep.sum())
            sparse_names.extend([shot.id] * sparse_count)
            sparse_xy.append(captured[keep])
            sparse_z.append(local[keep, 2])
        dense = aligned_depth_map(undistorted, shot, width, height)
        coverage[shot.id] = {"sparse_observations": sparse_count, "dense_pixels": int(np.count_nonzero(dense)),
                             "image_pixels": width * height}
        save_npz(output / "dense_depth" / f"{shot.id}.npz", depth=dense)
    if not sparse_names or not any(view["dense_pixels"] for view in coverage.values()):
        raise ValueError("OpenSfM did not produce usable sparse and dense depth")
    data.save_ply(reconstruction, tracks, filename="sparse.ply", no_cameras=True)
    save_npz(output / "camera_priors.npz", image_names=np.asarray(names), poses=np.asarray(poses, dtype=np.float32),
             intrinsics=np.asarray(intrinsics, dtype=np.float32))
    save_npz(output / "sparse_depth.npz", image_names=np.asarray(sparse_names),
             xy=np.concatenate(sparse_xy).astype(np.float32), depths=np.concatenate(sparse_z).astype(np.float32))
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(f"Exported {len(names)} OpenSfM cameras and {len(reconstruction.points):,} sparse cloud points")
    print(f"Kept {len(sparse_names):,} sparse observations and "
          f"{sum(view['dense_pixels'] for view in coverage.values()):,} clean depth pixels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    opensfm = modes.add_parser("opensfm", help="Convert an OpenSfM dataset after dense reconstruction")
    opensfm.add_argument("--dataset", type=Path, required=True)
    opensfm.add_argument("--output-dir", type=Path, required=True)
    blender = modes.add_parser("blender", help="Convert Blender JSON and optional GCP observations")
    blender.add_argument("--images", type=Path, required=True)
    blender.add_argument("--blender-dir", type=Path, required=True)
    blender.add_argument("--gcp", type=Path)
    blender.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.mode == "opensfm":
        export_opensfm(args.dataset.expanduser().resolve(), output)
    else:
        gcp = args.gcp.expanduser().resolve() if args.gcp else None
        export_blender(args.images.expanduser().resolve(), args.blender_dir.expanduser().resolve(), output, gcp)


if __name__ == "__main__":
    main()
