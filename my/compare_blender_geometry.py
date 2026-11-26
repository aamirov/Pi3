"""Compare Pi3X geometry against a Blender scene using the Blender cameras."""

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare_priors import load_opensfm, save_npz, undistorted_pixels


def similarity(source: np.ndarray, target: np.ndarray, scale_bounds: tuple[float, float] | None = None):
    x = source - source.mean(axis=0)
    y = target - target.mean(axis=0)
    if np.linalg.matrix_rank(x) < 2:
        raise ValueError("Similarity alignment needs at least three noncollinear points")
    left, singular, right = np.linalg.svd(y.T @ x)
    signs = np.array([1, 1, np.linalg.det(left @ right)])
    rotation = (left * signs) @ right
    scale = (singular @ signs) / np.square(x).sum()
    if scale_bounds is not None:
        scale = np.clip(scale, *scale_bounds)
    translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return scale, rotation, translation


def world_rays(xy: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pixels = np.column_stack((xy + 0.5, np.ones(len(xy))))
    return pixels @ (pose[:3, :3] @ np.linalg.inv(intrinsics)).T


def world_points(xy: np.ndarray, z: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return world_rays(xy, pose, intrinsics) * z[:, None] + pose[:3, 3]


def original_grid(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.column_stack((x.ravel(), y.ravel()))


def pixel_map(dataset: Path, sizes: dict) -> dict:
    from opensfm.features import normalized_image_coordinates

    _, undistorted, _, pairs = load_opensfm(dataset)
    mapping = {}
    for original, shot in pairs:
        w, h = sizes[original.id]
        height, width = undistorted.undistorted_image_size(shot.id)
        normalized = normalized_image_coordinates(original_grid(w, h), w, h)
        coordinates = undistorted_pixels(original, shot, normalized, width, height)
        mapping[original.id] = (shot.id, (width, height), coordinates)
    return mapping


def map_predictions(path: Path, grids: dict, percentile: int):
    with np.load(path, allow_pickle=False) as prediction:
        depths, confidence = prediction["depths"], prediction["depths_conf"]
        valid_pixels = prediction["valid_pixels"]
        transforms = prediction.get("image_pixel_transforms")
        eligible = valid_pixels & (depths > 0) & np.isfinite(depths)
        eligible &= np.isfinite(prediction["world_points"]).all(axis=-1) & np.isfinite(confidence)
        cutoff = float(np.percentile(confidence[eligible], percentile))
        selected = eligible & (confidence >= cutoff)
        order = {name: index for index, name in enumerate(prediction["image_names"])}
    if set(order) != {item[0] for item in grids.values()}:
        raise ValueError(f"Prediction image set does not match the Blender cameras: {path}")

    aligned_depth, aligned_mask = {}, {}
    for name, (shot, (width, height), xy) in grids.items():
        index = order[shot]
        if transforms is None:
            rows = np.flatnonzero(valid_pixels[index].any(axis=1))
            cols = np.flatnonzero(valid_pixels[index].any(axis=0))
            transform = np.array([[cols.size / width, 0, cols[0]], [0, rows.size / height, rows[0]], [0, 0, 1]])
        else:
            transform = transforms[index]
        mapped = (xy + 0.5) @ transform[:2, :2].T + transform[:2, 2]
        map_height, map_width = depths[index].shape
        inside = np.isfinite(mapped).all(axis=1) & ((mapped >= 0) & (mapped < [map_width, map_height])).all(axis=1)
        x, y = np.floor(mapped[inside]).astype(np.intp).T
        values = np.zeros(len(xy), dtype=np.float32)
        mask = np.zeros(len(xy), dtype=bool)
        values[inside] = np.where(eligible[index, y, x], depths[index, y, x], 0)
        mask[inside] = selected[index, y, x]
        aligned_depth[name], aligned_mask[name] = values, mask
    return aligned_depth, aligned_mask, cutoff


def fit_to_controls(depths: dict, poses: dict, intrinsics: dict, sizes: dict, controls: list):
    groups, ratios = [], []
    for control in controls:
        target = np.asarray(control["coordinates"])
        samples = []
        for name, xy in control["images"].items():
            if xy is None or name not in depths:
                continue
            width, height = sizes[name]
            x, y = np.floor(np.asarray(xy) + 0.5).astype(int)
            if not (0 <= x < width and 0 <= y < height):
                continue
            z = depths[name][y * width + x]
            target_z = poses[name][:3, 2] @ (target - poses[name][:3, 3])
            if z > 0 and target_z > 0:
                ratios.append(target_z / z)
                samples.append((name, np.asarray([x, y]), z))
        if samples:
            groups.append((target, samples))
    if len(groups) < 3:
        raise ValueError("Need three or more observed Blender control points")
    depth_scale = float(np.median(ratios))
    source, target = [], []
    for world, samples in groups:
        estimates = [world_points(xy[None], np.array([z * depth_scale]), poses[name], intrinsics[name])[0]
                     for name, xy, z in samples]
        source.append(np.median(estimates, axis=0))
        target.append(world)
    scale, rotation, translation = similarity(np.asarray(source), np.asarray(target))
    return depth_scale, scale, rotation, translation


def save_mesh_samples(output: Path, fits: dict, samples: dict, originals: dict, poses: dict,
                      intrinsics: dict, depths: dict, header: dict) -> None:
    origins, directions, pixels = [], [], []
    for selection in (fits, samples):
        for name, indices in selection.items():
            xy = originals[name][2][indices]
            origins.append(np.broadcast_to(poses[name][:3, 3], (len(xy), 3)))
            directions.append(world_rays(xy, poses[name], intrinsics[name]))
            pixels.append(xy)
    depth_values = [
        np.concatenate([views[name][indices] for selection in (fits, samples) for name, indices in selection.items()])
        for views in depths.values()
    ]
    save_npz(output, names=np.asarray(list(samples)), runs=np.asarray(list(depths)),
             origins=np.concatenate(origins), directions=np.concatenate(directions).astype(np.float32),
             xy=np.concatenate(pixels), depths=np.asarray(depth_values, np.float32),
             metadata=np.asarray(json.dumps(header)))


def prepare(args: argparse.Namespace) -> None:
    from PIL import Image

    with np.load(args.blender_priors / "camera_priors.npz", allow_pickle=False) as archive:
        names = archive["image_names"].tolist()
        poses = dict(zip(names, archive["poses"], strict=True))
        intrinsics = dict(zip(names, archive["intrinsics"], strict=True))
    sizes = {}
    for name in names:
        with Image.open(args.images / name) as image:
            sizes[name] = image.size
    originals = {name: (name, size, original_grid(*size)) for name, size in sizes.items()}
    undistorted = pixel_map(args.opensfm_dataset, sizes) if args.opensfm_dataset else None
    depths, masks, cutoffs = {}, {}, {}
    for path in sorted(args.predictions.rglob("pi3x_predictions.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if len(archive["image_names"]) != len(names):
                continue
        relative = path.parent.relative_to(args.predictions)
        run = relative.as_posix()
        grids = undistorted if relative.parts and relative.parts[0] == "opensfm" else originals
        if grids is None:
            raise ValueError("OpenSfM prediction directories require --opensfm-dataset")
        depths[run], masks[run], cutoffs[run] = map_predictions(path, grids, args.confidence_percentile)
        print(f"{run}: {sum(mask.sum() for mask in masks[run].values()):,} matching source pixels", flush=True)
    if not depths:
        raise ValueError("No predictions with the full set of Blender cameras")
    rng = np.random.default_rng(0)
    samples, coverage, fits = {}, {}, {}
    for name in names:
        common = np.flatnonzero(np.logical_and.reduce([run[name] for run in masks.values()]))
        if not common.size:
            raise ValueError(f"No common confidence-filtered pixels for {name}")
        coverage[name] = len(common)
        if args.alignment == "mesh":
            if len(common) < 2:
                raise ValueError(f"Need separate fit and evaluation pixels for {name}")
            count = min(args.fit_samples_per_view, max(1, len(common) // 4))
            fits[name] = np.sort(rng.choice(common, count, replace=False))
            common = np.setdiff1d(common, fits[name], assume_unique=True)
        samples[name] = np.sort(rng.choice(common, min(len(common), args.samples_per_view), replace=False))
    header = {
        "confidence_percentile": args.confidence_percentile, "common_pixels_per_view": coverage,
        "samples_per_view": {key: len(value) for key, value in samples.items()},
        "alignment_method": args.alignment,
    }
    if args.alignment == "mesh":
        header["fit_samples_per_view"] = {key: len(value) for key, value in fits.items()}
        header["confidence_cutoffs"] = cutoffs
        save_mesh_samples(args.output, fits, samples, originals, poses, intrinsics, depths, header)
        print(f"Saved {sum(map(len, fits.values())):,} fit and {sum(map(len, samples.values())):,} "
              f"held-out points per run to {args.output}")
        return

    controls = json.loads((args.blender_priors / "gcp.json").read_text())
    evaluation = []
    alignments = {}
    for run, views in depths.items():
        depth_scale, scale, rotation, translation = fit_to_controls(views, poses, intrinsics, sizes, controls)
        aligned = []
        for name, indices in samples.items():
            xy = originals[name][2][indices]
            points = world_points(xy, depth_scale * views[name][indices], poses[name], intrinsics[name])
            aligned.append(scale * (points @ rotation.T) + translation)
        evaluation.append(np.concatenate(aligned))
        alignments[run] = {"depth_scale": depth_scale, "similarity_scale": scale, "rotation": rotation.tolist(),
                           "translation": translation.tolist(), "confidence_cutoff": cutoffs[run]}
    header["alignment"] = alignments
    save_npz(args.output, names=np.asarray(names), runs=np.asarray(list(depths)), points=np.asarray(evaluation, np.float32),
             metadata=np.asarray(json.dumps(header)),
             control_points=np.asarray([item["coordinates"] for item in controls]))
    print(f"Saved {sum(map(len, samples.values())):,} shared evaluation points per run to {args.output}")


def mesh_bvh(path: Path):
    import bpy
    from mathutils.bvhtree import BVHTree

    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, triangles = [], []
    offset = 0
    for entry in graph.object_instances:
        obj = entry.object
        if obj.type != "MESH" or obj.original.hide_render or (entry.parent and entry.parent.original.hide_render):
            continue
        mesh = obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            xyz = np.empty((len(mesh.vertices), 3), dtype=np.float32)
            faces = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
            mesh.vertices.foreach_get("co", xyz.ravel())
            mesh.loop_triangles.foreach_get("vertices", faces.ravel())
            transform = np.asarray(entry.matrix_world)
            vertices.append((xyz @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32))
            triangles.append(faces + offset)
            offset += len(xyz)
        finally:
            obj.to_mesh_clear()
    if not triangles:
        raise ValueError("No renderable meshes in Blender's active scene")
    faces = np.concatenate(triangles)
    tree = BVHTree.FromPolygons(np.concatenate(vertices).tolist(), faces.tolist(), all_triangles=True)
    return tree, bpy.context.scene.name, len(faces)


def summarise(distances: np.ndarray) -> dict:
    return {"count": len(distances), "median": float(np.median(distances)), "mean": float(np.mean(distances)),
            "p90": float(np.percentile(distances, 90)), "rmse": float(np.sqrt(np.mean(distances ** 2)))}


def ray_hit_depths(tree, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    depth = np.full(len(origins), np.nan)
    for index, (origin, direction) in enumerate(zip(origins.tolist(), directions.tolist(), strict=True)):
        distance = tree.ray_cast(origin, direction)[3]
        if distance is not None:
            depth[index] = distance
    return depth / np.linalg.norm(directions, axis=1)


def nearest_points(tree, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hits = [tree.find_nearest(point) for point in points.tolist()]
    return np.asarray([tuple(hit[0]) for hit in hits]), np.asarray([hit[3] for hit in hits])


def fit_mesh_similarity(tree, source: np.ndarray, max_iterations: int):
    # The depth seed already puts the cloud in Blender units. Bound the residual scale
    # to keep point-to-surface minimization from shrinking the cloud into one surface.
    scale_bounds = (0.85, 1.15)
    scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
    target, distances = nearest_points(tree, source)
    start = mse = np.mean(distances ** 2)
    steps = 0
    for _ in range(max_iterations):
        candidate = similarity(source, target, scale_bounds)
        s, r, t = candidate
        target, distances = nearest_points(tree, s * (source @ r.T) + t)
        after = np.mean(distances ** 2)
        if after >= mse:
            break
        steps += 1
        gain = mse - after
        scale, rotation, translation = candidate
        mse = after
        if gain <= 1e-5 * max(mse, 1e-12):
            break
    return scale, rotation, translation, {
        "iterations": steps, "scale_bounds": scale_bounds, "initial_fit_rmse": float(np.sqrt(start)),
        "aligned_fit_rmse": float(np.sqrt(mse)),
    }


def align_mesh(tree, archive, report: dict, max_iterations: int):
    origins, directions, depths = (archive[key].astype(np.float64) for key in ("origins", "directions", "depths"))
    n_fit = sum(report["fit_samples_per_view"].values())
    ray_depth = ray_hit_depths(tree, origins[:n_fit], directions[:n_fit])
    hit = np.isfinite(ray_depth) & (ray_depth > 0)
    if not hit.any():
        raise ValueError("The fitted camera rays do not intersect the Blender mesh")
    report["mesh_hit_fit_rays"] = int(hit.sum())
    report["alignment"] = {}
    for run, z in zip(archive["runs"], depths, strict=True):
        seed = float(np.median(ray_depth[hit] / z[:n_fit][hit]))
        points = origins + (seed * z[:, None]) * directions
        print(f"{run}: aligning {n_fit:,} fit points", flush=True)
        scale, rotation, translation, fit = fit_mesh_similarity(tree, points[:n_fit], max_iterations)
        report["alignment"][run] = {
            "depth_seed": seed, "similarity_scale": float(scale), "rotation": rotation.tolist(),
            "translation": translation.tolist(), **fit,
        }
        yield run, scale * (points[n_fit:] @ rotation.T) + translation


def score(args: argparse.Namespace) -> None:
    tree, scene, faces = mesh_bvh(args.blend)
    with np.load(args.samples, allow_pickle=False) as archive:
        names = archive["names"]
        report = json.loads(str(archive["metadata"]))
        report.update(scene=scene, triangles=faces)
        if report.get("alignment_method") == "mesh":
            clouds = align_mesh(tree, archive, report, args.max_iterations)
        else:
            clouds = zip(archive["runs"], archive["points"], strict=True)
            report["control_point_surface_distances"] = [
                tree.find_nearest(point)[3] for point in archive["control_points"].tolist()
            ]
        report["runs"] = {}
        bounds = np.cumsum([0, *(report["samples_per_view"][name] for name in names)])
        for run, cloud in clouds:
            distance = np.fromiter((tree.find_nearest(point)[3] for point in cloud.tolist()),
                                   dtype=np.float64, count=len(cloud))
            if not np.isfinite(distance).all():
                raise RuntimeError(f"Nonfinite Blender mesh distance for {run}")
            report["runs"][run] = {"overall": summarise(distance), "views": {
                name: summarise(distance[bounds[index] : bounds[index + 1]]) for index, name in enumerate(names)
            }}
            print(run, report["runs"][run]["overall"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(report['runs'])} geometry comparisons against {faces:,} Blender triangles to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Back-project and align predictions in Blender coordinates")
    for name in ("predictions", "images", "blender-priors", "output"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_parser.add_argument("--opensfm-dataset", type=Path)
    prepare_parser.add_argument("--confidence-percentile", type=int, default=40, choices=range(100), metavar="0–99")
    prepare_parser.add_argument("--samples-per-view", type=int, default=4096)
    prepare_parser.add_argument("--alignment", choices=("mesh", "gcps"), default="mesh")
    prepare_parser.add_argument("--fit-samples-per-view", type=int, default=2048)
    score_parser = commands.add_parser("score", help="Measure exact point-to-mesh distances using bpy")
    for name in ("blend", "samples", "output"):
        score_parser.add_argument(f"--{name}", type=Path, required=True)
    score_parser.add_argument("--max-iterations", type=int, default=120)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.command == "prepare":
        if min(args.samples_per_view, args.fit_samples_per_view) < 1:
            parser.error("Sample counts must be positive")
        prepare(args)
    else:
        if args.max_iterations < 1:
            parser.error("--max-iterations must be positive")
        score(args)


if __name__ == "__main__":
    main()
