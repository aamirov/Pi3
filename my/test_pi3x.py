"""Run Pi3X and save predictions for the Three.js point cloud browser."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image
from safetensors import safe_open

from pi3.models.pi3x import Pi3X
from pi3.utils.geometry import recover_intrinsic_from_rays_d


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PATCH_SIZE = 14
DEPTH_KEYS = ("depth_encoder.", "depth_emb")
MULTIMODAL_KEYS = (*DEPTH_KEYS, "ray_embed.", "pose_inject_blk.")


class InferencePi3X(Pi3X):
    def _chunked_conv_head(self, head, feat, patch_h, patch_w, chunk_size=4):
        # Decode the views jointly and upsample their dense outputs in smaller batches.
        return super()._chunked_conv_head(head, feat, patch_h, patch_w, chunk_size=chunk_size)


class OffloadedDepthEncoder(torch.nn.Module):
    """Keep the depth encoder on GPU only while extracting per-image features."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, inputs: torch.Tensor, *, is_training: bool) -> dict[str, torch.Tensor]:
        self.encoder.to(inputs.device)
        try:
            patches = [
                self.encoder(batch, is_training=is_training)["x_norm_patchtokens"]
                for batch in inputs.split(4)
            ]
        finally:
            self.encoder.cpu()
        return {"x_norm_patchtokens": torch.cat(patches)}


def load_images(paths: list[Path], image_size: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    original_sizes = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            original_sizes.append(image.size)
            if image_size is not None and max(image.size) > image_size:
                image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
            frames.append(np.asarray(image))

    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    height = ((height + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    width = ((width + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    rgb = np.zeros((len(paths), height, width, 3), dtype=np.uint8)
    valid = np.zeros((len(paths), height, width), dtype=bool)
    pixel_transforms = np.zeros((len(paths), 3, 3), dtype=np.float32)
    for index, (frame, (original_w, original_h)) in enumerate(zip(frames, original_sizes, strict=True)):
        h, w, _ = frame.shape
        top, left = (height - h) // 2, (width - w) // 2
        rgb[index, top : top + h, left : left + w] = frame
        valid[index, top : top + h, left : left + w] = True
        pixel_transforms[index] = ((w / original_w, 0, left), (0, h / original_h, top), (0, 0, 1))
    return rgb, valid, pixel_transforms


def load_camera_priors(
    path: Path, image_paths: list[Path], pixel_transforms: np.ndarray
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["image_names"]
        poses = archive["poses"].astype(np.float32)
        intrinsics = archive["intrinsics"].astype(np.float32)

    if names.ndim != 1 or names.dtype.kind != "U" or len(set(names)) != len(names):
        raise ValueError("Camera priors must contain unique image_names")
    if poses.shape != (len(names), 4, 4) or intrinsics.shape != (len(names), 3, 3):
        raise ValueError("Camera priors must contain one 4x4 pose and 3x3 intrinsics matrix per image")
    if not (np.isfinite(poses).all() and np.isfinite(intrinsics).all()):
        raise ValueError("Camera priors must be finite")
    rotation = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1])
        or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-4)
        or not (np.linalg.det(rotation) > 0).all()
    ):
        raise ValueError("Camera priors must contain rigid camera-to-world poses")
    if not np.allclose(intrinsics[:, 2], [0, 0, 1]) or not (intrinsics[:, (0, 1), (0, 1)] > 0).all():
        raise ValueError("Camera prior intrinsics must have positive focal lengths and a homogeneous last row")

    by_name = {name: row for row, name in enumerate(names)}
    missing = [image.name for image in image_paths if image.name not in by_name]
    if missing:
        raise ValueError(f"Camera priors are missing for {', '.join(missing)}")
    order = [by_name[image.name] for image in image_paths]
    return {"poses": poses[order], "intrinsics": pixel_transforms @ intrinsics[order]}


def load_sparse_depth(
    path: Path, image_paths: list[Path], valid: np.ndarray, pixel_transforms: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["image_names"]
        xy = archive["xy"]
        depths = archive["depths"].astype(np.float32)

    if names.ndim != 1 or names.dtype.kind != "U" or xy.shape != (len(names), 2) or depths.shape != names.shape:
        raise ValueError("Sparse depth requires one image name, (x, y) pixel, and camera Z per observation")
    if not (np.isfinite(xy).all() and np.isfinite(depths).all() and (depths > 0).all()):
        raise ValueError("Sparse depth observations need finite coordinates and positive depths")

    maps = np.full(valid.shape, np.inf, dtype=np.float32)
    for index, path in enumerate(image_paths):
        selected = names == path.name
        transform = pixel_transforms[index]
        # Map zero-based pixel centers through the same resize and padding as RGB.
        mapped = (xy[selected] + 0.5) @ transform[:2, :2].T + transform[:2, 2]
        height, width = valid[index].shape
        if ((mapped < 0) | (mapped >= [width, height])).any():
            raise ValueError(f"Sparse depth observation outside {path.name}")
        x, y = np.floor(mapped).astype(np.intp).T
        if not valid[index, y, x].all():
            raise ValueError(f"Sparse depth observation outside {path.name}")
        # When observations land on the same pixel, keep the nearest surface.
        np.minimum.at(maps[index], (y, x), depths[selected])
    missing = np.isinf(maps)
    if missing.all():
        raise ValueError("No sparse depth observations match the selected images")
    maps[missing] = 0
    return maps


def load_dense_depth(
    directory: Path, image_paths: list[Path], valid: np.ndarray, pixel_transforms: np.ndarray
) -> np.ndarray:
    maps = np.zeros(valid.shape, dtype=np.float32)
    for index, path in enumerate(image_paths):
        with np.load(directory / f"{path.name}.npz", allow_pickle=False) as archive:
            depth = archive["depth"].astype(np.float32)
        rows = np.flatnonzero(valid[index].any(axis=1))
        columns = np.flatnonzero(valid[index].any(axis=0))
        top, left = rows[0], columns[0]
        height, width = rows[-1] - top + 1, columns[-1] - left + 1
        sx, sy = pixel_transforms[index, (0, 1), (0, 1)]
        expected = (round(height / float(sy)), round(width / float(sx)))
        if depth.shape != expected:
            raise ValueError(f"{path.name}: expected original depth shape {expected}, got {depth.shape}")
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
        if depth.shape != (height, width):
            depth = np.asarray(Image.fromarray(depth).resize((width, height), Image.Resampling.NEAREST))
        maps[index, top : top + height, left : left + width] = depth
        maps[index, ~valid[index]] = 0
    if not np.any(maps):
        raise ValueError("No positive dense depths in the selected images")
    return maps


def load_checkpoint(checkpoint: str) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        path = path / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if path.is_absolute() or checkpoint.startswith(".") or path.suffix == ".safetensors":
        raise FileNotFoundError(path)

    try:
        return Path(hf_hub_download(checkpoint, "model.safetensors", local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(hf_hub_download(checkpoint, "model.safetensors"))


def predict(
    rgb: np.ndarray, valid: np.ndarray, checkpoint: Path,
    camera_priors: dict[str, np.ndarray] | None = None, depth_priors: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    use_multimodal = camera_priors is not None or depth_priors is not None
    model = InferencePi3X(use_multimodal=use_multimodal).eval()
    ignored_keys = () if use_multimodal else MULTIMODAL_KEYS
    if use_multimodal and depth_priors is None:
        # Camera-only conditioning leaves the depth encoder idle.
        del model.depth_encoder
        del model.depth_emb
        ignored_keys = DEPTH_KEYS
    with safe_open(checkpoint, framework="pt") as weights:
        model.load_state_dict({
            key: weights.get_tensor(key) for key in weights.keys() if not key.startswith(ignored_keys)
        })
    if depth_priors is not None:
        depth_encoder = model.depth_encoder
        del model.depth_encoder
        model.cuda()
        model.depth_encoder = OffloadedDepthEncoder(depth_encoder).eval()
    else:
        model.cuda()
    images = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].to(device="cuda", dtype=torch.float32) / 255
    conditions = {key: torch.from_numpy(value)[None].cuda() for key, value in (camera_priors or {}).items()}
    if depth_priors is not None:
        conditions["depths"] = torch.from_numpy(depth_priors)[None].cuda()
        conditions["mask_add_depth"] = (conditions["depths"] > 0).flatten(2).any(dim=-1)
    elif use_multimodal:
        conditions["mask_add_depth"] = torch.zeros(images.shape[:2], device="cuda", dtype=torch.bool)
    torch.cuda.synchronize()
    started = perf_counter()
    with torch.inference_mode():
        result = model(imgs=images, **conditions)
        intrinsics = recover_intrinsic_from_rays_d(result["rays"], force_center_principal_point=True)
    torch.cuda.synchronize()
    print(f"Inference and intrinsics took {perf_counter() - started:.2f} s", flush=True)

    views, height, width = valid.shape
    fields = {
        "depths": (result["local_points"][0, ..., 2], (views, height, width)),
        "world_points": (result["points"][0], rgb.shape),
        "depths_conf": (result["conf"][0, ..., 0].sigmoid(), (views, height, width)),
        "camera_to_worlds": (result["camera_poses"][0, :, :3], (views, 3, 4)),
        "intrinsics": (intrinsics[0], (views, 3, 3)),
    }
    arrays = {}
    for name, (value, shape) in fields.items():
        if value.shape != shape:
            raise RuntimeError(f"{name}: expected shape {shape}, got {tuple(value.shape)}")
        array = value.float().cpu().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name}: found non-finite predictions")
        arrays[name] = array
    if not (valid & (arrays["depths"] > 0)).any(axis=(1, 2)).all():
        raise RuntimeError("Expected positive depth in every input view")
    if not (arrays["intrinsics"][:, (0, 1), (0, 1)] > 0).all():
        raise RuntimeError("Expected positive focal lengths in every input view")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path.home() / "work/dvlt/images/construction/3")
    parser.add_argument("--image-size", type=int, help="Maximum input long edge before patch padding")
    parser.add_argument("--checkpoint", default="yyfz233/Pi3X", help="Model ID or local safetensors checkpoint")
    parser.add_argument(
        "--camera-priors", type=Path,
        help="NPZ with image_names, OpenCV camera-to-world poses, and intrinsics at original image size",
    )
    depth_options = parser.add_mutually_exclusive_group()
    depth_options.add_argument(
        "--sparse-depth", type=Path,
        help="NPZ with image_names, xy at original pixel centers (zero-based), and positive camera-Z depths",
    )
    depth_options.add_argument(
        "--dense-depth-dir", type=Path,
        help="Directory of {image_name}.npz files with original-resolution camera-Z depth arrays",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.image_size is not None and args.image_size < PATCH_SIZE:
        parser.error(f"--image-size must be at least {PATCH_SIZE}")
    image_dir = args.images.expanduser()
    if not image_dir.is_dir():
        parser.error(f"image directory does not exist: {image_dir}")
    paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < 2:
        parser.error(f"expected at least two images in {image_dir}")
    if not torch.cuda.is_available():
        parser.error("a CUDA device is required")

    rgb, valid, pixel_transforms = load_images(paths, args.image_size)
    camera_priors = None
    if args.camera_priors:
        try:
            camera_priors = load_camera_priors(args.camera_priors.expanduser(), paths, pixel_transforms)
        except (KeyError, OSError, ValueError) as error:
            parser.error(f"could not load camera priors: {error}")
    depth_priors = None
    if args.sparse_depth:
        try:
            depth_priors = load_sparse_depth(args.sparse_depth.expanduser(), paths, valid, pixel_transforms)
        except (KeyError, OSError, ValueError) as error:
            parser.error(f"could not load sparse depth: {error}")
    if args.dense_depth_dir:
        try:
            depth_priors = load_dense_depth(args.dense_depth_dir.expanduser(), paths, valid, pixel_transforms)
        except (KeyError, OSError, ValueError) as error:
            parser.error(f"could not load dense depth: {error}")
    _, height, width, _ = rgb.shape
    print(f"Running Pi3X on {len(paths)} images at {width}x{height} on {torch.cuda.get_device_name()} (FP32)", flush=True)
    if camera_priors is not None:
        print(f"Conditioning on {len(paths)} camera poses and intrinsics from {args.camera_priors.expanduser()}", flush=True)
    if depth_priors is not None:
        depth_source = (args.sparse_depth or args.dense_depth_dir).expanduser()
        print(f"Conditioning on {np.count_nonzero(depth_priors):,} depths from {depth_source}", flush=True)
    checkpoint = load_checkpoint(args.checkpoint)
    print(f"Loading checkpoint {checkpoint}", flush=True)
    arrays = predict(rgb, valid, checkpoint, camera_priors, depth_priors)
    if camera_priors is not None:
        arrays.update(input_camera_to_worlds=camera_priors["poses"], input_intrinsics=camera_priors["intrinsics"])
    if depth_priors is not None:
        arrays["input_depths"] = depth_priors

    default_run = "3-with-poses" if camera_priors is not None else "3"
    if depth_priors is not None:
        default_run += "-sparse-depth" if args.sparse_depth else "-dense-depth"
    output_dir = (args.output_dir or Path.home() / "work/pi3x/construction" / default_run).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "pi3x_predictions.npz"
    with TemporaryDirectory(dir=output_dir) as staging:
        staged = Path(staging) / archive.name
        np.savez_compressed(
            staged, **arrays, image_names=np.asarray([path.name for path in paths]), images_rgb=rgb,
            valid_pixels=valid, image_pixel_transforms=pixel_transforms,
        )
        staged.replace(archive)
    print(f"PASS: {len(paths)} cameras and {width}x{height} depth maps with finite 3D points")
    print(f"Predictions saved to {archive}")


if __name__ == "__main__":
    main()
