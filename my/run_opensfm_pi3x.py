"""Run OpenSfM and compare camera, sparse depth, and dense depth priors in Pi3X."""

import argparse
import filecmp
import json
import os
from pathlib import Path
import shutil
import subprocess


REPO = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def prepare_dataset(images: Path, work: Path, config: dict) -> list[str]:
    source = {path.name: path for path in sorted(images.iterdir())
              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    if len(source) < 2:
        raise ValueError(f"Expected at least two images in {images}")
    target = work / "images"
    if target.resolve() == images:
        raise ValueError("The OpenSfM work directory must be separate from the input images")
    target.mkdir(parents=True, exist_ok=True)
    config_file = work / "config.yaml"
    config_text = json.dumps(config, indent=2) + "\n"
    if config_file.exists() and config_file.read_text() != config_text:
        raise ValueError(f"OpenSfM configuration changed. Use a fresh --work-dir for this configuration: {work}")
    staged = {path.name for path in target.iterdir()}
    if staged - source.keys() or (staged != source.keys() and any((work / "logs").glob("*.done"))):
        raise ValueError(f"OpenSfM inputs changed. Use a fresh --work-dir: {work}")
    for name, path in source.items():
        copy = target / name
        if copy.exists():
            if not filecmp.cmp(path, copy, shallow=False):
                raise ValueError(f"Image {name} changed. Use a fresh --work-dir: {work}")
        else:
            shutil.copy2(path, copy)
    config_file.write_text(config_text)
    return list(source)


def complete_component(work: Path, image_names: list[str]) -> int:
    reconstructions = json.loads((work / "reconstruction.json").read_text())
    expected = set(image_names)
    for index, reconstruction in enumerate(reconstructions):
        if expected == set(reconstruction["shots"]) and reconstruction["points"]:
            return index
    reconstructed = max((len(item["shots"]) for item in reconstructions), default=0)
    raise ValueError(f"OpenSfM registered {reconstructed}/{len(expected)} images in its largest component at {work}")


def run_command(command: list[str], environment: str, log: Path, pythonpath: Path | None = None) -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {command[-1] if len(command) == 1 else ' '.join(command[:3])} [{log}]", flush=True)
    with log.open("w") as output:
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", environment, *command],
            cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"Command failed. See {log}\n{tail}")


def run_opensfm(args: argparse.Namespace, images: list[str]) -> None:
    work = args.work_dir
    launcher = str(args.opensfm_repo / "bin/opensfm")
    steps = ("extract_metadata", "detect_features", "match_features", "create_tracks", "reconstruct",
             "undistort", "compute_depthmaps")
    for step in steps:
        log = work / "logs" / f"{step}.log"
        done = log.with_suffix(".done")
        command = [launcher, step, str(work)]
        if step == "undistort":
            command += ["--reconstruction-index", str(complete_component(work, images))]
        if not done.exists():
            run_command(command, args.opensfm_env, log)
            if step == "reconstruct":
                complete_component(work, images)
            if step == "compute_depthmaps":
                cloud = work / "undistorted/depthmaps/merged.ply"
                with cloud.open() as ply:
                    count = next(
                        (int(line.split()[-1]) for line in iter(ply.readline, "") if line.startswith("element vertex ")),
                        0,
                    )
                if count <= 0:
                    raise ValueError(f"OpenSfM produced an empty dense cloud: {cloud}. See {log}")
                print(f"OpenSfM merged {count:,} dense points", flush=True)
            done.touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path.home()
    parser.add_argument("--images", type=Path, default=home / "work/dvlt/images/construction/all")
    parser.add_argument("--work-dir", type=Path, default=home / "work/dvlt/opensfm/construction")
    parser.add_argument("--output-dir", type=Path, default=home / "work/pi3x/construction/opensfm")
    parser.add_argument("--opensfm-repo", type=Path, default=home / "repos/OpenSfM")
    parser.add_argument("--opensfm-env", default="opensfm")
    parser.add_argument("--pi3x-env", default="pi3x")
    parser.add_argument("--stage", choices=("all", "reconstruction", "priors", "prediction", "evaluation"), default="all")
    parser.add_argument("--variant", choices=("cameras", "sparse", "dense"), action="append",
                        help="Run only this prediction variant. May be repeated.")
    parser.add_argument("--blender-priors", type=Path, help="Blender camera and GCP export directory for evaluation")
    parser.add_argument("--feature-type", choices=("HAHOG", "SIFT"), default="HAHOG")
    parser.add_argument("--processes", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--depthmap-resolution", type=int, default=640)
    parser.add_argument("--image-size", type=int, help="Long-edge cap for all three Pi3X variants")
    parser.add_argument("--checkpoint", help="Pi3X Hub ID or local safetensors checkpoint")
    args = parser.parse_args()
    if args.processes < 1 or args.depthmap_resolution < 1:
        parser.error("--processes and --depthmap-resolution must be positive")
    if args.stage == "evaluation" and args.blender_priors is None:
        parser.error("--stage evaluation requires --blender-priors")
    for name in ("images", "work_dir", "output_dir", "opensfm_repo"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    work = args.work_dir
    priors = work / "priors"

    if args.stage in ("all", "reconstruction"):
        config = {
            "feature_type": args.feature_type, "processes": args.processes,
            "matching_gps_distance": 0, "bundle_use_gps": False, "bundle_use_gcp": False,
            "undistorted_image_format": "png", "depthmap_resolution": args.depthmap_resolution,
        }
        image_names = prepare_dataset(args.images, work, config)
        run_opensfm(args, image_names)
    if args.stage in ("all", "priors"):
        run_command(
            ["python", "-m", "my.prepare_priors", "opensfm", "--dataset", str(work), "--output-dir", str(priors)],
            args.opensfm_env, work / "logs/prepare_priors.log", pythonpath=args.opensfm_repo,
        )
        print((work / "logs/prepare_priors.log").read_text().strip(), flush=True)
    if args.stage in ("all", "prediction"):
        command = [
            "python", "-u", "-m", "my.test_pi3x", "--images", str(work / "undistorted/images"),
            "--camera-priors", str(priors / "camera_priors.npz"),
        ]
        if args.image_size is not None:
            command += ["--image-size", str(args.image_size)]
        if args.checkpoint:
            command += ["--checkpoint", args.checkpoint]
        for variant, depth_args in (
            ("cameras", []),
            ("sparse", ["--sparse-depth", str(priors / "sparse_depth.npz")]),
            ("dense", ["--dense-depth-dir", str(priors / "dense_depth")]),
        ):
            if args.variant and variant not in args.variant:
                continue
            output = args.output_dir / variant
            run_command(command + depth_args + ["--output-dir", str(output)], args.pi3x_env, output / "inference.log")
            print((output / "inference.log").read_text().strip(), flush=True)
    if args.stage in ("all", "evaluation") and args.blender_priors:
        log = args.output_dir / "evaluation.log"
        run_command(
            ["python", "-m", "my.evaluate_opensfm_pi3x", "--dataset", str(work),
             "--predictions", str(args.output_dir), "--blender-priors", str(args.blender_priors.expanduser().resolve())],
            args.opensfm_env, log, pythonpath=args.opensfm_repo,
        )
        print(log.read_text().strip(), flush=True)


if __name__ == "__main__":
    main()
