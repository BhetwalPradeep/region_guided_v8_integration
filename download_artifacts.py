#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

S3_PREFIX = "s3://viewray-ai/Patient-vision/Pradeep/Region-aware-v8"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DOCS_DIR = ROOT / "docs"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_aws() -> None:
    if shutil.which("aws") is None:
        raise RuntimeError(
            "AWS CLI is required. Install it and configure credentials or an IAM role with access to the S3 prefix."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download versioned model artifacts from S3")
    parser.add_argument("--release", choices=["legacy", "final-14p"], default="legacy")
    args = parser.parse_args()
    ensure_aws()
    if args.release == "final-14p":
        manifest = json.loads((ROOT / "releases" / "final-14p.json").read_text())
        destination = ROOT / manifest["model_directory"]
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoint_best.weights.h5", "config.json"):
            run(["aws", "s3", "cp", f'{manifest["s3_prefix"]}/model/{name}', str(destination / name)])
        checkpoint = destination / "checkpoint_best.weights.h5"
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest["checkpoint_sha256"]:
            raise RuntimeError("Final-14p checkpoint checksum mismatch; do not use these weights")
        config = json.loads((destination / "config.json").read_text())
        if config.get("target_direction") != manifest["target_direction"]:
            raise RuntimeError("Final-14p configuration direction mismatch")
        print(f"Verified Final-14p artifacts in {destination}")
        return
    MODEL_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    run(["aws", "s3", "cp", f"{S3_PREFIX}/model/checkpoint_best.weights.h5", str(MODEL_DIR / "checkpoint_best.weights.h5")])
    run(["aws", "s3", "cp", f"{S3_PREFIX}/model/config.json", str(MODEL_DIR / "config.json")])
    run([
        "aws", "s3", "cp",
        f"{S3_PREFIX}/docs/",
        str(DOCS_DIR),
        "--recursive",
        "--exclude",
        "*",
        "--include",
        "*.csv",
        "--include",
        "*.tar.gz",
    ])

    print(f"Artifacts downloaded to {ROOT}")
    print(f"S3 source: {S3_PREFIX}")


if __name__ == "__main__":
    main()
