#!/usr/bin/env python3
from __future__ import annotations

import os
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
    ensure_aws()
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
