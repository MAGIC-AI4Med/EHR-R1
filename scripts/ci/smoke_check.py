#!/usr/bin/env python3
"""Lightweight CI smoke checks for repository health."""

from __future__ import annotations

import json
import os
import py_compile
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_MIN = (3, 10)
SKIP_DIRS = {
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def iter_json_files(root: Path):
    for path in root.rglob("*.json"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def check_python_version() -> None:
    if sys.version_info < PYTHON_MIN:
        min_version = ".".join(map(str, PYTHON_MIN))
        current_version = ".".join(map(str, sys.version_info[:3]))
        raise RuntimeError(
            f"Python {min_version}+ is required, current version is {current_version}."
        )


def compile_python_sources() -> int:
    count = 0
    for path in iter_python_files(REPO_ROOT):
        py_compile.compile(str(path), doraise=True)
        count += 1
    return count


def validate_json_files() -> int:
    count = 0
    for path in iter_json_files(REPO_ROOT):
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        count += 1
    return count


def main() -> int:
    os.chdir(REPO_ROOT)
    check_python_version()

    python_file_count = compile_python_sources()
    json_file_count = validate_json_files()

    print(
        f"Smoke checks passed: compiled {python_file_count} Python files, "
        f"validated {json_file_count} JSON files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
