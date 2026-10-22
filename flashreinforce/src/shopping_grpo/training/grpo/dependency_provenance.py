"""Verify local archive installs against the pinned upstream Python source tree."""
from __future__ import annotations
import hashlib
from pathlib import Path


def verify_python_source(manifest: dict, package: Path, expected_revision: str) -> int:
    expected_url = f"https://codeload.github.com/huggingface/transformers/tar.gz/{expected_revision}"
    if manifest.get("revision") != expected_revision or manifest.get("archive_url") != expected_url:
        raise ValueError("source provenance does not identify the pinned upstream commit")
    expected = manifest.get("package_file_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("source provenance is missing Python file hashes")
    package = package.resolve()
    actual = {p.relative_to(package).as_posix(): p for p in package.rglob("*.py")}
    if set(actual) != set(expected):
        raise ValueError("installed Python file set differs from pinned upstream source")
    for name, path in actual.items():
        if not path.resolve().is_relative_to(package):
            raise ValueError("installed package contains a source link outside its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f"installed source differs from pinned upstream: {name}")
    return len(expected)
