#!/usr/bin/env python3
"""Download the complete AnyWord-3M and AnyText benchmark assets.

This script is intentionally resumable and writes all large artifacts under
the ignored data/ directory. It records remote revisions, expected sizes, and
the final local inventory without putting credentials in the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANYWORD_DIR = ROOT / "data/raw/AnyWord-3M"
DEFAULT_BENCHMARK_DIR = ROOT / "data/raw/AnyText-benchmark"
DEFAULT_CACHE_DIR = ROOT / "data/cache/huggingface"
ANYWORD_REPO = "stzhao/AnyWord-3M"
ANYWORD_REVISION = "54f67ddf34eb4a5aac46539c2a5b511dd0ca1fc4"

BENCHMARK_ASSETS: dict[str, dict[str, Any]] = {
    "benchmark.zip": {
        "file_id": "1KtW3BysAEDG9R3HJwA8SSwVoRXL4VZmZ",
        "expected_bytes": 101863540,
        "source_page": "https://drive.google.com/drive/folders/1Eesj6HTqT1kCi6QLyL5j0mL_ELYRp3GV",
    },
    "FID.zip": {
        "file_id": "1qXgpUcGlQBpNb3HO9ibbP_Bx6vat6n4S",
        "expected_bytes": 4084523407,
        "source_page": "https://drive.google.com/drive/folders/1Eesj6HTqT1kCi6QLyL5j0mL_ELYRp3GV",
    },
    "anytext_eval_imgs.zip": {
        "file_id": "1pGN35myilYY04ChFtgAosYr0oqeBy4NU",
        # The Drive viewer metadata reports 4,653,294,494 bytes, while the
        # downloadable object and the completed archive are 4,653,291,494.
        "expected_bytes": 4653291494,
        "source_page": "https://drive.google.com/file/d/1pGN35myilYY04ChFtgAosYr0oqeBy4NU/view?usp=drive_link",
    },
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def anyword_manifest(token: str | None) -> dict[str, Any]:
    api = HfApi(token=token or None)
    info = api.dataset_info(ANYWORD_REPO, revision=ANYWORD_REVISION, files_metadata=True)
    files: list[dict[str, Any]] = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if not (name.endswith(".parquet") or name in {"README.md", ".gitattributes"}):
            continue
        lfs = getattr(sibling, "lfs", None) or {}
        files.append(
            {
                "path": name,
                "size": getattr(sibling, "size", None) or lfs.get("size"),
                "oid": lfs.get("oid"),
                "sha256": getattr(sibling, "blob_id", None),
            }
        )
    files.sort(key=lambda x: x["path"])
    return {
        "repo_id": ANYWORD_REPO,
        "revision": ANYWORD_REVISION,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(x["size"] or 0) for x in files),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_anyword(args: argparse.Namespace, token: str) -> dict[str, Any]:
    destination = args.anyword_dir
    destination.mkdir(parents=True, exist_ok=True)
    manifest = anyword_manifest(token)
    write_json(destination / "REMOTE_MANIFEST.json", manifest)
    print(
        f"AnyWord-3M: {manifest['file_count']} files, "
        f"{manifest['total_bytes'] / 1024**3:.2f} GiB at {manifest['revision']}",
        flush=True,
    )
    snapshot_download(
        repo_id=ANYWORD_REPO,
        repo_type="dataset",
        revision=ANYWORD_REVISION,
        token=token,
        local_dir=str(destination),
        cache_dir=str(args.cache_dir),
        max_workers=args.hf_workers,
        allow_patterns=["*.parquet", "**/*.parquet", "README.md", ".gitattributes"],
    )
    return manifest


def download_google_asset(name: str, spec: dict[str, Any], destination: Path, retries: int = 8) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / name
    part_path = destination / (name + ".part")
    url = f"https://drive.usercontent.google.com/download?id={spec['file_id']}&export=download&confirm=t"
    expected = int(spec["expected_bytes"])
    existing = part_path.stat().st_size if part_path.exists() else 0
    if final_path.exists() and final_path.stat().st_size == expected:
        print(f"benchmark asset already complete: {name}", flush=True)
        return {**spec, "name": name, "local_path": str(final_path), "local_bytes": expected}
    if existing == expected:
        part_path.replace(final_path)
        print(f"benchmark asset already complete (resuming finalization): {name}", flush=True)
        return {**spec, "name": name, "local_path": str(final_path), "local_bytes": expected}
    if existing > expected:
        raise RuntimeError(f"partial file is larger than expected: {existing} > {expected} for {name}")

    session = requests.Session()
    for attempt in range(1, retries + 1):
        try:
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            with session.get(url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True) as response:
                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                if existing > 0 and not append:
                    existing = 0
                    part_path.unlink(missing_ok=True)
                mode = "ab" if append else "wb"
                downloaded = existing
                with part_path.open(mode) as handle:
                    last_report = downloaded
                    for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 512 * 1024 * 1024:
                            print(f"  {name}: {downloaded / 1024**3:.2f}/{expected / 1024**3:.2f} GiB", flush=True)
                            last_report = downloaded
                if downloaded != expected:
                    raise RuntimeError(f"size mismatch after download: {downloaded} != {expected}")
            part_path.replace(final_path)
            print(f"benchmark asset complete: {name} ({expected / 1024**3:.2f} GiB)", flush=True)
            return {**spec, "name": name, "local_path": str(final_path), "local_bytes": expected}
        except Exception as exc:
            print(f"download attempt {attempt}/{retries} failed for {name}: {exc!r}", file=sys.stderr, flush=True)
            existing = part_path.stat().st_size if part_path.exists() else 0
            if attempt == retries:
                raise
            time.sleep(min(60, 2**attempt))
    raise AssertionError("unreachable")


def download_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    asset_dir = args.benchmark_dir / "assets"
    records = []
    for name, spec in BENCHMARK_ASSETS.items():
        records.append(download_google_asset(name, spec, asset_dir))
    write_json(asset_dir / "REMOTE_MANIFEST.json", {"assets": records})
    return records


def local_inventory(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            files.append({"path": str(path.relative_to(root)), "size": path.stat().st_size})
    return {"root": str(root), "file_count": len(files), "total_bytes": sum(x["size"] for x in files), "files": files}


def verify_anyword(root: Path, manifest: dict[str, Any]) -> None:
    missing = []
    wrong = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.exists():
            missing.append(entry["path"])
        elif entry["size"] is not None and path.stat().st_size != int(entry["size"]):
            wrong.append({"path": entry["path"], "expected": entry["size"], "actual": path.stat().st_size})
    if missing or wrong:
        raise RuntimeError(f"AnyWord verification failed: missing={len(missing)}, wrong_size={len(wrong)}")
    print(f"AnyWord verification passed: {manifest['file_count']} files", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anyword-dir", type=Path, default=DEFAULT_ANYWORD_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--hf-workers", type=int, default=8)
    parser.add_argument("--skip-anyword", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    token = (os.environ.get("HF_TOKEN") or os.environ.get("GLYPH_OMNI_HF_TOKEN", "")).strip()
    if not token and not args.skip_anyword:
        print("GLYPH_OMNI_HF_TOKEN is required for AnyWord-3M", file=sys.stderr)
        return 2
    manifests: dict[str, Any] = {}
    if not args.skip_anyword:
        manifests["anyword"] = download_anyword(args, token)
        verify_anyword(args.anyword_dir, manifests["anyword"])
        write_json(args.anyword_dir / "LOCAL_INVENTORY.json", local_inventory(args.anyword_dir))
    if not args.skip_benchmark:
        manifests["benchmark_assets"] = download_benchmark(args)
        write_json(args.benchmark_dir / "assets" / "LOCAL_INVENTORY.json", local_inventory(args.benchmark_dir / "assets"))
    write_json(args.benchmark_dir / "DOWNLOAD_MANIFEST.json", manifests)
    print("full dataset download completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
