"""Resume a large HTTP download with parallel byte ranges and MD5 checking."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def download_range(url: str, path: Path, start: int, end: int) -> int:
    expected = end - start + 1
    existing = path.stat().st_size if path.exists() else 0
    if existing > expected:
        raise RuntimeError(f"{path} is larger than its assigned byte range")
    if existing == expected:
        return expected

    request_start = start + existing
    headers = {
        "Range": f"bytes={request_start}-{end}",
        "User-Agent": "Mozilla/5.0 dataset-downloader/1.0",
    }
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if response.status_code != 206:
            raise RuntimeError(
                f"Server did not honor range {request_start}-{end}: "
                f"HTTP {response.status_code}"
            )
        with path.open("ab") as output:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    output.write(chunk)

    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Incomplete segment {path}: {actual} != {expected}")
    return actual


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    output = args.output.resolve()
    prefix = output.with_suffix(output.suffix + ".part")
    assembling = output.with_suffix(output.suffix + ".assembling")

    if output.exists():
        if output.stat().st_size == args.size and file_md5(output) == args.md5:
            print(f"Already complete: {output}")
            return
        raise RuntimeError(f"Refusing to overwrite existing invalid file: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    prefix_size = prefix.stat().st_size if prefix.exists() else 0
    if prefix_size > args.size:
        raise RuntimeError(f"Partial file is larger than expected: {prefix}")

    remaining = args.size - prefix_size
    if remaining:
        worker_count = min(args.workers, remaining)
        base_size, extra = divmod(remaining, worker_count)
        ranges = []
        cursor = prefix_size
        for index in range(worker_count):
            length = base_size + (1 if index < extra else 0)
            start = cursor
            end = cursor + length - 1
            segment = output.with_suffix(output.suffix + f".seg{index:02d}")
            ranges.append((segment, start, end))
            cursor = end + 1

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(download_range, args.url, segment, start, end): segment
                for segment, start, end in ranges
            }
            for future in as_completed(futures):
                size = future.result()
                print(f"Completed {futures[future].name}: {size} bytes", flush=True)
    else:
        ranges = []

    with assembling.open("wb") as destination:
        if prefix.exists():
            with prefix.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        for segment, _, _ in ranges:
            with segment.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)

    if assembling.stat().st_size != args.size:
        raise RuntimeError(
            f"Assembled size mismatch: {assembling.stat().st_size} != {args.size}"
        )
    actual_md5 = file_md5(assembling)
    if actual_md5.lower() != args.md5.lower():
        raise RuntimeError(f"MD5 mismatch: {actual_md5} != {args.md5}")

    os.replace(assembling, output)
    if prefix.exists():
        prefix.unlink()
    for segment, _, _ in ranges:
        if segment.exists():
            segment.unlink()
    print(f"Verified download: {output}")


if __name__ == "__main__":
    main()
