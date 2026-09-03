"""Audit image-level development and held-out splits without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from scipy.fft import dctn


ImageFile.LOAD_TRUNCATED_IMAGES = True
POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--held-out-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _hash_bits(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def _phash(image: Image.Image, crop: bool = False) -> int:
    if crop:
        width, height = image.size
        image = image.crop((int(width * 0.08), int(height * 0.08), int(width * 0.92), int(height * 0.92)))
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(gray, dtype=np.float32), norm="ortho")[:8, :8]
    flattened = coefficients.ravel()
    bits = flattened > np.median(flattened[1:])
    bits[0] = False
    return _hash_bits(bits)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _collect(root: Path, split: str) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.glob("*/*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(path) as image:
            image.load()
            records.append({
                "split": split,
                "class": path.parent.name,
                "path": str(path),
                "sha256": _digest(path),
                "phash": _phash(image),
                "crop_phash": _phash(image, crop=True),
            })
    return records


def _hamming(block: np.ndarray, references: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(block[:, None], references[None, :])
    return POPCOUNT[xor.view(np.uint8)].reshape(len(block), len(references), 8).sum(axis=2)


def main() -> None:
    args = parse_args()
    development = _collect(args.development_root, "development")
    held_out = _collect(args.held_out_root, "held_out")
    sha_to_development = defaultdict(list)
    for index, record in enumerate(development):
        sha_to_development[record["sha256"]].append(index)
    exact_duplicates = sum(len(sha_to_development[record["sha256"]]) for record in held_out)

    development_hashes = np.array([record["phash"] for record in development], dtype=np.uint64)
    development_crop_hashes = np.array([record["crop_phash"] for record in development], dtype=np.uint64)
    near_counts = {"both_phash_le_4": 0, "both_phash_le_6": 0}
    for start in range(0, len(held_out), 64):
        block = held_out[start:start + 64]
        full = _hamming(np.array([record["phash"] for record in block], dtype=np.uint64), development_hashes)
        cropped = _hamming(np.array([record["crop_phash"] for record in block], dtype=np.uint64), development_crop_hashes)
        nearest = np.argmin(full.astype(np.uint16) + cropped.astype(np.uint16), axis=1)
        for offset, index in enumerate(nearest):
            full_distance = int(full[offset, index])
            crop_distance = int(cropped[offset, index])
            near_counts["both_phash_le_4"] += full_distance <= 4 and crop_distance <= 4
            near_counts["both_phash_le_6"] += full_distance <= 6 and crop_distance <= 6

    result = {
        "development_images": len(development),
        "held_out_images": len(held_out),
        "total_images": len(development) + len(held_out),
        "exact_cross_split_duplicates": exact_duplicates,
        "near_pair_threshold_counts": near_counts,
        "development_counts_by_class": Counter(record["class"] for record in development),
        "held_out_counts_by_class": Counter(record["class"] for record in held_out),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
