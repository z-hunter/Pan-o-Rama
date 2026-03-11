#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FACE_KEYS = ("front", "back", "left", "right", "top", "down")


def load_face(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def build_maps(width: int, height: int, start_row: int, rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    ys = (np.arange(start_row, start_row + rows, dtype=np.float32) + 0.5) / height

    theta = (xs[None, :] - 0.5) * (2.0 * math.pi)
    phi = (0.5 - ys[:, None]) * math.pi

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def face_uv(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    is_x_major = (abs_x >= abs_y) & (abs_x >= abs_z)
    is_y_major = (abs_y > abs_x) & (abs_y >= abs_z)
    is_z_major = ~(is_x_major | is_y_major)

    masks = {
        "right": is_x_major & (x > 0),
        "left": is_x_major & (x <= 0),
        "top": is_y_major & (y > 0),
        "down": is_y_major & (y <= 0),
        "front": is_z_major & (z > 0),
        "back": is_z_major & (z <= 0),
    }

    us = {
        "front": x / abs_z,
        "back": -x / abs_z,
        "right": -z / abs_x,
        "left": z / abs_x,
        "top": x / abs_y,
        "down": x / abs_y,
    }
    vs = {
        "front": -y / abs_z,
        "back": -y / abs_z,
        "right": -y / abs_x,
        "left": -y / abs_x,
        "top": z / abs_y,
        "down": -z / abs_y,
    }
    return masks, us, vs


def remap_face(face: np.ndarray, u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    size = face.shape[0]
    map_x = np.zeros(u.shape, dtype=np.float32)
    map_y = np.zeros(v.shape, dtype=np.float32)
    map_x[mask] = (u[mask] + 1.0) * 0.5 * (size - 1)
    map_y[mask] = (v[mask] + 1.0) * 0.5 * (size - 1)
    remapped = cv2.remap(face, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return remapped


def convert(faces_dir: Path, output_path: Path, width: int, chunk_rows: int) -> None:
    face_paths = {key: faces_dir / f"{key}.tif" for key in FACE_KEYS}
    missing = [key for key, path in face_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cubemap faces: {', '.join(missing)}")

    faces = {key: load_face(path) for key, path in face_paths.items()}
    sizes = {key: faces[key].shape[:2] for key in FACE_KEYS}
    unique_sizes = {size for size in sizes.values()}
    if len(unique_sizes) != 1:
        raise ValueError(f"All faces must be the same size, got: {sizes}")

    face_h, face_w = next(iter(unique_sizes))
    if face_h != face_w:
        raise ValueError(f"Cubemap faces must be square, got {face_w}x{face_h}")

    height = width // 2
    output = np.zeros((height, width, 3), dtype=np.uint8)

    for start_row in range(0, height, chunk_rows):
        rows = min(chunk_rows, height - start_row)
        x, y, z = build_maps(width, height, start_row, rows)
        masks, us, vs = face_uv(x, y, z)
        chunk = np.zeros((rows, width, 3), dtype=np.uint8)
        for key in FACE_KEYS:
            mask = masks[key]
            if not np.any(mask):
                continue
            remapped = remap_face(faces[key], us[key], vs[key], mask)
            chunk[mask] = remapped[mask]
        output[start_row:start_row + rows] = chunk
        print(f"Processed rows {start_row}-{start_row + rows - 1} / {height - 1}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(output)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95, subsampling=0)
    else:
        image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cubemap TIFF faces into one equirectangular panorama.")
    parser.add_argument("--faces-dir", type=Path, required=True, help="Directory containing front/back/left/right/top/down .tif files.")
    parser.add_argument("--output", type=Path, required=True, help="Output panorama path (.jpg, .png, .tif).")
    parser.add_argument("--width", type=int, default=8192, help="Output width. Height is width/2. Default: 8192.")
    parser.add_argument("--chunk-rows", type=int, default=256, help="Rows per processing chunk to limit RAM usage.")
    args = parser.parse_args()

    if args.width < 1024 or args.width % 2 != 0:
        raise SystemExit("--width must be an even integer >= 1024")

    convert(args.faces_dir, args.output, args.width, args.chunk_rows)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
