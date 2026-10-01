"""Convert a standard pseudo-Gaussian PLY into compact streaming chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .export_pseudo_gaussian_ply import SH_DC_CONSTANT
except ImportError:
    from export_pseudo_gaussian_ply import SH_DC_CONSTANT


@dataclass(frozen=True)
class GaussianPlyHeader:
    vertex_count: int
    data_offset: int
    stride: int
    offsets: dict[str, int]
    color_kind: str


def compact_dtype() -> np.dtype:
    """Return the 24-byte-per-Gaussian layout consumed by the WebGL viewer."""
    return np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("color", "<u4"),
            ("radius", "<f4"),
            ("opacity", "<f4"),
        ]
    )


def parse_gaussian_ply_header(path: Path) -> GaussianPlyHeader:
    """Parse the fixed binary vertex layout from a little-endian Gaussian PLY."""
    property_sizes = {"float": 4, "float32": 4, "uchar": 1, "uint8": 1}
    vertex_count: int | None = None
    offsets: dict[str, int] = {}
    offset = 0
    current_element: str | None = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"Incomplete PLY header: {path}")
            if raw_line.strip() == b"end_header":
                data_offset = handle.tell()
                break
            line = raw_line.decode("ascii").strip()
            fields = line.split()
            if len(fields) >= 3 and fields[0] == "element":
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
            elif (
                fields
                and fields[0] == "property"
                and len(fields) == 3
                and current_element == "vertex"
            ):
                property_type, name = fields[1:]
                if property_type not in property_sizes:
                    raise ValueError(f"Unsupported Gaussian PLY property type: {property_type}")
                offsets[name] = offset
                offset += property_sizes[property_type]

    if vertex_count is None:
        raise ValueError(f"PLY has no vertex element: {path}")
    required = ("x", "y", "z")
    missing = [name for name in required if name not in offsets]
    if missing:
        raise ValueError(f"Point PLY is missing required properties: {', '.join(missing)}")

    rgb_names = ("red", "green", "blue")
    sh_names = ("f_dc_0", "f_dc_1", "f_dc_2")
    if all(name in offsets for name in rgb_names):
        color_kind = "rgb"
    elif all(name in offsets for name in sh_names):
        color_kind = "sh_dc"
    else:
        raise ValueError("Point PLY has neither RGB nor f_dc color properties")
    return GaussianPlyHeader(vertex_count, data_offset, offset, offsets, color_kind)


def _read_floats(raw: np.ndarray, name: str, offsets: dict[str, int], count: int) -> np.ndarray:
    column = np.ascontiguousarray(raw[:, offsets[name] : offsets[name] + 4], dtype=np.uint8)
    return column.reshape(-1).view("<f4").copy()


def _read_compact_chunk(
    raw: np.ndarray,
    header: GaussianPlyHeader,
    radius_override: float | None,
    opacity_override: float | None,
    flip_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(raw)
    vertices = np.empty(count, dtype=compact_dtype())
    offsets = header.offsets

    for axis in ("x", "y", "z"):
        vertices[axis] = _read_floats(raw, axis, offsets, count)
    if flip_y:
        vertices["y"] = -vertices["y"]
        vertices["z"] = -vertices["z"]

    rgb = np.empty((count, 3), dtype=np.uint8)
    if header.color_kind == "rgb":
        rgb[:, 0] = raw[:, offsets["red"]]
        rgb[:, 1] = raw[:, offsets["green"]]
        rgb[:, 2] = raw[:, offsets["blue"]]
    else:
        for channel, name in enumerate(("f_dc_0", "f_dc_1", "f_dc_2")):
            encoded = raw[:, offsets[name]].astype(np.float64) / 255.0
            sh_dc = encoded * 2.0 - 1.0
            color = np.clip(0.5 + SH_DC_CONSTANT * sh_dc, 0.0, 1.0)
            rgb[:, channel] = np.rint(color * 255.0).astype(np.uint8)

    vertices["color"] = (
        rgb[:, 0].astype(np.uint32)
        | rgb[:, 1].astype(np.uint32) << 8
        | rgb[:, 2].astype(np.uint32) << 16
        | 0xFF000000
    )

    if radius_override is not None:
        vertices["radius"] = radius_override
    else:
        scale_names = ("scale_0", "scale_1", "scale_2")
        available = [name for name in scale_names if name in offsets]
        if not available:
            raise ValueError("Gaussian PLY has no scale properties")
        radius = None
        for name in available:
            scale = np.exp(_read_floats(raw, name, offsets, count))
            radius = scale if radius is None else np.maximum(radius, scale)
        vertices["radius"] = radius

    if opacity_override is not None:
        vertices["opacity"] = opacity_override
    elif "opacity" in offsets:
        opacity = _read_floats(raw, "opacity", offsets, count)
        vertices["opacity"] = 1.0 / (1.0 + np.exp(-opacity))
    else:
        raise ValueError("Point PLY has no opacity property; pass --opacity")
    bounds_min = np.asarray([np.min(vertices[axis]) for axis in ("x", "y", "z")])
    bounds_max = np.asarray([np.max(vertices[axis]) for axis in ("x", "y", "z")])
    return vertices, bounds_min, bounds_max


def _bounds_to_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values)]


def export_chunks(
    input_path: Path,
    output_dir: Path,
    chunk_size: int = 250_000,
    radius_override: float | None = None,
    opacity_override: float | None = None,
    flip_y: bool = False,
) -> dict[str, Any]:
    """Split a Gaussian PLY into compact chunks and write a JSON manifest."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if radius_override is not None and radius_override <= 0:
        raise ValueError("radius_override must be positive")
    if opacity_override is not None and not 0.0 < opacity_override < 1.0:
        raise ValueError("opacity_override must be between 0 and 1, exclusive")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    header = parse_gaussian_ply_header(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.memmap(
        input_path,
        dtype=np.uint8,
        mode="r",
        offset=header.data_offset,
        shape=(header.vertex_count, header.stride),
    )

    chunk_count = (header.vertex_count + chunk_size - 1) // chunk_size
    chunks: list[dict[str, Any]] = []
    global_min = np.asarray([np.inf, np.inf, np.inf], dtype=np.float64)
    global_max = np.asarray([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        stop = min(start + chunk_size, header.vertex_count)
        vertices, chunk_min, chunk_max = _read_compact_chunk(
            raw[start:stop],
            header,
            radius_override,
            opacity_override,
            flip_y,
        )
        file_name = f"chunk-{chunk_index:05d}.pbin"
        vertices.tofile(output_dir / file_name)

        global_min = np.minimum(global_min, chunk_min)
        global_max = np.maximum(global_max, chunk_max)
        chunks.append(
            {
                "file": file_name,
                "offset": start,
                "count": stop - start,
                "bounds": {
                    "min": _bounds_to_list(chunk_min),
                    "max": _bounds_to_list(chunk_max),
                },
            }
        )

    manifest: dict[str, Any] = {
        "format": "abot-pseudo-gaussian-chunks",
        "version": 1,
        "vertex_count": header.vertex_count,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "stride": compact_dtype().itemsize,
        "bounds": {"min": _bounds_to_list(global_min), "max": _bounds_to_list(global_max)},
        "chunks": chunks,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pseudo-Gaussian PLY to compact streaming chunks."
    )
    parser.add_argument("--input", type=Path, required=True, help="Standard 3DGS PLY input")
    parser.add_argument("--output-dir", type=Path, required=True, help="Chunk output directory")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--radius", type=float, default=None, help="Optional radius override in meters"
    )
    parser.add_argument(
        "--opacity", type=float, default=None, help="Optional opacity override in (0, 1)"
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Convert camera-style Y-down/Z-forward coordinates to Y-up viewer coordinates",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = export_chunks(
        args.input,
        args.output_dir,
        chunk_size=args.chunk_size,
        radius_override=args.radius,
        opacity_override=args.opacity,
        flip_y=args.flip_y,
    )
    total_bytes = manifest["stride"] * manifest["vertex_count"]
    print(
        f"Wrote {args.output_dir}: {manifest['chunk_count']} chunks, "
        f"{manifest['vertex_count']:,} Gaussians, {total_bytes / 1_000_000:.1f} MB compact data"
    )


if __name__ == "__main__":
    main()
