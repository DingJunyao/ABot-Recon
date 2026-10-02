"""Convert an RGB reconstruction PLY into compact streamed point chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PLY_TYPE_SIZES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}

_AXIS_TYPES = frozenset(("float", "float32"))
_COLOR_TYPES = frozenset(("uchar", "uint8"))


@dataclass(frozen=True)
class RgbPlyHeader:
    vertex_count: int
    data_offset: int
    stride: int
    offsets: dict[str, int]


def rgb_ply_dtype(offsets: dict[str, int]) -> np.dtype:
    """Return the packed dtype containing the six required RGB point fields."""
    fields = [
        ("x", "<f4", offsets["x"]),
        ("y", "<f4", offsets["y"]),
        ("z", "<f4", offsets["z"]),
        ("red", "u1", offsets["red"]),
        ("green", "u1", offsets["green"]),
        ("blue", "u1", offsets["blue"]),
    ]
    return np.dtype(
        {
            "names": [name for name, _, _ in fields],
            "formats": [format for _, format, _ in fields],
            "offsets": [offset for _, _, offset in fields],
            "itemsize": max(
                offset + np.dtype(format).itemsize for _, format, offset in fields
            ),
        }
    )


def parse_rgb_ply_header(path: Path) -> RgbPlyHeader:
    """Parse a binary-little-endian RGB PLY vertex layout.

    Properties outside the vertex element do not affect the vertex stride. Scalar
    properties not consumed by the viewer still advance the stride so later rows
    remain aligned.
    """
    vertex_count: int | None = None
    offsets: dict[str, int] = {}
    property_types: dict[str, str] = {}
    offset = 0
    current_element: str | None = None
    format_seen = False
    data_offset: int | None = None

    with Path(path).open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != b"ply":
            raise ValueError(f"Not a PLY file: {path}")

        while data_offset is None:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"Incomplete PLY header: {path}")
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"Incomplete PLY header: {path}")

            try:
                line = raw_line.decode("ascii").rstrip("\r\n")
            except UnicodeDecodeError as error:
                raise ValueError(f"Non-ASCII PLY header: {path}") from error
            fields = line.split()

            if not fields:
                continue
            if fields[0] == "format":
                if len(fields) != 3 or fields[1:] != ["binary_little_endian", "1.0"]:
                    raise ValueError(
                        f"Only binary_little_endian 1.0 PLY files are supported: {path}"
                    )
                format_seen = True
            elif fields[0] == "element":
                if len(fields) != 3:
                    raise ValueError(f"Malformed PLY element line: {line}")
                if current_element is not None and fields[1] == "vertex":
                    raise ValueError(f"PLY has more than one vertex element: {path}")
                if current_element is not None and vertex_count is None:
                    raise ValueError(f"Vertex element must precede other elements: {path}")
                current_element = fields[1]
                if current_element == "vertex":
                    try:
                        vertex_count = int(fields[2])
                    except ValueError as error:
                        raise ValueError(f"Invalid PLY vertex count: {fields[2]}") from error
                    if vertex_count < 0:
                        raise ValueError("PLY vertex count must be non-negative")
            elif fields[0] == "property":
                if current_element != "vertex":
                    continue
                if len(fields) >= 2 and fields[1] == "list":
                    raise ValueError(f"List properties are not supported: {path}")
                if len(fields) != 3:
                    raise ValueError(f"Malformed PLY property line: {line}")
                property_type, name = fields[1:]
                if property_type not in PLY_TYPE_SIZES:
                    raise ValueError(f"Unsupported PLY property type: {property_type}")
                offsets[name] = offset
                property_types[name] = property_type
                offset += PLY_TYPE_SIZES[property_type]
            elif fields[0] == "end_header":
                data_offset = handle.tell()

    if not format_seen:
        raise ValueError(f"Missing PLY format line: {path}")
    if vertex_count is None:
        raise ValueError(f"PLY has no vertex element: {path}")

    required = ("x", "y", "z", "red", "green", "blue")
    missing = [name for name in required if name not in offsets]
    if missing:
        raise ValueError(f"RGB PLY is missing required properties: {', '.join(missing)}")

    for name in ("x", "y", "z"):
        if property_types[name] not in _AXIS_TYPES:
            raise ValueError(f"PLY property {name} must be a 32-bit float")
    for name in ("red", "green", "blue"):
        if property_types[name] not in _COLOR_TYPES:
            raise ValueError(f"PLY property {name} must be an unsigned byte")

    return RgbPlyHeader(vertex_count, data_offset, offset, offsets)


def _map_vertices(path: Path, header: RgbPlyHeader) -> np.ndarray:
    """Map required fields while honoring any ignored vertex properties."""
    dtype = rgb_ply_dtype(header.offsets)
    if dtype.itemsize == header.stride:
        return np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=header.data_offset,
            shape=(header.vertex_count,),
        )

    raw = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=header.data_offset,
        shape=(header.vertex_count, header.stride),
    )
    return np.ndarray(
        raw.shape[:1],
        dtype=dtype,
        buffer=raw,
        strides=(header.stride,),
    )


def read_rgb_ply(
    path: Path, header: RgbPlyHeader | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read finite XYZ points and RGB colors from a binary RGB PLY."""
    header = parse_rgb_ply_header(path) if header is None else header
    vertices = _map_vertices(Path(path), header)

    points = np.empty((header.vertex_count, 3), dtype=np.float32)
    points[:, 0] = vertices["x"]
    points[:, 1] = vertices["y"]
    points[:, 2] = vertices["z"]
    colors = np.empty((header.vertex_count, 3), dtype=np.uint8)
    colors[:, 0] = vertices["red"]
    colors[:, 1] = vertices["green"]
    colors[:, 2] = vertices["blue"]

    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def robust_bounds(points: np.ndarray, sample_limit: int = 20000) -> dict[str, list[float]]:
    """Return outlier-resistant framing bounds for an XYZ point array."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have XYZ columns")
    finite_points = points[np.isfinite(points).all(axis=1)]
    if len(finite_points) == 0:
        raise ValueError("No finite XYZ points for robust bounds")

    if len(finite_points) > sample_limit:
        indices = np.linspace(
            0, len(finite_points) - 1, sample_limit, dtype=np.int64
        )
        finite_points = finite_points[indices]

    center = np.median(finite_points, axis=0)
    distances = np.linalg.norm(finite_points - center, axis=1)
    radius = max(float(np.percentile(distances, 96)), 1e-3)
    return {
        "center": [float(value) for value in center],
        "radius": radius,
    }


def load_camera_poses(path: Path) -> np.ndarray:
    """Load and validate an [N, 4, 4] finite camera-pose array."""
    poses = np.load(Path(path), allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("camera poses must have shape [N, 4, 4]")
    if not np.isfinite(poses).all():
        raise ValueError("camera poses must contain only finite values")
    return poses


def build_trajectory(poses: np.ndarray, fps: float | None = None) -> dict[str, Any]:
    """Convert camera centers and optical axes to a JSON-safe trajectory."""
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("camera poses must have shape [N, 4, 4]")

    centers = poses[:, :3, 3]
    forwards = poses[:, :3, 2]
    lengths = np.linalg.norm(forwards, axis=1)
    valid = (
        np.isfinite(centers).all(axis=1)
        & np.isfinite(forwards).all(axis=1)
        & np.isfinite(lengths)
        & (lengths > 0)
    )

    trajectory = {
        "format": "abot-point-cloud-trajectory",
        "version": 1,
        "coordinate_system": "original ABot-Recon world coordinates",
        "frame_count": int(valid.sum()),
        "positions": centers[valid].astype(float, copy=False).tolist(),
        "forwards": (forwards[valid] / lengths[valid, None]).tolist(),
    }
    if fps is not None and np.isfinite(fps) and fps > 0:
        trajectory["fps"] = float(fps)
    return trajectory


def point_budget_indices(point_count: int, max_points: int) -> np.ndarray:
    """Retain every point, or evenly sample a bounded point sequence."""
    if max_points < 0:
        raise ValueError("max_points must be non-negative")
    if max_points == 0 or point_count <= max_points:
        return np.arange(point_count, dtype=np.int64)
    if max_points == 1:
        return np.array([point_count // 2], dtype=np.int64)
    return np.linspace(0, point_count - 1, max_points, dtype=np.int64)


def compact_dtype(position_encoding: str = "uint16") -> np.dtype:
    """Return the on-disk record layout for a compact point chunk."""
    if position_encoding == "uint16":
        return np.dtype([("position", "<u2", (3,)), ("color", "<u4")])
    if position_encoding == "float32":
        return np.dtype([("position", "<f4", (3,)), ("color", "<u4")])
    raise ValueError(f"Unsupported position encoding: {position_encoding}")


def _bounds_manifest(points: np.ndarray) -> dict[str, list[float]]:
    """Return JSON-safe source-coordinate bounds for finite XYZ points."""
    return {
        "min": points.min(axis=0).astype(np.float64, copy=False).tolist(),
        "max": points.max(axis=0).astype(np.float64, copy=False).tolist(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, pretty UTF-8 JSON."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_point_cloud_chunks(
    input_path: Path,
    output_dir: Path,
    poses_path: Path | None = None,
    chunk_size: int = 250_000,
    max_points: int = 0,
    position_encoding: str = "uint16",
    fps: float | None = None,
) -> dict[str, Any]:
    """Convert a finite RGB PLY into compact chunks and a viewing manifest."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_points < 0:
        raise ValueError("max_points must be non-negative")
    dtype = compact_dtype(position_encoding)

    points, colors = read_rgb_ply(Path(input_path))
    selected = point_budget_indices(len(points), max_points)
    points = points[selected]
    colors = colors[selected]
    if len(points) == 0:
        raise ValueError("Input point cloud has no finite XYZ points")

    trajectory = None
    if poses_path is not None:
        trajectory = build_trajectory(load_camera_poses(Path(poses_path)), fps=fps)
        if not trajectory["positions"]:
            trajectory = None

    bounds_min = points.min(axis=0).astype(np.float64, copy=False)
    bounds_max = points.max(axis=0).astype(np.float64, copy=False)
    scale = np.maximum(bounds_max - bounds_min, np.finfo(np.float64).eps)
    packed_colors = (
        colors[:, 0].astype(np.uint32)
        | (colors[:, 1].astype(np.uint32) << np.uint32(8))
        | (colors[:, 2].astype(np.uint32) << np.uint32(16))
        | np.uint32(0xFF000000)
    )

    chunks: list[dict[str, Any]] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    for chunk_index, start in enumerate(range(0, len(points), chunk_size)):
        stop = min(start + chunk_size, len(points))
        chunk_points = points[start:stop]

        records = np.empty(stop - start, dtype=dtype)
        if position_encoding == "uint16":
            records["position"] = np.rint(
                (chunk_points.astype(np.float64, copy=False) - bounds_min) / scale * 65535.0
            ).astype("<u2", copy=False)
        else:
            records["position"] = chunk_points.astype("<f4", copy=False)
        records["color"] = packed_colors[start:stop]

        chunk_file = f"chunk-{chunk_index:05d}.pbin"
        records.tofile(output_dir / chunk_file)
        chunks.append(
            {
                "file": chunk_file,
                "offset": offset,
                "count": stop - start,
                "bounds": _bounds_manifest(chunk_points),
            }
        )
        offset += stop - start

    manifest: dict[str, Any] = {
        "format": "abot-point-cloud-chunks",
        "version": 1,
        "point_count": len(points),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "position_encoding": position_encoding,
        "stride": dtype.itemsize,
        "bounds": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
        },
        "robust_bounds": robust_bounds(points),
        "chunks": chunks,
    }

    if trajectory is not None:
        _write_json(output_dir / "trajectory.json", trajectory)
        manifest["trajectory"] = "trajectory.json"
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse converter command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input binary RGB PLY")
    parser.add_argument("--poses", type=Path, help="optional [N,4,4] camera poses")
    parser.add_argument("--fps", type=float, help="source frames per second for real-time playback")
    parser.add_argument("--output-dir", required=True, type=Path, help="output directory")
    parser.add_argument("--chunk-size", type=int, default=250_000, help="points per chunk")
    parser.add_argument(
        "--max-points", type=int, default=0, help="maximum retained points (0 means all)"
    )
    parser.add_argument(
        "--position-encoding",
        choices=("uint16", "float32"),
        default="uint16",
        help="XYZ storage layout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the point-cloud chunk exporter and print a compact summary."""
    args = parse_args(argv)
    manifest = export_point_cloud_chunks(
        args.input,
        args.output_dir,
        poses_path=args.poses,
        chunk_size=args.chunk_size,
        max_points=args.max_points,
        position_encoding=args.position_encoding,
        fps=args.fps,
    )
    compact_mb = manifest["point_count"] * manifest["stride"] / 1_000_000
    trajectory_written = "yes" if "trajectory" in manifest else "no"
    print(
        f"Wrote {manifest['chunk_count']} chunks, {manifest['point_count']} points, "
        f"{compact_mb:.2f} MB compact; trajectory: {trajectory_written}"
    )


if __name__ == "__main__":
    main()
