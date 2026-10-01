import json
from pathlib import Path

import numpy as np

from scripts.export_pseudo_gaussian_chunks import (
    compact_dtype,
    export_chunks,
    parse_gaussian_ply_header,
)
from scripts.export_pseudo_gaussian_ply import (
    gaussian_vertex_data,
    write_binary_gaussian_ply,
)


def make_gaussian_ply(path: Path) -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    vertices = gaussian_vertex_data(points, colors, radius=0.25, opacity=0.8)
    write_binary_gaussian_ply(path, vertices)


def make_rgb_point_ply(path: Path) -> None:
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    points = np.zeros(2, dtype=dtype)
    points["x"] = [1.0, 2.0]
    points["y"] = [3.0, 4.0]
    points["z"] = [5.0, 6.0]
    points["r"] = [10, 20]
    points["g"] = [30, 40]
    points["b"] = [50, 60]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + points.tobytes())


def make_rgb_point_ply_with_edge_element(path: Path) -> None:
    make_rgb_point_ply(path)
    header, payload = path.read_bytes().split(b"end_header\n", 1)
    header += (
        b"element edge 0\n"
        b"property int vertex1\n"
        b"property int vertex2\n"
        b"end_header\n"
    )
    path.write_bytes(header + payload)


def test_compact_dtype_uses_24_byte_gpu_layout():
    dtype = compact_dtype()

    assert dtype.itemsize == 24
    assert dtype.names == ("x", "y", "z", "color", "radius", "opacity")
    assert dtype["x"].base == np.dtype("<f4")
    assert dtype["color"].base == np.dtype("<u4")
    assert dtype["radius"].base == np.dtype("<f4")
    assert dtype["opacity"].base == np.dtype("<f4")


def test_parse_gaussian_ply_header_reads_count_and_offsets(tmp_path):
    path = tmp_path / "input.ply"
    make_gaussian_ply(path)
    header = parse_gaussian_ply_header(path)

    assert header.vertex_count == 3
    assert header.stride == 104
    assert header.data_offset > 0
    assert header.offsets["x"] == 0
    assert header.offsets["f_dc_0"] == 24
    assert header.offsets["opacity"] == 72
    assert header.offsets["rot_3"] == 100


def test_export_chunks_writes_manifest_and_compact_records(tmp_path):
    input_path = tmp_path / "input.ply"
    output_dir = tmp_path / "chunks"
    make_gaussian_ply(input_path)

    manifest = export_chunks(input_path, output_dir, chunk_size=2)

    assert manifest["format"] == "abot-pseudo-gaussian-chunks"
    assert manifest["version"] == 1
    assert manifest["vertex_count"] == 3
    assert manifest["chunk_size"] == 2
    assert manifest["chunk_count"] == 2
    assert manifest["stride"] == 24
    assert len(manifest["chunks"]) == 2
    assert manifest["chunks"][0]["count"] == 2
    assert manifest["chunks"][1]["count"] == 1
    assert np.allclose(manifest["bounds"]["min"], [0.0, 0.0, 0.0])
    assert np.allclose(manifest["bounds"]["max"], [4.0, 5.0, 6.0])

    manifest_on_disk = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_on_disk == manifest
    first = np.fromfile(output_dir / manifest["chunks"][0]["file"], dtype=compact_dtype())
    second = np.fromfile(output_dir / manifest["chunks"][1]["file"], dtype=compact_dtype())
    assert len(first) == 2
    assert len(second) == 1
    assert np.allclose(first["x"], [0.0, 1.0])
    assert np.allclose(first["radius"], 0.25, atol=1e-6)
    assert np.allclose(first["opacity"], 0.8, atol=1e-6)

    raw = first.view(np.uint8).reshape(-1, compact_dtype().itemsize)
    assert raw[0, 12] > 180
    assert raw[0, 13] < 80


def test_export_chunks_preserves_rgb_point_colors_and_can_flip_y(tmp_path):
    input_path = tmp_path / "points.ply"
    output_dir = tmp_path / "rgb-chunks"
    make_rgb_point_ply(input_path)

    manifest = export_chunks(
        input_path,
        output_dir,
        chunk_size=2,
        radius_override=0.2,
        opacity_override=0.75,
        flip_y=True,
    )

    data = np.fromfile(
        output_dir / manifest["chunks"][0]["file"], dtype=compact_dtype()
    )
    assert np.allclose(data["x"], [1.0, 2.0])
    assert np.allclose(data["y"], [-3.0, -4.0])
    assert np.allclose(data["z"], [-5.0, -6.0])
    assert np.allclose(data["radius"], 0.2)
    assert np.allclose(data["opacity"], 0.75)
    raw = data.view(np.uint8).reshape(-1, compact_dtype().itemsize)
    assert np.array_equal(raw[0, 12:16], [10, 30, 50, 255])
    assert np.array_equal(raw[1, 12:16], [20, 40, 60, 255])


def test_parse_gaussian_ply_header_ignores_non_vertex_elements(tmp_path):
    path = tmp_path / "points-with-edge.ply"
    make_rgb_point_ply_with_edge_element(path)

    header = parse_gaussian_ply_header(path)

    assert header.vertex_count == 2
    assert header.color_kind == "rgb"
    assert "vertex1" not in header.offsets
