import numpy as np
import pytest

from scripts.export_point_cloud_chunks import (
    parse_rgb_ply_header,
    point_budget_indices,
    read_rgb_ply,
    rgb_ply_dtype,
)


def make_rgb_ply(path, points, colors, *, fmt="binary_little_endian", extra_edge=False):
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ]
    )
    vertices = np.zeros(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["r"], vertices["g"], vertices["b"] = colors.T
    properties = (
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
    )
    edge = b"element edge 0\nproperty int vertex1\nproperty int vertex2\n" if extra_edge else b""
    header = (
        b"ply\nformat " + fmt.encode("ascii") + b" 1.0\n" +
        f"element vertex {len(vertices)}\n".encode("ascii") +
        properties + edge + b"end_header\n"
    )
    path.write_bytes(header + vertices.tobytes())


def test_parse_rgb_ply_header_reads_vertex_layout_and_ignores_edges(tmp_path):
    path = tmp_path / "points.ply"
    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    make_rgb_ply(path, points, colors, extra_edge=True)

    header = parse_rgb_ply_header(path)

    assert header.vertex_count == 2
    assert header.offsets == {"x": 0, "y": 4, "z": 8, "red": 12, "green": 13, "blue": 14}
    assert header.stride == 15
    assert header.data_offset > 0
    assert rgb_ply_dtype(header.offsets).itemsize == 15


def test_read_rgb_ply_returns_points_and_colors_and_filters_nonfinite(tmp_path):
    path = tmp_path / "points.ply"
    points = np.array(
        [[0.0, 1.0, 2.0], [np.nan, 4.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    colors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    make_rgb_ply(path, points, colors)

    parsed_points, parsed_colors = read_rgb_ply(path)

    assert parsed_points.dtype == np.float32
    assert parsed_colors.dtype == np.uint8
    assert np.array_equal(parsed_points, points[[0, 2]])
    assert np.array_equal(parsed_colors, colors[[0, 2]])


def test_parse_rgb_ply_header_rejects_unsupported_formats(tmp_path):
    path = tmp_path / "ascii.ply"
    path.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 0\n"
        b"property float x\nproperty uchar red\nend_header\n"
    )

    with pytest.raises(ValueError, match="Only binary_little_endian"):
        parse_rgb_ply_header(path)


def test_parse_rgb_ply_header_rejects_list_properties(tmp_path):
    path = tmp_path / "list.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
        b"property list int float x\nend_header\n"
    )

    with pytest.raises(ValueError, match="List properties are not supported"):
        parse_rgb_ply_header(path)


def test_point_budget_indices_retains_or_evenly_samples():
    assert np.array_equal(point_budget_indices(3, 0), [0, 1, 2])
    assert np.array_equal(point_budget_indices(0, 10), np.empty(0, dtype=np.int64))
    assert np.array_equal(point_budget_indices(1, 4), [0])
    assert np.array_equal(point_budget_indices(5, 3), [0, 2, 4])
