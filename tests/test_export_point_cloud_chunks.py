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


def test_rgb_ply_handles_trailing_ignored_scalar_vertex_property(tmp_path):
    path = tmp_path / "points-with-extra-property.ply"
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("quality", "<u2"),
        ]
    )
    vertices = np.array(
        [
            (0.0, 1.0, 2.0, 10, 20, 30, 111),
            (3.0, 4.0, 5.0, 40, 50, 60, 222),
        ],
        dtype=vertex_dtype,
    )
    path.write_bytes(
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 2\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"property ushort quality\n"
        b"end_header\n" + vertices.tobytes()
    )

    header = parse_rgb_ply_header(path)

    assert header.offsets == {
        "x": 0,
        "y": 4,
        "z": 8,
        "red": 12,
        "green": 13,
        "blue": 14,
        "quality": 15,
    }
    assert header.stride == 17
    assert rgb_ply_dtype(header.offsets).itemsize == 15

    points, colors = read_rgb_ply(path, header)

    assert np.array_equal(
        points, np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    )
    assert np.array_equal(
        colors, np.column_stack([vertices["red"], vertices["green"], vertices["blue"]])
    )


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

from scripts.export_point_cloud_chunks import build_trajectory, load_camera_poses, robust_bounds


def test_robust_bounds_uses_median_center_and_high_distance_percentile():
    inliers = np.column_stack((
        np.arange(100, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
        np.zeros(100, dtype=np.float32),
    ))
    points = np.vstack((inliers, [[1000.0, 0.0, 0.0]]), dtype=np.float32)
    result = robust_bounds(points, sample_limit=1000)
    assert result["center"] == [50.0, 0.0, 0.0]
    assert 47.0 <= result["radius"] <= 50.0


def test_load_camera_poses_validates_shape_and_finite_values(tmp_path):
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    path = tmp_path / "camera_poses.npy"
    np.save(path, poses)

    assert np.array_equal(load_camera_poses(path), poses)

    bad = poses.copy()
    bad[1, 0, 0] = np.nan
    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, bad)
    with pytest.raises(ValueError, match="finite"):
        load_camera_poses(bad_path)


def test_build_trajectory_returns_centers_and_normalized_optical_forwards():
    poses = np.zeros((2, 4, 4), dtype=np.float64)
    poses[:, 3, 3] = 1.0
    poses[:, :3, 3] = [[0, 0, 0], [2, 0, 0]]
    poses[:, :3, 2] = [0, 2, 0]

    result = build_trajectory(poses)

    assert result["format"] == "abot-point-cloud-trajectory"
    assert result["version"] == 1
    assert result["coordinate_system"] == "original ABot-Recon world coordinates"
    assert result["positions"] == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert result["forwards"] == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
