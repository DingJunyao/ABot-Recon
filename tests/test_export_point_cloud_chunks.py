import numpy as np
import pytest

from scripts.export_point_cloud_chunks import (
    build_trajectory,
    load_camera_poses,
    parse_rgb_ply_header,
    point_budget_indices,
    read_rgb_ply,
    rgb_ply_dtype,
    robust_bounds,
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


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 3), dtype=np.float64),
        np.array([[np.nan, 0.0, 0.0], [np.inf, 1.0, 1.0]]),
    ],
)
def test_robust_bounds_rejects_input_without_finite_xyz_points(points):
    with pytest.raises(ValueError, match="No finite XYZ points for robust bounds"):
        robust_bounds(points)


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


def test_build_trajectory_converts_integer_positions_to_json_safe_floats():
    poses = np.zeros((2, 4, 4), dtype=np.int64)
    poses[:, 3, 3] = 1
    poses[:, :3, 3] = [[0, 0, 0], [2, 0, 0]]
    poses[:, :3, 2] = [0, 2, 0]

    result = build_trajectory(poses)

    assert all(
        isinstance(value, float)
        for position in result["positions"]
        for value in position
    )
    assert result["positions"] == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


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

import json

from scripts.export_point_cloud_chunks import (
    compact_dtype,
    export_point_cloud_chunks,
    main,
    parse_args,
)


def test_compact_dtypes_have_exact_layouts():
    quantized = compact_dtype("uint16")
    exact = compact_dtype("float32")

    assert quantized.itemsize == 10
    assert quantized.names == ("position", "color")
    assert quantized["position"].base == np.dtype("<u2")
    assert quantized["position"].shape == (3,)
    assert quantized.fields["position"][1] == 0
    assert quantized["color"].base == np.dtype("<u4")
    assert quantized.fields["color"][1] == 6

    assert exact.itemsize == 16
    assert exact.names == ("position", "color")
    assert exact["position"].base == np.dtype("<f4")
    assert exact["position"].shape == (3,)
    assert exact.fields["position"][1] == 0
    assert exact["color"].base == np.dtype("<u4")
    assert exact.fields["color"][1] == 12


def test_export_chunks_uses_quantized_records_manifest_and_trajectory(tmp_path):
    input_path = tmp_path / "points.ply"
    output_dir = tmp_path / "chunks"
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 2.5], [2.0, 4.0, 5.0], [32767.0, 8.0, 7.5]],
        dtype=np.float32,
    )
    colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [1, 2, 3]], dtype=np.uint8
    )
    make_rgb_ply(input_path, points, colors)

    poses = np.zeros((2, 4, 4), dtype=np.float32)
    poses[:, 3, 3] = 1.0
    poses[:, :3, 3] = [[0, 0, 0], [2, 0, 0]]
    poses[:, :3, 2] = [0, 0, 1]
    poses_path = tmp_path / "camera_poses.npy"
    np.save(poses_path, poses)

    manifest = export_point_cloud_chunks(
        input_path,
        output_dir,
        poses_path=poses_path,
        chunk_size=3,
        position_encoding="uint16",
    )

    assert manifest["format"] == "abot-point-cloud-chunks"
    assert manifest["version"] == 1
    assert manifest["point_count"] == 4
    assert manifest["chunk_count"] == 2
    assert manifest["chunk_size"] == 3
    assert manifest["position_encoding"] == "uint16"
    assert manifest["stride"] == 10
    assert manifest["bounds"]["min"] == [0.0, 0.0, 0.0]
    assert manifest["bounds"]["max"] == [32767.0, 8.0, 7.5]
    assert manifest["chunks"][0]["count"] == 3
    assert manifest["chunks"][0]["bounds"] == {
        "min": [0.0, 0.0, 0.0],
        "max": [2.0, 4.0, 5.0],
    }
    assert manifest["chunks"][1]["count"] == 1
    assert manifest["trajectory"] == "trajectory.json"
    assert manifest["robust_bounds"]["center"] == [1.5, 3.0, 3.75]
    assert manifest["robust_bounds"]["radius"] > 0

    loaded = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert loaded == manifest
    first = np.fromfile(output_dir / "chunk-00000.pbin", dtype=compact_dtype("uint16"))
    assert len(first) == 3
    assert first[0]["color"] == 0xFF0000FF
    assert first[1]["color"] == 0xFF00FF00
    assert first[2]["color"] == 0xFFFF0000

    decoded_min = manifest["bounds"]["min"]
    decoded_max = manifest["bounds"]["max"]
    decoded = decoded_min + first["position"].astype(np.float64) * (
        (np.asarray(decoded_max) - np.asarray(decoded_min)) / 65535.0
    )
    assert np.allclose(decoded[:3], points[:3], atol=1.0)

    trajectory = json.loads((output_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["positions"] == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def test_export_chunks_supports_float32_and_endpoint_inclusive_point_budget(tmp_path):
    input_path = tmp_path / "points.ply"
    output_dir = tmp_path / "chunks"
    points = np.arange(12, dtype=np.float32).reshape(4, 3)
    colors = np.repeat(np.array([[9, 8, 7]], dtype=np.uint8), 4, axis=0)
    make_rgb_ply(input_path, points, colors)

    manifest = export_point_cloud_chunks(
        input_path,
        output_dir,
        chunk_size=2,
        max_points=2,
        position_encoding="float32",
    )

    assert manifest["point_count"] == 2
    assert manifest["stride"] == 16
    assert manifest["chunks"][0]["bounds"] == {
        "min": [0.0, 1.0, 2.0],
        "max": [9.0, 10.0, 11.0],
    }
    records = np.fromfile(output_dir / "chunk-00000.pbin", dtype=compact_dtype("float32"))
    assert np.array_equal(records["position"], points[[0, 3]])
    assert np.all(records["color"] == 0xFF070809)


def test_export_chunks_rejects_invalid_arguments_before_creating_output(tmp_path):
    input_path = tmp_path / "points.ply"
    make_rgb_ply(input_path, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8))
    invalid_arguments = [
        ({"chunk_size": 0}, tmp_path / "a"),
        ({"max_points": -1}, tmp_path / "b"),
        ({"position_encoding": "int8"}, tmp_path / "c"),
    ]

    for extra_args, output_dir in invalid_arguments:
        with pytest.raises(ValueError, match="must be|Unsupported position encoding"):
            export_point_cloud_chunks(input_path, output_dir, **extra_args)
        assert not output_dir.exists()


def test_parse_args_uses_full_resolution_defaults(tmp_path):
    args = parse_args(
        [
            "--input", str(tmp_path / "in.ply"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )

    assert args.chunk_size == 250000
    assert args.max_points == 0
    assert args.position_encoding == "uint16"
    assert args.poses is None


def test_main_writes_compact_chunks_and_reports_result(tmp_path, capsys):
    input_path = tmp_path / "points.ply"
    output_dir = tmp_path / "chunks"
    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    make_rgb_ply(input_path, points, colors)

    main(["--input", str(input_path), "--output-dir", str(output_dir)])

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "chunk-00000.pbin").stat().st_size == 20
    assert not (output_dir / "trajectory.json").exists()
    assert "Wrote 1 chunks, 2 points, 0.00 MB compact; trajectory: no" in capsys.readouterr().out
