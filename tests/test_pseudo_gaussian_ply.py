import numpy as np

from scripts.export_pseudo_gaussian_ply import (
    gaussian_vertex_data,
    gaussian_vertex_dtype,
    rgb_to_sh_dc,
    write_binary_gaussian_ply,
)


def test_gaussian_vertex_dtype_matches_3dgs_ply_layout():
    dtype = gaussian_vertex_dtype()

    assert dtype.names[:3] == ("x", "y", "z")
    assert dtype.names[3:6] == ("nx", "ny", "nz")
    assert dtype.names[6:9] == ("f_dc_0", "f_dc_1", "f_dc_2")
    assert "f_rest_0" in dtype.names and "f_rest_44" in dtype.names
    assert dtype.names[-8:] == (
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )


def test_rgb_to_sh_dc_uses_standard_gaussian_encoding():
    values = rgb_to_sh_dc(np.array([[0, 127, 255], [255, 255, 255]], dtype=np.uint8))

    assert values.dtype == np.uint8
    assert values.shape == (2, 3)
    assert values.min() >= 0
    assert values.max() <= 255
    assert np.all(values[0, 0] < values[0, 2])


def test_gaussian_vertex_data_contains_positions_scales_opacity_and_identity_rotation():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    vertices = gaussian_vertex_data(points, colors, radius=0.25, opacity=0.8)

    assert vertices.dtype == gaussian_vertex_dtype()
    assert len(vertices) == 2
    assert np.allclose(vertices["x"], [1.0, 4.0])
    assert np.allclose(vertices["y"], [2.0, 5.0])
    assert np.allclose(vertices["z"], [3.0, 6.0])
    assert np.allclose(vertices["scale_0"], np.log(0.25))
    assert np.allclose(vertices["scale_1"], np.log(0.25))
    assert np.allclose(vertices["scale_2"], np.log(0.25))
    assert np.allclose(vertices["opacity"], np.log(0.8 / 0.2))
    assert np.allclose(vertices["rot_0"], 1.0)
    assert np.allclose(np.vstack([vertices[name] for name in ("rot_1", "rot_2", "rot_3")]).T, 0.0)


def test_write_binary_gaussian_ply_writes_header_and_records(tmp_path):
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    colors = np.array([[10, 20, 30]], dtype=np.uint8)
    vertices = gaussian_vertex_data(points, colors, radius=0.4, opacity=0.7)
    path = tmp_path / "pseudo.ply"

    write_binary_gaussian_ply(path, vertices)

    data = path.read_bytes()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    assert header.startswith("ply\n")
    assert "format binary_little_endian 1.0" in header
    assert "element vertex 1" in header
    assert "property float x" in header
    assert "property uchar f_rest_44" in header
    assert "property float opacity" in header
    assert len(data) == header_end + vertices.itemsize
