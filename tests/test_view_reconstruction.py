
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.view_reconstruction import (
    camera_view_parameters,
    make_trajectory_arrays,
    next_playback_index,
    world_to_y_up_transform,
)


def identity_pose(translation):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = translation
    return pose


def test_trajectory_arrays_create_connected_time_colored_route():
    poses = np.stack([identity_pose([float(i), 0.0, float(i * 2.0)]) for i in range(3)])
    centers, edges, colors = make_trajectory_arrays(poses)

    assert centers.shape == (3, 3)
    assert np.allclose(centers[:, 0], [0.0, 1.0, 2.0])
    assert edges.shape == (2, 2)
    assert np.allclose(edges, [[0, 1], [1, 2]])
    assert colors.shape == (3, 3)
    assert colors.min() >= 0.0
    assert colors.max() <= 1.0
    assert np.allclose(colors[0], [1.0, colors[0, 1], 0.0])
    assert np.allclose(colors[-1], [0.0, colors[-1, 1], 1.0])


def test_world_to_y_up_transform_flips_camera_y_and_z():
    transform = world_to_y_up_transform()
    point = np.array([1.0, 2.0, 3.0, 1.0])

    assert np.allclose(transform, np.diag([1.0, -1.0, -1.0, 1.0]))
    assert np.allclose((transform @ point)[:3], [1.0, -2.0, -3.0])


def test_camera_view_parameters_use_open3d_camera_convention():
    pose = identity_pose([1.0, 2.0, 3.0])
    view = camera_view_parameters(pose, look_ahead=8.0)

    assert np.allclose(view.center, [1.0, 2.0, 3.0])
    assert np.allclose(view.forward, [0.0, 0.0, 1.0])
    assert np.allclose(view.up, [0.0, -1.0, 0.0])
    assert np.allclose(view.lookat, [1.0, 2.0, 11.0])


def test_camera_view_parameters_can_reverse_view_direction():
    pose = identity_pose([0.0, 0.0, 0.0])
    view = camera_view_parameters(pose, look_ahead=2.0, invert_view=True)

    assert np.allclose(view.forward, [0.0, 0.0, -1.0])
    assert np.allclose(view.lookat, [0.0, 0.0, -2.0])


@pytest.mark.parametrize(
    ("current", "frame_step", "start", "stop", "expected"),
    [
        (0, 5, 0, 10, 5),
        (8, 5, 0, 10, 0),
        (12, 5, 10, 20, 15),
    ],
)
def test_next_playback_index_wraps_inside_selected_range(current, frame_step, start, stop, expected):
    assert next_playback_index(current, frame_step, start, stop) == expected


def test_load_poses_rejects_invalid_shape(tmp_path):
    import numpy as np

    from scripts.view_reconstruction import load_poses

    path = tmp_path / "poses.npy"
    np.save(path, np.zeros((3, 4), dtype=np.float64))

    with pytest.raises(ValueError, match="Expected poses with shape"):
        load_poses(path)


def test_point_budget_indices_preserve_order_and_endpoints():
    from scripts.view_reconstruction import point_budget_indices

    indices = point_budget_indices(point_count=1000, max_points=4)

    assert indices.shape == (4,)
    assert indices[0] == 0
    assert indices[-1] == 999
    assert np.all(np.diff(indices) > 0)


def test_point_budget_indices_keep_all_points_when_no_budget():
    from scripts.view_reconstruction import point_budget_indices

    indices = point_budget_indices(point_count=7, max_points=0)

    assert np.array_equal(indices, np.arange(7))


def test_chase_pose_applies_camera_local_offset():
    from scripts.view_reconstruction import chase_pose

    pose = identity_pose([1.0, 2.0, 3.0])
    result = chase_pose(pose, np.array([0.0, -1.5, -8.0]))

    assert np.allclose(result[:3, 3], [1.0, 0.5, -5.0])
    assert np.allclose(result[:3, :3], pose[:3, :3])


def test_invert_pose_orientation_turns_camera_without_moving_it():
    from scripts.view_reconstruction import invert_pose_orientation

    pose = identity_pose([1.0, 2.0, 3.0])
    result = invert_pose_orientation(pose)

    assert np.allclose(result[:3, 3], pose[:3, 3])
    assert np.allclose(result[:3, :3], np.diag([-1.0, 1.0, -1.0]))


def test_show_orbit_applies_render_options_and_destroys_window():
    from scripts.view_reconstruction import show_orbit

    class Render:
        point_size = None
        line_width = None
        background_color = None

    class Visualizer:
        def __init__(self):
            self.render = Render()
            self.geometries = []
            self.created = False
            self.destroyed = False

        def create_window(self, **kwargs):
            self.created = True
            self.window_kwargs = kwargs

        def add_geometry(self, geometry):
            self.geometries.append(geometry)

        def get_render_option(self):
            return self.render

        def poll_events(self):
            return False

        def update_renderer(self):
            pass

        def destroy_window(self):
            self.destroyed = True

    args = type(
        "Args",
        (),
        {
            "window_width": 640,
            "window_height": 480,
            "point_size": 1.5,
            "line_width": 3.0,
            "background": "dark",
        },
    )()
    geometries = ["point-cloud", "route"]
    fake = SimpleNamespace(visualization=SimpleNamespace(Visualizer=Visualizer))

    visualizer = show_orbit(fake, geometries, args)
    assert visualizer.created
    assert visualizer.geometries == geometries
    assert visualizer.render.point_size == 1.5
    assert visualizer.render.line_width == 3.0
    assert visualizer.destroyed


def test_focal_length_supports_zoom_and_vertical_fov():
    from scripts.view_reconstruction import focal_length_from_fov

    base = focal_length_from_fov(width=1600, height=900, fov_degrees=70.0, zoom=1.0)
    zoomed = focal_length_from_fov(width=1600, height=900, fov_degrees=70.0, zoom=2.0)
    narrow = focal_length_from_fov(width=1600, height=900, fov_degrees=35.0, zoom=1.0)

    assert base > 0
    assert np.isclose(zoomed, 2.0 * base)
    assert narrow > base
