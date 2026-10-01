"""Interactive Open3D viewer for ABot-Recon point clouds and driving trajectories."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraView:
    center: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    lookat: np.ndarray


def world_to_y_up_transform() -> np.ndarray:
    """Map ABot-Recon's camera-style world axes to Open3D's Y-up convention."""
    transform = np.eye(4, dtype=np.float64)
    transform[1, 1] = -1.0
    transform[2, 2] = -1.0
    return transform


def make_trajectory_arrays(
    poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centers, edge indices, and time-colored RGB values for a route."""
    centers = np.asarray(poses[:, :3, 3], dtype=np.float64).copy()
    if len(centers) < 2:
        edges = np.empty((0, 2), dtype=np.int32)
    else:
        edges = np.column_stack(
            (
                np.arange(len(centers) - 1, dtype=np.int32),
                np.arange(1, len(centers), dtype=np.int32),
            )
        )

    if len(centers) <= 1:
        alpha = np.zeros(len(centers), dtype=np.float64)
    else:
        alpha = np.linspace(0.0, 1.0, len(centers), dtype=np.float64)
    green = 0.25 + 0.5 * (1.0 - np.abs(2.0 * alpha - 1.0))
    colors = np.column_stack((1.0 - alpha, green, alpha))
    return centers, edges, colors


def camera_view_parameters(
    pose: np.ndarray, look_ahead: float, invert_view: bool = False
) -> CameraView:
    """Convert an ABot camera pose to Open3D's lookat/front/up view convention."""
    center = np.asarray(pose[:3, 3], dtype=np.float64)
    forward = np.asarray(pose[:3, 2], dtype=np.float64)
    up = -np.asarray(pose[:3, 1], dtype=np.float64)

    forward_norm = np.linalg.norm(forward)
    up_norm = np.linalg.norm(up)
    if forward_norm < 1e-8 or up_norm < 1e-8:
        raise ValueError("Camera pose contains a degenerate forward or up axis")
    forward = forward / forward_norm
    up = up / up_norm

    if invert_view:
        forward = -forward
    return CameraView(
        center=center,
        forward=forward,
        up=up,
        lookat=center + forward * float(look_ahead),
    )


def next_playback_index(current: int, frame_step: int, start: int, stop: int) -> int:
    """Advance to the next sampled playback slot and wrap inside the range."""
    span = stop - start
    if span <= 0:
        raise ValueError("Playback range must contain at least one frame")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    offset = current - start
    next_slot = (offset // frame_step + 1) * frame_step
    if next_slot >= span:
        next_slot = 0
    return start + next_slot


def selected_frame_range(args: argparse.Namespace, frame_count: int) -> tuple[int, int]:
    stop = frame_count if args.stop_frame is None else args.stop_frame
    start = max(0, min(args.start_frame, frame_count - 1))
    stop = max(start + 1, min(stop, frame_count))
    return start, stop


def load_poses(path: Path) -> np.ndarray:
    poses = np.asarray(np.load(path), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses with shape [N,4,4], got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("Pose array contains NaN or Inf")
    if len(poses) < 2:
        raise ValueError("At least two poses are required to draw a trajectory")
    return poses


def point_budget_indices(point_count: int, max_points: int) -> np.ndarray:
    if max_points < 0:
        raise ValueError("max_points must be non-negative")
    if max_points == 0 or point_count <= max_points:
        return np.arange(point_count, dtype=np.int64)
    if max_points == 1:
        return np.array([point_count // 2], dtype=np.int64)
    return np.linspace(0, point_count - 1, max_points, dtype=np.int64)


def chase_pose(pose: np.ndarray, camera_offset: np.ndarray) -> np.ndarray:
    """Move a camera pose by an offset in its local right/down/forward coordinates."""
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3, 3] += result[:3, :3] @ np.asarray(camera_offset, dtype=np.float64)
    return result


def invert_pose_orientation(pose: np.ndarray) -> np.ndarray:
    """Turn a camera 180 degrees around its local down axis."""
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3, :3] = result[:3, :3] @ np.diag([-1.0, 1.0, -1.0])
    return result

def background_color(name: str) -> np.ndarray:
    if name == "dark":
        return np.asarray([0.05, 0.06, 0.08], dtype=np.float64)
    return np.asarray([0.93, 0.94, 0.96], dtype=np.float64)


def import_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as error:
        raise SystemExit(
            "Open3D is required for the interactive viewer. Install it with:\n"
            "  python -m pip install open3d"
        ) from error
    return o3d


def trajectory_line_set(o3d: Any, poses: np.ndarray) -> Any:
    centers, edges, colors = make_trajectory_arrays(poses)
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(centers)
    line.lines = o3d.utility.Vector2iVector(edges)
    line.colors = o3d.utility.Vector3dVector(colors)
    return line


def endpoint_marker(o3d: Any, center: np.ndarray, radius: float, color: np.ndarray) -> Any:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=24)
    marker.paint_uniform_color(color)
    marker.translate(center)
    return marker


def load_point_cloud(o3d: Any, args: argparse.Namespace) -> Any:
    point_cloud = o3d.io.read_point_cloud(str(args.ply))
    if not point_cloud.has_points():
        raise SystemExit(f"No points found in {args.ply}")
    if not point_cloud.has_colors():
        point_cloud.paint_uniform_color([0.65, 0.68, 0.72])

    indices = point_budget_indices(len(point_cloud.points), args.max_points)
    if len(indices) != len(point_cloud.points):
        point_cloud = point_cloud.select_by_index(indices)
    if args.voxel > 0:
        point_cloud = point_cloud.voxel_down_sample(args.voxel)
    if args.flip_y:
        point_cloud.transform(world_to_y_up_transform())
    return point_cloud


def focal_length_from_fov(width: int, height: int, fov_degrees: float, zoom: float) -> float:
    """Return the vertical focal length after an additional zoom factor."""
    if width <= 0 or height <= 0:
        raise ValueError("Window dimensions must be positive")
    if not 10.0 <= fov_degrees <= 120.0:
        raise ValueError("FOV must be between 10 and 120 degrees")
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    return 0.5 * height / np.tan(np.deg2rad(fov_degrees) / 2.0) * zoom

def pinhole_parameters(
    o3d: Any,
    pose: np.ndarray,
    width: int,
    height: int,
    fov_degrees: float,
    zoom: float,
) -> Any:
    parameters = o3d.camera.PinholeCameraParameters()
    focal = focal_length_from_fov(width, height, fov_degrees, zoom)
    parameters.intrinsic.set_intrinsics(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=width / 2.0,
        cy=height / 2.0,
    )
    parameters.extrinsic = np.linalg.inv(pose)
    return parameters


def apply_render_options(o3d: Any, vis: Any, args: argparse.Namespace) -> None:
    render = vis.get_render_option()
    render.point_size = args.point_size
    render.line_width = args.line_width
    render.background_color = background_color(args.background)


def show_orbit(o3d: Any, geometries: list[Any], args: argparse.Namespace) -> Any:
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name="ABot-Recon driving scene",
        width=args.window_width,
        height=args.window_height,
    )
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    apply_render_options(o3d, visualizer, args)
    while visualizer.poll_events():
        visualizer.update_renderer()
    visualizer.destroy_window()
    return visualizer

def validate_args(args: argparse.Namespace) -> None:
    if not args.ply.is_file():
        raise SystemExit(f"PLY file does not exist: {args.ply}")
    if not args.poses.is_file():
        raise SystemExit(f"Pose file does not exist: {args.poses}")
    if args.voxel < 0:
        raise SystemExit("--voxel must be non-negative")
    if args.point_size <= 0:
        raise SystemExit("--point-size must be positive")
    if args.line_width <= 0:
        raise SystemExit("--line-width must be positive")
    if args.frame_step <= 0:
        raise SystemExit("--frame-step must be positive")
    if not 0.1 <= args.playback_fps <= 120:
        raise SystemExit("--playback-fps must be between 0.1 and 120")
    if args.look_ahead < 0:
        raise SystemExit("--look-ahead must be non-negative")
    if args.zoom <= 0:
        raise SystemExit("--zoom must be positive")
    if args.window_width < 320 or args.window_height < 240:
        raise SystemExit("Window dimensions must be at least 320x240")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively inspect an ABot-Recon PLY and driving trajectory."
    )
    parser.add_argument("--ply", type=Path, required=True, help="Colored point-cloud PLY")
    parser.add_argument("--poses", type=Path, required=True, help="camera_poses.npy [N,4,4]")
    parser.add_argument(
        "--mode",
        choices=("orbit", "drive"),
        default="orbit",
        help="orbit: free mouse navigation; drive: camera follows the trajectory",
    )
    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel size in meters")
    parser.add_argument("--max-points", type=int, default=0, help="Point budget; 0 keeps all")
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument("--line-width", type=float, default=3.0)
    parser.add_argument(
        "--flip-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map camera-style Y-down/Z-forward axes to Open3D Y-up coordinates",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--playback-fps", type=float, default=24.0)
    parser.add_argument("--look-ahead", type=float, default=8.0)
    parser.add_argument(
        "--camera-offset",
        type=float,
        nargs=3,
        metavar=("RIGHT", "DOWN", "FORWARD"),
        default=(0.0, -1.5, -8.0),
        help="Chase-camera offset in local camera coordinates",
    )
    parser.add_argument("--zoom", type=float, default=1.0, help="Drive-camera zoom; 2.0 doubles focal length")
    parser.add_argument("--camera-fov", type=float, default=70.0, help="Drive-camera vertical FOV in degrees")
    parser.add_argument("--invert-view", action="store_true")
    parser.add_argument("--background", choices=("dark", "light"), default="dark")
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=900)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    o3d = import_open3d()

    poses = load_poses(args.poses)
    if args.flip_y:
        poses = world_to_y_up_transform() @ poses

    print(f"Loading {args.ply}")
    point_cloud = load_point_cloud(o3d, args)
    route = trajectory_line_set(o3d, poses)
    route_extent = float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]))
    marker_radius = max(0.5, route_extent * 0.002)
    start_marker = endpoint_marker(
        o3d, poses[0, :3, 3], marker_radius, np.asarray([219 / 255, 57 / 255, 57 / 255])
    )
    end_marker = endpoint_marker(
        o3d, poses[-1, :3, 3], marker_radius, np.asarray([43 / 255, 108 / 255, 235 / 255])
    )
    geometries = [point_cloud, route, start_marker, end_marker]

    print(
        f"Loaded {len(point_cloud.points):,} points and {len(poses):,} poses; "
        f"route extent: {route_extent:.1f} m"
    )

    if args.mode == "orbit":
        show_orbit(o3d, geometries, args)
        return

    start, stop = selected_frame_range(args, len(poses))
    state = {"index": start, "last_time": None}
    frame_interval = 1.0 / args.playback_fps
    camera_offset = np.asarray(args.camera_offset, dtype=np.float64)

    def animation_callback(vis: Any) -> bool:
        now = time.perf_counter()
        last_time = state["last_time"]
        if last_time is not None and now - last_time < frame_interval:
            return False
        state["last_time"] = now

        index = int(state["index"])
        pose = poses[index]
        if args.invert_view:
            pose = invert_pose_orientation(pose)
        display_pose = chase_pose(pose, camera_offset)
        view_control = vis.get_view_control()
        parameters = pinhole_parameters(
            o3d, display_pose, args.window_width, args.window_height, args.camera_fov, args.zoom
        )
        view_control.convert_from_pinhole_camera_parameters(
            parameters, allow_arbitrary=True
        )
        apply_render_options(o3d, vis, args)
        vis.update_renderer()

        state["index"] = next_playback_index(index, args.frame_step, start, stop)
        print(
            f"\rPlayback frame {index + 1:,}/{len(poses):,} "
            f"(selected {start + 1:,}-{stop:,})   ",
            end="",
        )
        return False

    print(
        "Drive mode controls: mouse wheel resizes points while the animation runs; "
        "close the window to stop."
    )
    o3d.visualization.draw_geometries_with_animation_callback(
        geometries,
        animation_callback,
        window_name="ABot-Recon driving playback",
        width=args.window_width,
        height=args.window_height,
    )
    print()


if __name__ == "__main__":
    main()
