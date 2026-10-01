# Streaming RGB Point-Cloud Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a progressive three.js RGB point-cloud viewer that can display `demo_loop`-scale reconstructions with interactive orbit controls and camera-trajectory playback.

**Architecture:** A new NumPy-only converter parses an existing binary RGB PLY, optionally samples it, computes robust framing bounds, and emits compact chunk plus JSON manifest files. A new browser page fetches those chunks, decodes them into one three.js `Points` object per chunk, overlays a rainbow camera trajectory, and provides orbit and drive-camera modes. The existing repository-root HTTP server serves both the chunks and the viewer.

**Tech Stack:** Python 3.10+, NumPy, stdlib `argparse`/`json`/`pathlib`, HTML/CSS/JavaScript, WebGL2, three.js 0.170, pytest, Playwright MCP for manual browser verification.

**Spec:** `docs/superpowers/specs/2026-10-01-streaming-point-cloud-viewer-design.md`

## Global Constraints

- Do not add a Python runtime dependency; converter code may use only Python stdlib and NumPy.
- Default compact encoding is exactly 10 bytes per point: three little-endian `uint16` coordinates plus one packed `uint32` RGBA color.
- Optional `float32` encoding is exactly 16 bytes per point.
- Preserve original ABot-Recon world coordinates; do not flip Y/Z in the new point-cloud viewer.
- Default `--max-points 0` means retain every finite point.
- Default `--chunk-size 250000`.
- Manifest format is `abot-point-cloud-chunks`, version `1`.
- Trajectory format is `abot-point-cloud-trajectory`, version `1`.
- Orbit viewing must work without poses; trajectory and drive playback require poses.
- Run every new/changed Python test after each task.
- Commit after every completed task.

---

### Task 1: RGB PLY parsing and point selection

**Files:**
- Create: `scripts/export_point_cloud_chunks.py`
- Create: `tests/test_export_point_cloud_chunks.py`

**Interfaces:**
- Produces `RgbPlyHeader(vertex_count: int, data_offset: int, stride: int, offsets: dict[str, int])`.
- Produces `rgb_ply_dtype(offsets: dict[str, int]) -> np.dtype`.
- Produces `parse_rgb_ply_header(path: Path) -> RgbPlyHeader`.
- Produces `read_rgb_ply(path: Path, header: RgbPlyHeader | None = None) -> tuple[np.ndarray, np.ndarray]`.
- Produces `point_budget_indices(point_count: int, max_points: int) -> np.ndarray`.
- Later tasks consume these exact names.

- [ ] **Step 1: Write failing parser and selection tests**

Create `tests/test_export_point_cloud_chunks.py` with this initial content:

```python
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
```

- [ ] **Step 2: Run the new tests and verify the module import fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_point_cloud_chunks.py -q
```

Expected: all tests fail during import with `ModuleNotFoundError: No module named 'scripts.export_point_cloud_chunks'`.

- [ ] **Step 3: Implement the minimal parser**

Create `scripts/export_point_cloud_chunks.py` with:

```python
"""Convert an RGB reconstruction PLY into compact streamed point chunks."""

from __future__ import annotations

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


@dataclass(frozen=True)
class RgbPlyHeader:
    vertex_count: int
    data_offset: int
    stride: int
    offsets: dict[str, int]


def rgb_ply_dtype(offsets: dict[str, int]) -> np.dtype:
    fields = [
        ("x", "<f4", offsets["x"]),
        ("y", "<f4", offsets["y"]),
        ("z", "<f4", offsets["z"]),
        ("red", "u1", offsets["red"]),
        ("green", "u1", offsets["green"]),
        ("blue", "u1", offsets["blue"]),
    ]
    return np.dtype({"names": [n for n, _, _ in fields], "formats": [f for _, f, _ in fields], "offsets": [o for _, _, o in fields], "itemsize": max(o + np.dtype(f).itemsize for _, f, o in fields)})
```

Implement `parse_rgb_ply_header` by:

1. opening the file in binary mode;
2. requiring the first line to equal `ply`;
3. accepting only `format binary_little_endian 1.0`;
4. tracking properties only while `current_element == "vertex"`;
5. rejecting list properties;
6. recording the offset after `end_header\n`;
7. requiring all of `x`, `y`, `z`, `red`, `green`, and `blue`.

Implement `read_rgb_ply` by:

1. parsing the header;
2. using `np.memmap(..., dtype=rgb_ply_dtype(header.offsets), mode="r", offset=header.data_offset, shape=(header.vertex_count,))`;
3. converting XYZ to float32 and RGB to uint8;
4. returning only rows whose XYZ values are all finite.

Implement `point_budget_indices` with the same semantics as the existing pseudo-Gaussian exporter.

- [ ] **Step 4: Run parser tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_point_cloud_chunks.py -q
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit parser**

```powershell
git add scripts/export_point_cloud_chunks.py tests/test_export_point_cloud_chunks.py
git commit -m "feat: parse rgb point clouds for streaming"
```

---

### Task 2: Robust bounds and trajectory conversion

**Files:**
- Modify: `scripts/export_point_cloud_chunks.py`
- Modify: `tests/test_export_point_cloud_chunks.py`

**Interfaces:**
- Produces `robust_bounds(points: np.ndarray, sample_limit: int = 20000) -> dict[str, list[float]]`.
- Produces `load_camera_poses(path: Path) -> np.ndarray`.
- Produces `build_trajectory(poses: np.ndarray) -> dict[str, Any]`.
- Later tasks consume these exact functions.

- [ ] **Step 1: Add failing bounds and trajectory tests**

Append:

```python
from scripts.export_point_cloud_chunks import build_trajectory, load_camera_poses, robust_bounds


def test_robust_bounds_uses_median_center_and_high_distance_percentile():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [1000, 0, 0]],
        dtype=np.float32,
    )

    result = robust_bounds(points, sample_limit=1000)

    assert result["center"] == [1.0, 0.0, 0.0]
    assert 2.0 <= result["radius"] <= 4.0


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
```

- [ ] **Step 2: Run tests and verify the three new tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_point_cloud_chunks.py -q
```

Expected: `ImportError` for the three missing names.

- [ ] **Step 3: Implement bounds and trajectory helpers**

Implement `robust_bounds` exactly as:

1. retain finite XYZ rows;
2. sample at most `sample_limit` evenly spaced rows;
3. compute each coordinate median as center;
4. compute Euclidean distances from that center;
5. set radius to `max(np.percentile(distances, 96), 1e-3)`;
6. return JSON-safe float lists.

Implement `load_camera_poses` by:

1. calling `np.load(path, allow_pickle=False)`;
2. requiring shape `[N,4,4]`;
3. requiring all values finite.

Implement `build_trajectory` by extracting centers and normalized `pose[:3,2]` optical axes, skipping non-finite rows, and returning plain Python float lists. If no valid rows remain, return empty `positions` and `forwards` arrays.

- [ ] **Step 4: Run tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_point_cloud_chunks.py -q
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/export_point_cloud_chunks.py tests/test_export_point_cloud_chunks.py
git commit -m "feat: derive framing bounds and trajectory"
```

---

### Task 3: Compact chunk writer and CLI

**Files:**
- Modify: `scripts/export_point_cloud_chunks.py`
- Modify: `tests/test_export_point_cloud_chunks.py`

**Interfaces:**
- Produces `compact_dtype(position_encoding: str = "uint16") -> np.dtype`.
- Produces `export_point_cloud_chunks(input_path: Path, output_dir: Path, poses_path: Path | None = None, chunk_size: int = 250000, max_points: int = 0, position_encoding: str = "uint16") -> dict[str, Any]`.
- Produces `parse_args(argv: list[str] | None = None) -> argparse.Namespace`.
- Produces `main(argv: list[str] | None = None) -> None`.
- Viewer and documentation consume the resulting manifest schema.

- [ ] **Step 1: Add failing chunk/CLI tests**

Append:

```python
import json

from scripts.export_point_cloud_chunks import compact_dtype, export_point_cloud_chunks, parse_args


def test_compact_dtypes_have_exact_layouts():
    quantized = compact_dtype("uint16")
    exact = compact_dtype("float32")

    assert quantized.itemsize == 10
    assert quantized.names == ("position", "color")
    assert quantized["position"].base == np.dtype("<u2")
    assert quantized["position"].shape == (3,)
    assert quantized["color"].base == np.dtype("<u4")

    assert exact.itemsize == 16
    assert exact.names == ("position", "color")
    assert exact["position"].base == np.dtype("<f4")
    assert exact["position"].shape == (3,)
    assert exact["color"].base == np.dtype("<u4")


def test_export_chunks_uses_quantized_records_manifest_and_trajectory(tmp_path):
    input_path = tmp_path / "points.ply"
    output_dir = tmp_path / "chunks"
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 2.5], [2.0, 4.0, 5.0], [32767.0, 8.0, 7.5]],
        dtype=np.float32,
    )
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [1, 2, 3]], dtype=np.uint8)
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
    assert manifest["chunks"][1]["count"] == 1
    assert manifest["trajectory"] == "trajectory.json"
    assert manifest["robust_bounds"]["center"] == [1.0, 2.0, 2.5]
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
    records = np.fromfile(output_dir / "chunk-00000.pbin", dtype=compact_dtype("float32"))
    assert np.array_equal(records["position"], points[[0, 3]])
    assert np.all(records["color"] == 0xFF070809)


def test_export_chunks_rejects_invalid_arguments(tmp_path):
    input_path = tmp_path / "points.ply"
    make_rgb_ply(input_path, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8))

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        export_point_cloud_chunks(input_path, tmp_path / "a", chunk_size=0)
    with pytest.raises(ValueError, match="max_points must be non-negative"):
        export_point_cloud_chunks(input_path, tmp_path / "b", max_points=-1)
    with pytest.raises(ValueError, match="Unsupported position encoding"):
        export_point_cloud_chunks(input_path, tmp_path / "c", position_encoding="int8")


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
```

- [ ] **Step 2: Run tests and verify new failures**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_point_cloud_chunks.py -q
```

Expected: import failures for `compact_dtype`, `export_point_cloud_chunks`, and `parse_args`.

- [ ] **Step 3: Implement chunk writer and CLI**

Implement compact dtypes as structured arrays:

```python
def compact_dtype(position_encoding: str = "uint16") -> np.dtype:
    if position_encoding == "uint16":
        return np.dtype([("position", "<u2", (3,)), ("color", "<u4")])
    if position_encoding == "float32":
        return np.dtype([("position", "<f4", (3,)), ("color", "<u4")])
    raise ValueError(f"Unsupported position encoding: {position_encoding}")
```

Implement export by:

1. validating `chunk_size`, `max_points`, and encoding before creating output;
2. reading finite PLY points/colors;
3. applying `point_budget_indices`;
4. raising `ValueError("Input point cloud has no finite XYZ points")` when empty;
5. computing actual bounds and robust bounds;
6. splitting selected points into chunks;
7. packing colors as `red | green << 8 | blue << 16 | 0xFF000000`;
8. for `uint16`, quantizing against global bounds with:

```python
scale = np.maximum(bounds_max - bounds_min, np.finfo(np.float64).eps)
encoded = np.rint((chunk_points - bounds_min) / scale * 65535.0).astype(np.uint16)
```

9. writing files named `chunk-00000.pbin`;
10. recording each chunk's actual source-coordinate bounds;
11. writing pretty UTF-8 JSON;
12. writing trajectory JSON only when `poses_path` is supplied, and adding `"trajectory": "trajectory.json"` to the manifest.

CLI output must state chunk count, point count, compact MB, and whether trajectory was written.

- [ ] **Step 4: Run converter tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_point_cloud_chunks.py -q
```

Expected: 12 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/export_point_cloud_chunks.py tests/test_export_point_cloud_chunks.py
git commit -m "feat: stream rgb point clouds into chunks"
```

---

### Task 4: Progressive three.js viewer

**Files:**
- Create: `tools/point_cloud_viewer.html`
- Create: `tests/test_point_cloud_viewer_assets.py`

**Interfaces:**
- Consumes manifest fields from Task 3.
- Consumes `tools/point_cloud_datasets.json` entries shaped as:

```json
{
  "name": "demo_loop · full RGB 25.6M",
  "manifest": "/outputs/demo_loop/point_cloud_chunks/manifest.json",
  "trajectory": "/outputs/demo_loop/point_cloud_chunks/trajectory.json"
}
```

- Produces query parameters: `manifest=<URL>`, `trajectory=<URL|empty>`, `dataset=<catalog name|custom>`.

- [ ] **Step 1: Add failing viewer asset tests**

Create `tests/test_point_cloud_viewer_assets.py`:

```python
import json
import re
from pathlib import Path

import pytest

from scripts.serve_pseudo_gaussian_viewer import repository_root


@pytest.fixture()
def viewer() -> str:
    path = repository_root() / "tools" / "point_cloud_viewer.html"
    return path.read_text(encoding="utf-8")


def test_viewer_uses_three_js_hard_points(viewer):
    assert "three@0.170.0/build/three.module.min.js" in viewer
    assert "THREE.PointsMaterial" in viewer
    assert "vertexColors: true" in viewer
    assert "transparent: false" in viewer
    assert "alphaTest: 0.5" in viewer
    assert "SRC_ALPHA" not in viewer


def test_viewer_declares_required_controls(viewer):
    control_ids = [
        "dataset",
        "manifestInput",
        "trajectoryInput",
        "loadDataset",
        "orbitButton",
        "driveButton",
        "playPauseButton",
        "resetButton",
        "trajectoryVisible",
        "pointSize",
        "playbackSpeed",
        "playbackProgress",
        "cameraHeight",
        "cameraBackoff",
        "status",
    ]
    for element_id in control_ids:
        assert f'id="{element_id}"' in viewer


def test_viewer_decodes_both_chunk_encodings(viewer):
    assert "abot-point-cloud-chunks" in viewer
    assert "position_encoding" in viewer
    assert "Uint16Array" in viewer
    assert "Float32Array" in viewer
    assert "decodeQuantizedChunk" in viewer
    assert "decodeFloatChunk" in viewer
    assert "Math.min(65535, Math.max(0" in viewer


def test_viewer_implements_trajectory_and_modes(viewer):
    for function_name in (
        "buildTrajectory",
        "resampleTrajectory",
        "applyDriveCamera",
        "frameScene",
        "setMode",
        "loadChunk",
    ):
        assert f"function {function_name}" in viewer


def test_viewer_loads_progressively_and_reports_errors(viewer):
    assert "loadedPointCount" in viewer
    assert "requestAnimationFrame(render)" in viewer
    assert "Could not load chunk" in viewer
    assert "Unsupported point-cloud chunk encoding" in viewer


def test_dataset_catalog_is_json_and_parseable():
    path = repository_root() / "tools" / "point_cloud_datasets.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data
    assert all({"name", "manifest"} <= set(item) for item in data)
```

- [ ] **Step 2: Run viewer tests and verify file absence fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_point_cloud_viewer_assets.py -q
```

Expected: viewer fixture fails because `tools/point_cloud_viewer.html` does not exist.

- [ ] **Step 3: Create the catalog**

Create `tools/point_cloud_datasets.json`:

```json
[
  {
    "name": "demo_loop · full RGB",
    "manifest": "/outputs/demo_loop/point_cloud_chunks/manifest.json",
    "trajectory": "/outputs/demo_loop/point_cloud_chunks/trajectory.json"
  }
]
```

- [ ] **Step 4: Implement viewer structure and logic**

Implement an HTML page with:

- full-viewport canvas;
- compact dark control panel;
- dataset select, manifest input, trajectory input, and load button;
- Orbit / Drive mode buttons;
- play-pause, reset, trajectory visibility;
- point size, speed, progress, camera height, and backoff controls;
- status line with loaded point count.

JavaScript must include these exact functions:

```javascript
async function ensureThree()
function initRenderer()
function resize()
function makeCircleSprite()
function decodeQuantizedChunk(buffer, count, boundsMin, boundsMax)
function decodeFloatChunk(buffer, count)
function updatePointSize()
function addChunkObject(chunk, positions, colors)
async function loadChunk(chunk)
async function loadChunks(manifestUrl, manifest)
function buildTrajectory(payload)
function resampleTrajectory(trajectory, sampleCount = 1024)
function buildTrajectoryObjects()
function samplePath(progress)
function applyOrbit()
function applyDriveCamera(progress)
function setMode(mode)
function frameScene()
async function loadScene()
function render(timestamp)
```

Decode rules:

- `uint16`: read three little-endian `uint16` values, reconstruct against global manifest bounds, read packed RGBA bytes as R/G/B, ignore A.
- `float32`: read three little-endian `float32` values, read packed RGBA bytes as R/G/B.
- Create `Float32Array` positions and `Uint8Array` RGB colors for each chunk.

Rendering rules:

- initialize WebGL with antialias where available;
- cap device pixel ratio at 2;
- use `THREE.PointsMaterial` with a white circle sprite, vertex colors, size attenuation, `transparent: false`, `alphaTest: 0.5`;
- enable depth testing;
- add each chunk immediately after successful fetch/decode/upload;
- use robust bounds and trajectory bounds in `frameScene()`;
- use original camera coordinates with `camera.up.set(0, -1, 0)`;
- trajectory uses a `THREE.Line`, rainbow vertex colors, a moving sphere marker, and start/end markers.

Interaction rules:

- left drag orbits;
- right or Shift+left drag pans;
- wheel zooms;
- progress bar and play/pause affect drive mode;
- mode switching preserves trajectory progress;
- reset restores automatic framing and orbit mode;
- chunk loading does not block rendering.

- [ ] **Step 5: Run viewer tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_point_cloud_viewer_assets.py -q
```

Expected: 6 tests pass.

- [ ] **Step 6: Commit viewer**

```powershell
git add tools/point_cloud_viewer.html tools/point_cloud_datasets.json tests/test_point_cloud_viewer_assets.py
git commit -m "feat: add progressive rgb point-cloud viewer"
```

---

### Task 5: Documentation and end-to-end verification

**Files:**
- Modify: `docs/reconstruction-visualization-tools-zh.md`
- Modify: `README_ZH.md` only if it has a visualization-tool index entry that should list the new viewer.
- Modify: `tools/point_cloud_datasets.json` only if browser verification reveals a path correction.
- No production-code changes unless verification exposes a defect.

**Interfaces:**
- Consumes all previous tasks.
- Produces documented commands and verified browser behavior.

- [ ] **Step 1: Update Chinese visualization guide**

Insert a new “官方风格硬点云网页查看” section before the pseudo-Gaussian workflow. Document:

```powershell
.\.venv\Scripts\python.exe scripts\export_point_cloud_chunks.py `
  --input outputs\demo_loop\reconstruction.ply `
  --poses outputs\demo_loop\camera_poses.npy `
  --output-dir outputs\demo_loop\point_cloud_chunks `
  --chunk-size 250000 `
  --max-points 0
```

Then:

```powershell
.\.venv\Scripts\python.exe scripts\serve_pseudo_gaussian_viewer.py --no-browser
```

Open:

```text
http://127.0.0.1:8765/tools/point_cloud_viewer.html
```

Document controls and tradeoffs:

- hard points are for judging reconstruction;
- pseudo-Gaussian is only a synthetic surface preview;
- `uint16` default is smaller and usually visually equivalent;
- `float32` preserves source coordinates exactly;
- `--max-points` creates lighter catalogs;
- no poses means orbit-only.

- [ ] **Step 2: Run all relevant automated tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_point_cloud_chunks.py tests\test_point_cloud_viewer_assets.py tests\test_serve_pseudo_gaussian_viewer.py -q
```

Expected: all pass.

- [ ] **Step 3: Convert demo_loop chunks**

Run the exact command from Task 5 Step 1. This may write approximately 256 MB under ignored `outputs/demo_loop/point_cloud_chunks`.

Expected output includes:

```text
103 chunks
25,596,210 points
trajectory.json
```

- [ ] **Step 4: Start local server**

```powershell
.\.venv\Scripts\python.exe scripts\serve_pseudo_gaussian_viewer.py --no-browser
```

Expected:

```text
Serving D:\code\ABot-Recon
Viewer: http://127.0.0.1:8765/tools/pseudo_gaussian_viewer.html
Press Ctrl+C to stop.
```

- [ ] **Step 5: Browser verification with Playwright MCP**

Open:

```text
http://127.0.0.1:8765/tools/point_cloud_viewer.html
```

Verify and capture screenshots:

1. catalog defaults to `demo_loop · full RGB`;
2. manifest and trajectory load;
3. loaded point count increases progressively;
4. initial scene is not blank after the first chunks;
5. orbit drag rotates;
6. wheel zoom changes view;
7. trajectory toggle works;
8. Drive mode follows the route;
9. progress scrub works;
10. reset restores framing;
11. browser console has no errors.

- [ ] **Step 6: Fix defects found by verification**

For each defect:

1. add or tighten the smallest failing automated test when feasible;
2. make one focused correction;
3. rerun that test plus the relevant task suite;
4. do not bundle unrelated cleanup.

- [ ] **Step 7: Final regression check**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_export_point_cloud_chunks.py tests\test_point_cloud_viewer_assets.py tests\test_serve_pseudo_gaussian_viewer.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit docs and verification fixes**

```powershell
git add docs/reconstruction-visualization-tools-zh.md tools/point_cloud_viewer.html tools/point_cloud_datasets.json scripts/export_point_cloud_chunks.py tests
git commit -m "docs: document streaming point-cloud viewer"
```

---

## Final Definition of Done

- `scripts/export_point_cloud_chunks.py` has no new Python dependency and all converter unit tests pass.
- `tools/point_cloud_viewer.html` renders hard RGB points, not pseudo-Gaussian splats.
- `demo_loop` chunks load progressively in the browser.
- Orbit mode works.
- Drive mode follows the reconstructed camera trajectory.
- Scrub, speed, height, backoff, reset, and trajectory visibility controls work.
- Missing poses disables trajectory/drive without breaking orbit viewing.
- Documentation includes conversion, serving, controls, and encoding tradeoffs.
- No requested automated test fails.
- Work is committed in focused commits.
