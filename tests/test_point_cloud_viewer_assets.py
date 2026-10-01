import json
import re

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
    assert "pointGroup.clear()" in viewer
    assert "trajectoryGroup.clear()" in viewer
    assert "Could not load chunk" in viewer
    assert "Unsupported point-cloud chunk encoding" in viewer


def test_dataset_catalog_is_json_and_parseable():
    path = repository_root() / "tools" / "point_cloud_datasets.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data
    assert all({"name", "manifest"} <= set(item) for item in data)
