import json
import re
import shutil
import subprocess
import textwrap

import pytest

from scripts.serve_pseudo_gaussian_viewer import repository_root


@pytest.fixture()
def viewer() -> str:
    path = repository_root() / "tools" / "point_cloud_viewer.html"
    return path.read_text(encoding="utf-8")


def viewer_function(viewer: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^(?:async )?function {re.escape(name)}\(.*?\n\}}(?=\n|\Z)",
        viewer,
    )
    assert match, f"viewer function {name!r} was not found"
    return match.group(0)


def run_node(script: str) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the inline viewer module")
    result = subprocess.run(
        [node, "--input-type=module", "--eval", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"node failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result.stdout.strip()


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


def test_resample_trajectory_reaches_the_final_segment(viewer):
    function = viewer_function(viewer, "resampleTrajectory")

    output = run_node(
        f"""
        {function}
        const samples = resampleTrajectory({{
          positions: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
          forwards: [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        }}, 11);
        process.stdout.write(JSON.stringify(samples.at(-1).position));
        """
    )

    assert json.loads(output) == [0, 1, 0]


def test_stale_trajectory_fetch_does_not_mutate_the_newer_scene(viewer):
    function = viewer_function(viewer, "loadScene")

    output = run_node(
        f"""
        const deferred = () => {{
          let resolve;
          const promise = new Promise(settle => {{ resolve = settle; }});
          return {{ promise, resolve }};
        }};
        const manifestA = deferred();
        const trajectoryA = deferred();
        const manifestB = deferred();
        const state = {{
          loadSequence: 0, activeLoadToken: 0, manifest: null, manifestUrl: null,
          trajectory: null, resampledTrajectory: null, pathDistances: null,
          totalPathLength: 0, loadedPointCount: 0, loadedChunkCount: 0,
          currentChunkFile: null, mode: 'orbit', playing: false, progress: 0,
        }};
        const elements = {{
          manifestInput: {{ value: 'a.json' }},
          trajectoryInput: {{ value: '', dataset: {{}} }},
          playbackProgress: {{ value: '' }},
          playbackProgressValue: {{ textContent: '' }},
          playPauseButton: {{ textContent: '', disabled: false }},
          driveButton: {{ disabled: false }},
        }};
        function setStatus() {{}}
        function resolveUrl(value, base = 'https://example.test/') {{ return new URL(value, base); }}
        function validateManifest(value) {{ return value; }}
        async function fetchJson(url) {{
          if (url.pathname.endsWith('/a.json')) return manifestA.promise;
          if (url.pathname.endsWith('/a-trajectory.json')) return trajectoryA.promise;
          if (url.pathname.endsWith('/b.json')) return manifestB.promise;
          throw new Error(`unexpected URL ${{url}}`);
        }}
        async function ensureThree() {{}}
        function initRenderer() {{}}
        function clearScene() {{
          state.manifest = null;
          state.trajectory = null;
          state.resampledTrajectory = null;
        }}
        function buildTrajectory(payload) {{ return payload; }}
        function resampleTrajectory() {{ return []; }}
        function frameScene() {{}}
        function buildTrajectoryObjects() {{ return {{}}; }}
        function updateTrajectoryMarker() {{}}
        function setMode(mode) {{ state.mode = mode; }}
        async function loadChunks() {{}}
        {function}
        const flush = async () => {{
          for (let index = 0; index < 20; index += 1) await Promise.resolve();
        }};
        const loadA = loadScene();
        manifestA.resolve({{ trajectory: 'a-trajectory.json', chunks: [], point_count: 0 }});
        await flush();
        elements.manifestInput.value = 'b.json';
        const loadB = loadScene();
        manifestB.resolve({{ chunks: [], point_count: 0 }});
        await Promise.all([loadB, flush()]);
        trajectoryA.resolve({{
          format: 'abot-point-cloud-trajectory',
          version: 1,
          positions: [[9, 9, 9]],
          forwards: [[1, 0, 0]],
        }});
        await Promise.all([loadA, flush()]);
        process.stdout.write(JSON.stringify(state.trajectory));
        """
    )

    assert json.loads(output) is None


def test_renderer_requests_linear_srgb_output(viewer):
    function = viewer_function(viewer, "initRenderer")

    output = run_node(
        f"""
        class FakeRenderer {{
          constructor() {{ this.outputColorSpace = null; }}
          setPixelRatio() {{}}
          setClearColor() {{}}
        }}
        class FakeMaterial {{
          constructor(options) {{ Object.assign(this, options); this.size = 0; }}
        }}
        const THREE = {{
          LinearSRGBColorSpace: 'linear-srgb',
          NoBlending: 0,
          PointsMaterial: FakeMaterial,
          WebGLRenderer: FakeRenderer,
          Scene: class {{ add() {{}} }},
          PerspectiveCamera: class {{
            constructor() {{ this.up = {{ set() {{}} }}; }}
          }},
          Group: class {{}},
          Vector3: class {{}},
        }};
        let renderer = null;
        let scene = null;
        let camera = null;
        let pointGroup = null;
        let trajectoryGroup = null;
        let pointMaterial = null;
        let circleTexture = null;
        let rendererInitialized = false;
        const elements = {{ pointSize: {{ value: '0.08' }} }};
        const canvas = {{
          addEventListener() {{}},
          setPointerCapture() {{}},
          hasPointerCapture() {{ return false; }},
          releasePointerCapture() {{}},
        }};
        const window = {{
          devicePixelRatio: 1,
          innerWidth: 800,
          innerHeight: 600,
          addEventListener() {{}},
        }};
        const cameraController = {{ target: null }};
        function makeCircleSprite() {{ return {{}}; }}
        function resize() {{}}
        function render() {{}}
        function requestAnimationFrame() {{}}
        function applyOrbit() {{}}
        function panCamera() {{}}
        function orbitCamera() {{}}
        {function}
        initRenderer();
        process.stdout.write(String(renderer.outputColorSpace));
        """
    )

    assert output == "linear-srgb"


def test_switching_to_orbit_stops_playback_and_updates_mode_state(viewer):
    function = viewer_function(viewer, "setMode")

    output = run_node(
        f"""
        const state = {{ mode: 'drive', trajectory: {{}}, playing: true }};
        const cameraController = {{ enabled: false }};
        const elements = {{
          orbitButton: {{
            classList: {{ toggle() {{}} }},
            setAttribute(name, value) {{ this[name] = value; }},
          }},
          driveButton: {{
            classList: {{ toggle() {{}} }},
            setAttribute(name, value) {{ this[name] = value; }},
          }},
          playPauseButton: {{ disabled: false, textContent: 'Pause' }},
        }};
        function setPlaying(playing) {{
          state.playing = playing;
          elements.playPauseButton.textContent = playing ? 'Pause' : 'Play';
        }}
        function applyOrbit() {{}}
        function applyDriveCamera() {{}}
        {function}
        setMode('orbit');
        process.stdout.write(JSON.stringify({{
          playing: state.playing,
          button: elements.playPauseButton.textContent,
          orbitPressed: elements.orbitButton['aria-pressed'],
          drivePressed: elements.driveButton['aria-pressed'],
        }}));
        """
    )

    assert json.loads(output) == {
        "playing": False,
        "button": "Play",
        "orbitPressed": "true",
        "drivePressed": "false",
    }


def test_manifest_query_with_dataset_uses_catalog_trajectory(viewer):
    function = viewer_function(viewer, "applyQueryParameters")

    output = run_node(
        f"""
        const datasetOption = {{
          value: 'demo',
          dataset: {{
            manifest: '/catalog/manifest.json',
            trajectory: '/catalog/trajectory.json',
          }},
        }};
        const elements = {{
          manifestInput: {{ value: '' }},
          trajectoryInput: {{ value: '', dataset: {{}} }},
          dataset: {{ value: 'custom', options: [datasetOption] }},
        }};
        const window = {{
          location: {{
            search: '?manifest=%2Fcustom%2Fmanifest.json&dataset=demo',
          }},
        }};
        function setStatus() {{}}
        function applyDatasetSelection() {{}}
        {function}
        const shouldLoad = applyQueryParameters();
        process.stdout.write(JSON.stringify({{
          shouldLoad,
          manifest: elements.manifestInput.value,
          trajectory: elements.trajectoryInput.value,
          dataset: elements.dataset.value,
        }}));
        """
    )

    assert json.loads(output) == {
        "shouldLoad": True,
        "manifest": "/custom/manifest.json",
        "trajectory": "/catalog/trajectory.json",
        "dataset": "demo",
    }


def test_viewer_omits_unused_reload_state_and_chunk_argument(viewer):
    assert "expectedPointCount" not in viewer
    assert "manifestUrl" not in viewer_function(viewer, "loadChunks")


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
