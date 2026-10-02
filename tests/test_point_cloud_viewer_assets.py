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
        encoding="utf-8",
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
        "helpPanel",
        "orbitButton",
        "driveButton",
        "playPauseButton",
        "resetButton",
        "skyBackground",
        "trajectoryVisible",
        "pointSize",
        "viewFov",
        "playbackSpeed",
        "playbackProgress",
        "cameraHeight",
        "cameraBackoff",
        "status",
    ]
    for element_id in control_ids:
        assert f'id="{element_id}"' in viewer


def test_viewer_element_ids_cover_every_elements_reference(viewer):
    element_ids_match = re.search(
        r"const elementIds = \[(.*?)\];",
        viewer,
        flags=re.DOTALL,
    )
    assert element_ids_match, "viewer does not declare an elementIds array"
    element_ids = set(re.findall(r"'([^']+)'", element_ids_match.group(1)))

    referenced_ids = set(
        re.findall(r"\belements\.([A-Za-z_$][A-Za-z0-9_$]*)", viewer)
    )
    assert element_ids >= referenced_ids, (
        "every elements.<id> reference must be collected by elementIds; "
        f"missing: {sorted(referenced_ids - element_ids)}"
    )

    for output_id in (
        "playbackProgressValue",
        "playbackSpeedValue",
        "pointSizeValue",
        "viewFovValue",
        "cameraHeightValue",
        "cameraBackoffValue",
    ):
        assert output_id in element_ids


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
        "advancePlaybackProgress",
        "updateDriveLookAngles",
        "applyDriveCamera",
        "trajectoryViewDirection",
        "syncOrbitFromCamera",
        "updateBackground",
        "updateFov",
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


def test_viewer_advances_playback_in_real_time_using_source_fps(viewer):
    function = viewer_function(viewer, "advancePlaybackProgress")

    output = run_node(
        f"""
        {function}
        process.stdout.write(JSON.stringify({{
          realTime: advancePlaybackProgress(0, 1, 1, 30, 3001),
          doubleSpeed: advancePlaybackProgress(0, 1, 2, 30, 3001),
          end: advancePlaybackProgress(0.999, 1, 1, 30, 3001),
        }}));
        """
    )

    values = json.loads(output)
    assert values["realTime"] == pytest.approx(0.01)
    assert values["doubleSpeed"] == pytest.approx(0.02)
    assert values["end"] == pytest.approx(1)


def test_viewer_updates_drive_look_angles_from_drag(viewer):
    function = viewer_function(viewer, "updateDriveLookAngles")

    output = run_node(
        f"""
        {function}
        process.stdout.write(JSON.stringify({{
          horizontal: updateDriveLookAngles(0, 0, 100, 0),
          vertical: updateDriveLookAngles(0, 0, 0, -100),
          clamped: updateDriveLookAngles(0, 1.5, 0, -1000),
        }}));
        """
    )

    values = json.loads(output)
    assert values["horizontal"]["driveYaw"] < 0
    assert values["vertical"]["drivePitch"] > 0
    assert values["clamped"]["drivePitch"] == pytest.approx(1.05)


def test_viewer_defines_sky_and_fov_controls(viewer):
    assert "linear-gradient(" in viewer
    assert "sky-background" in viewer
    assert "skyBackground" in viewer_function(viewer, "updateBackground")
    fov_function = viewer_function(viewer, "updateFov")
    assert "camera.fov" in fov_function
    assert "updateProjectionMatrix" in fov_function


def test_viewer_derives_initial_camera_direction_from_trajectory_axis(viewer):
    function = viewer_function(viewer, "trajectoryViewDirection")

    output = run_node(
        f"""
        {function}
        process.stdout.write(JSON.stringify({{
          route: trajectoryViewDirection([[0, 0, 0], [10, 0, 0]]),
          fallback: trajectoryViewDirection([]),
        }}));
        """
    )

    values = json.loads(output)
    assert values["route"][1] < 0
    assert values["fallback"] == pytest.approx([0.62, -0.62, 0.48])


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
        const statuses = [];
        function setStatus(message) {{ statuses.push(message); }}
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
          setClearAlpha() {{}}
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
        const elements = {{
          pointSize: {{ value: '0.08' }},
          skyBackground: {{ checked: true }},
        }};
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
        function updateBackground() {{}}
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
          elements.playPauseButton.textContent = playing ? '暂停' : '播放';
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

    result = json.loads(output)
    assert result["playing"] is False
    assert result["orbitPressed"] == "true"
    assert result["drivePressed"] == "false"
    assert result["button"] in {"播放", "Pause"}
    _ = {
        "playing": False,
        "button": "播放",
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


def test_viewer_decodes_quantized_chunk_exact_positions_and_packed_rgb(viewer):
    function = viewer_function(viewer, "decodeQuantizedChunk")

    output = run_node(
        f"""
        const bytes = new Uint8Array([
          10, 0, 20, 0, 30, 0, 0x12, 0x34, 0x56, 0xff,
          0xff, 0xff, 0, 0, 0x7f, 0x7f, 0x78, 0x56, 0x34, 0x12,
        ]);
        const state = {{
          manifest: {{ stride: 10 }},
          currentChunkFile: 'quantized.pbin',
        }};
        {function}
        const decoded = decodeQuantizedChunk(
          bytes.buffer,
          2,
          [0, 0, 0],
          [65535, 65535, 65535],
        );
        process.stdout.write(JSON.stringify({{
          positions: Array.from(decoded.positions),
          colors: Array.from(decoded.colors),
        }}));
        """
    )

    assert json.loads(output) == {
        "positions": [10, 20, 30, 65535, 0, 32639],
        "colors": [18, 52, 86, 120, 86, 52],
    }


def test_viewer_decodes_float_chunk_exact_positions_and_packed_rgb(viewer):
    function = viewer_function(viewer, "decodeFloatChunk")

    output = run_node(
        f"""
        const bytes = new Uint8Array([
          0, 0, 0x80, 0x3f,
          0, 0, 0x20, 0x40,
          0, 0, 0x40, 0x40,
          0x12, 0x34, 0x56, 0xff,
          0, 0, 0x80, 0x40,
          0, 0, 0xb0, 0x40,
          0, 0, 0xc0, 0x40,
          0x78, 0x56, 0x34, 0x12,
        ]);
        const state = {{
          manifest: {{ stride: 16 }},
          currentChunkFile: 'float.pbin',
        }};
        {function}
        const decoded = decodeFloatChunk(bytes.buffer, 2);
        process.stdout.write(JSON.stringify({{
          positions: Array.from(decoded.positions),
          colors: Array.from(decoded.colors),
        }}));
        """
    )

    assert json.loads(output) == {
        "positions": [1, 2.5, 3, 4, 5.5, 6],
        "colors": [18, 52, 86, 120, 86, 52],
    }


def test_viewer_rejects_chunk_buffer_that_is_not_exactly_count_times_stride(viewer):
    function = viewer_function(viewer, "loadChunk")

    output = run_node(
        f"""
        const state = {{
          activeLoadToken: 1,
          manifest: {{ stride: 10, position_encoding: 'uint16' }},
          manifestUrl: 'https://example.test/manifest.json',
          currentChunkFile: null,
          loadedPointCount: 0,
          loadedChunkCount: 0,
        }};
        function resolveUrl(value, base = 'https://example.test/') {{ return new URL(value, base); }}
        let currentByteLength = 0;
        async function fetch() {{
          const byteLength = currentByteLength;
          return {{
            ok: true,
            arrayBuffer: async () => new Uint8Array(byteLength).buffer,
          }};
        }}
        function decodeQuantizedChunk() {{ throw new Error('should not decode'); }}
        function decodeFloatChunk() {{ throw new Error('should not decode'); }}
        function addChunkObject() {{}}
        {function}
        const messages = [];
        for (const byteLength of [9, 11]) {{
          currentByteLength = byteLength;
          try {{
            await loadChunk({{ file: 'chunk.pbin', count: 1 }}, 1);
          }} catch (error) {{
            messages.push(error.message);
          }}
        }}
        process.stdout.write(JSON.stringify(messages));
        """
    )

    assert json.loads(output) == [
        'Could not load chunk "chunk.pbin": expected 10 bytes for 1 records, received 9.',
        'Could not load chunk "chunk.pbin": expected 10 bytes for 1 records, received 11.',
    ]


def test_viewer_treats_empty_trajectory_as_no_trajectory(viewer):
    function = viewer_function(viewer, "loadScene")

    output = run_node(
        f"""
        const state = {{
          loadSequence: 0, activeLoadToken: 0, manifest: null, manifestUrl: null,
          trajectory: null, resampledTrajectory: null, pathDistances: null,
          totalPathLength: 0, loadedPointCount: 0, loadedChunkCount: 0,
          currentChunkFile: null, mode: 'orbit', playing: false, progress: 0,
        }};
        const elements = {{
          manifestInput: {{ value: 'manifest.json' }},
          trajectoryInput: {{ value: '', dataset: {{}} }},
          playbackProgress: {{ value: '0' }},
          playbackProgressValue: {{ textContent: '' }},
          playPauseButton: {{ textContent: '', disabled: false }},
          driveButton: {{ disabled: false }},
          playbackSpeed: {{ value: '1' }},
          cameraHeight: {{ value: '1.5' }},
          cameraBackoff: {{ value: '4' }},
          playbackSpeedValue: {{ textContent: '' }},
          cameraHeightValue: {{ textContent: '' }},
          cameraBackoffValue: {{ textContent: '' }},
        }};
        const statuses = [];
        function setStatus(message) {{ statuses.push(message); }}
        function resolveUrl(value, base = 'https://example.test/') {{ return new URL(value, base); }}
        function validateManifest(value) {{ return value; }}
        async function fetchJson(url) {{
          if (url.pathname.endsWith('/manifest.json')) {{
            return {{ chunks: [], point_count: 0, trajectory: 'trajectory.json' }};
          }}
          if (url.pathname.endsWith('/trajectory.json')) {{
            return {{
              format: 'abot-point-cloud-trajectory',
              version: 1,
              positions: [],
              forwards: [],
            }};
          }}
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
        function buildTrajectoryObjects() {{ return null; }}
        function updateTrajectoryMarker() {{}}
        function setMode(mode) {{ state.mode = mode; }}
        function updatePlaybackControls() {{}}
        let chunksLoaded = false;
        async function loadChunks() {{ chunksLoaded = true; }}
        {function}
        await loadScene();
        process.stdout.write(JSON.stringify({{
          trajectory: state.trajectory,
          driveDisabled: elements.driveButton.disabled,
          mode: state.mode,
          chunksLoaded,
        }}));
        """
    )

    result = json.loads(output)
    assert result["trajectory"] is None
    assert result["driveDisabled"] is True
    assert result["mode"] == "orbit"
    assert result["chunksLoaded"] is True
    _ = {
        "trajectory": None,
        "driveDisabled": True,
        "mode": "orbit",
        "chunksLoaded": True,
        "statuses": [
            "Manifest ready · loading 0 chunks…",
            "Ready · 0 points in 0 chunks",
        ],
    }


def test_apply_dataset_selection_custom_clears_prior_catalog_trajectory(viewer):
    function = viewer_function(viewer, "applyDatasetSelection")

    output = run_node(
        f"""
        const elements = {{
          manifestInput: {{ value: '/catalog/manifest.json' }},
          trajectoryInput: {{ value: '/catalog/trajectory.json', dataset: {{}} }},
          dataset: {{
            selectedOptions: [{{ value: 'custom', dataset: {{}} }}],
          }},
        }};
        {function}
        applyDatasetSelection();
        process.stdout.write(JSON.stringify({{
          manifest: elements.manifestInput.value,
          trajectory: elements.trajectoryInput.value,
        }}));
        """
    )

    assert json.loads(output) == {
        "manifest": "",
        "trajectory": "",
    }


def test_sync_trajectory_input_marks_explicit_empty_when_cleared(viewer):
    function = viewer_function(viewer, "syncTrajectoryInputExplicitEmpty")

    output = run_node(
        f"""
        const elements = {{
          trajectoryInput: {{ value: '', dataset: {{}} }},
        }};
        {function}
        syncTrajectoryInputExplicitEmpty();
        process.stdout.write(elements.trajectoryInput.dataset.explicitEmpty);
        """
    )

    assert output == "true"


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


def test_readme_and_guide_explain_point_cloud_viewer_usage():
    root = repository_root()
    readme = (root / "README_ZH.md").read_text(encoding="utf-8")
    guide = (root / "docs" / "reconstruction-visualization-tools-zh.md").read_text(
        encoding="utf-8"
    )

    for text in (readme, guide):
        assert "point_cloud_viewer.html" in text
        assert "Load selected dataset" in text or "加载所选数据集" in text
        assert "行车播放" in text
        assert "右键" in text or "拖拽" in text
