# Streaming RGB Point-Cloud Viewer Design

Date: 2026-10-01
Status: Approved for specification review

## Goal

Add an official-Hugging-Face-demo-style viewer for long, dense ABot-Recon results such as `outputs/demo_loop/reconstruction.ply`, without inheriting the online demo's 200-frame inference or 500,000-point browser limits.

The viewer must:

- render the original colored point cloud rather than pseudo-Gaussian splats;
- support results with tens of millions of points;
- support interactive orbit/pan/zoom viewing;
- support trajectory playback and scrubbing;
- preserve or improve the official demo's visual framing and camera-trajectory affordances;
- load progressively so useful geometry appears before every chunk has been downloaded.

## Scope

### In Scope

- A converter from an existing binary RGB PLY to compact streamed chunks.
- A browser viewer using three.js hard points.
- Rainbow trajectory visualization from camera poses.
- Orbit and drive-camera modes.
- Unit tests for conversion and manifest generation.
- Local HTTP serving through the existing repository-wide static server.
- Chinese documentation for the recommended workflow.

### Out of Scope

- Re-running inference.
- Producing or optimizing true 3D Gaussian splats.
- Browser-side PLY editing or export.
- Changing the inference output format.
- Network-hosted deployment.

## Data Flow

1. Run inference and export a binary-little-endian RGB PLY with the existing tools.
2. Run `scripts/export_point_cloud_chunks.py`:
   - input: RGB `reconstruction.ply`;
   - input: optional `camera_poses*.npy`;
   - output: chunk directory containing compact binary chunks;
   - output: `manifest.json`;
   - output: optional `trajectory.json` derived from poses.
3. Start `scripts/serve_pseudo_gaussian_viewer.py --no-browser`, which already serves the repository root.
4. Open `tools/point_cloud_viewer.html` with query parameters pointing to the manifest.
5. The viewer fetches the manifest and trajectory first, then progressively loads chunks and adds each visible chunk to the three.js scene.

## Converter

### CLI

```text
python scripts/export_point_cloud_chunks.py \
  --input outputs/demo_loop/reconstruction.ply \
  --poses outputs/demo_loop/camera_poses.npy \
  --output-dir outputs/demo_loop/point_cloud_chunks \
  --chunk-size 250000 \
  --max-points 0 \
  --position-encoding uint16
```

Arguments:

- `--input`: required binary RGB PLY.
- `--poses`: optional `[N,4,4]` camera-to-world NumPy array.
- `--output-dir`: required destination.
- `--chunk-size`: default `250000`; must be positive.
- `--max-points`: default `0`, meaning retain every finite point.
- `--position-encoding`: `uint16` or `float32`; default `uint16`.

### Supported Input

The converter accepts binary-little-endian PLY files with vertex properties:

```text
x, y, z, red, green, blue
```

It rejects:

- ASCII or big-endian PLY;
- list-valued vertex properties;
- missing XYZ or RGB properties;
- truncated vertex data;
- non-finite XYZ values.

Unsupported properties may be present and are ignored.

### Chunk Encoding

Each vertex uses 8 bytes in the default `uint16` mode:

```text
x: uint16
y: uint16
z: uint16
rgba: uint32, little-endian packed 0xAABBGGRR
```

The manifest stores scene bounds. The viewer reconstructs float positions with:

```text
world = min + normalized_uint16 * (max - min) / 65535
```

The optional `float32` mode stores:

```text
x: float32
y: float32
z: float32
rgba: uint32
```

This preserves the source PLY coordinates exactly but uses 16 bytes per point.

### Point Budget

When `--max-points` is greater than zero and the source has more finite points, the converter selects evenly spaced indices over the complete point sequence. This mirrors the existing reconstruction exporter and avoids taking only the beginning of a long route.

### Manifest

Example logical schema:

```json
{
  "format": "abot-point-cloud-chunks",
  "version": 1,
  "point_count": 25596210,
  "chunk_count": 103,
  "chunk_size": 250000,
  "position_encoding": "uint16",
  "stride": 8,
  "bounds": {
    "min": [0.0, 0.0, 0.0],
    "max": [100.0, 20.0, 100.0]
  },
  "robust_bounds": {
    "center": [50.0, 5.0, 50.0],
    "radius": 60.0
  },
  "chunks": [
    {
      "file": "chunk-00000.pbin",
      "offset": 0,
      "count": 250000,
      "bounds": {"min": [0,0,0], "max": [1,1,1]}
    }
  ]
}
```

`robust_bounds` is calculated from a bounded sample using coordinate-wise medians and a high distance percentile. It drives automatic framing so a few far-away outliers do not shrink the entire scene. Per-chunk bounds permit future frustum or LOD work without changing this format.

## Trajectory

When `--poses` is supplied, the converter writes `trajectory.json` beside the manifest:

```json
{
  "format": "abot-point-cloud-trajectory",
  "version": 1,
  "coordinate_system": "original ABot-Recon world coordinates",
  "positions": [[0.0, 0.0, 0.0]],
  "forwards": [[0.0, 0.0, 1.0]]
}
```

- Positions are camera centers `pose[:3, 3]`.
- Forwards are normalized camera-local optical axes `pose[:3, 2]`.
- Non-finite poses are skipped.
- Missing poses leave trajectory and drive mode disabled, but orbit viewing remains available.
- The viewer resamples the polyline by cumulative arc length for stable playback.

## Viewer

### File

```text
tools/point_cloud_viewer.html
```

### Rendering

- three.js `Points` geometry, one object per chunk.
- Circular alpha-tested sprite, following the official demo.
- `vertexColors: true`.
- Depth testing enabled.
- No Gaussian alpha blending.
- Adaptive point size based on robust extent and loaded point count.
- Linear color output to match the official demo's direct RGB appearance.
- Progressive visibility: a chunk is added as soon as it downloads and uploads successfully.

### Camera Modes

Orbit mode:

- left drag rotates;
- right drag or Shift+left drag pans;
- wheel zooms;
- reset restores the automatic side/above view.

Drive mode:

- follows trajectory positions and optical directions;
- height and backoff controls remain adjustable;
- playback speed is adjustable;
- progress bar scrubs through the resampled trajectory;
- pause/resume and reset are available.

### Trajectory Display

- Rainbow line follows temporal order.
- Start and end are visible.
- Current camera marker moves during drive playback.
- Trajectory can be toggled without interrupting playback.

### Framing

Initial framing uses:

- robust point bounds;
- camera trajectory bounds;
- viewport aspect and field of view.

It chooses a side/above direction relative to the trajectory axis so long routes span the viewport rather than pointing directly into the camera.

### Errors

The viewer reports actionable messages for:

- unsupported WebGL context;
- missing or malformed manifest;
- unsupported manifest version or encoding;
- chunk HTTP failure;
- chunk byte-length mismatch;
- missing trajectory;
- three.js CDN failure.

Already loaded chunks remain visible after a later chunk fails.

## Dataset Catalog

Add:

```text
tools/point_cloud_datasets.json
```

It initially contains a `demo_loop` entry pointing to the generated chunks and poses. The viewer also accepts `manifest` and optional `trajectory` query parameters, so custom datasets do not require catalog changes.

## Testing

### Unit Tests

Add `tests/test_export_point_cloud_chunks.py` covering:

- binary RGB PLY parsing;
- finite-point filtering;
- exact chunk counts and sizes;
- `uint16` quantization and reconstruction bounds;
- packed RGB values;
- `float32` encoding;
- even point-budget sampling;
- manifest fields and chunk bounds;
- trajectory positions and normalized forwards;
- rejection of malformed or unsupported inputs;
- behavior without poses.

Run:

```text
python -m pytest tests/test_export_point_cloud_chunks.py
```

### Browser Verification

Use a small synthetic chunk set for functional checks, then `demo_loop` for scale checks. Verify:

- manifest and trajectory load;
- chunks appear progressively;
- orbit controls work;
- drive playback follows the path;
- scrubbing and reset work;
- no browser console errors occur.

## Performance Expectations

For the current 25,596,210-point `demo_loop` result:

- default wire size: about 205 MB;
- 103 chunks at 250,000 points;
- approximately 103 draw calls once fully loaded;
- no alpha sorting cost;
- lower memory and network use than the existing 24-byte pseudo-Gaussian chunks.

Actual performance depends on browser GPU memory. Users can create a lighter catalog entry with `--max-points` without changing the viewer.

## Documentation

Update the Chinese visualization guide to describe:

- when to use hard-point versus pseudo-Gaussian viewing;
- chunk conversion;
- serving;
- query parameters;
- controls;
- quality/performance tradeoffs.
