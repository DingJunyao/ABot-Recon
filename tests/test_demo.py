from pathlib import Path

import pytest

from demo import build_parser


def test_demo_accepts_video_input_without_image_dir():
    args = build_parser().parse_args(["--video", "input.mp4"])

    assert args.video == Path("input.mp4")
    assert args.image_dir == Path("examples/images")


def test_demo_rejects_video_and_image_dir_at_the_same_time():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--video", "input.mp4", "--image-dir", "images"])


def test_extract_video_frames_selects_and_converts_requested_frames(tmp_path):
    import numpy as np
    from PIL import Image

    from demo import extract_video_frames

    frames = [
        np.array([[[255, 0, 0]]], dtype=np.uint8),
        np.array([[[0, 255, 0]]], dtype=np.uint8),
        np.array([[[0, 0, 255]]], dtype=np.uint8),
        np.array([[[255, 255, 255]]], dtype=np.uint8),
    ]
    factory_calls = []

    class FakeCapture:
        def __init__(self, path):
            factory_calls.append(path)
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def read(self):
            try:
                return True, next(self.frames)
            except StopIteration:
                return False, None

        def release(self):
            pass

    selected = extract_video_frames(
        Path("input.mp4"),
        tmp_path,
        start=1,
        end=4,
        stride=2,
        video_capture_factory=FakeCapture,
    )

    assert selected == [tmp_path / "00000000.jpg", tmp_path / "00000001.jpg"]
    assert factory_calls == [Path("input.mp4")]
    colors = [Image.open(path).getpixel((0, 0)) for path in selected]
    assert all(
        abs(actual[channel] - expected[channel]) <= 2
        for actual, expected in zip(colors, [(0, 255, 0), (255, 255, 255)])
        for channel in range(3)
    )


def test_main_runs_video_frames_from_a_temporary_directory(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from PIL import Image

    import demo

    model_calls = []
    extract_calls = []
    save_calls = []
    temporary_dirs = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model_calls.append((args, kwargs))
            return cls()

        def infer(self, images, **kwargs):
            model_calls.append((list(images), kwargs))
            return SimpleNamespace()

    def fake_extract(video_path, output_dir, *, start, end, stride, video_capture_factory=None):
        extract_calls.append((video_path, output_dir, start, end, stride))
        temporary_dirs.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / "00000000.jpg", output_dir / "00000001.jpg"]
        for path in paths:
            Image.new("RGB", (1, 1)).save(path)
        return paths

    def fake_save(output_dir, result, images, dense_indices):
        save_calls.append((output_dir, list(images), dense_indices))

    monkeypatch.setattr(demo, "ABotRecon", FakeModel)
    monkeypatch.setattr(demo, "extract_video_frames", fake_extract)
    output_dir = tmp_path / "result"
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake video")
    monkeypatch.setattr(demo, "save_result", fake_save)
    monkeypatch.setattr(
        "sys.argv",
        [
            "demo.py",
            "--video",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--no-loop-closure",
        ],
    )

    demo.main()

    assert len(extract_calls) == 1
    assert extract_calls[0][0] == video_path
    assert extract_calls[0][2:] == (0, None, 1)
    expected_paths = [
        extract_calls[0][1] / "00000000.jpg",
        extract_calls[0][1] / "00000001.jpg",
    ]
    assert model_calls[1][0] == expected_paths
    assert save_calls[0][1] == model_calls[1][0]
    assert not extract_calls[0][1].exists()
    assert temporary_dirs == [extract_calls[0][1]]


def test_extract_video_frames_rejects_unreadable_video(tmp_path):
    from demo import extract_video_frames

    releases = []

    class UnreadableCapture:
        def isOpened(self):
            return False

        def read(self):
            return False, None

        def release(self):
            releases.append(True)

    with pytest.raises(ValueError, match="Could not open video"):
        extract_video_frames(
            Path("input.mp4"),
            tmp_path,
            start=0,
            end=None,
            stride=1,
            video_capture_factory=lambda path: UnreadableCapture(),
        )

    assert releases == [True]


def test_missing_opencv_error_mentions_video_extra(monkeypatch, tmp_path):
    import builtins

    from demo import extract_video_frames

    real_import = builtins.__import__

    def import_without_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("cv2 is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cv2)

    with pytest.raises(RuntimeError, match=r"\.\[video\]"):
        extract_video_frames(
            Path("input.mp4"),
            tmp_path,
            start=0,
            end=None,
            stride=1,
        )
