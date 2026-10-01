import pytest
import torch

from abot_recon.modeling.streaming.network import ABotReconNetwork


def _fake_network(forward):
    network = object.__new__(ABotReconNetwork)
    network.infer_mode = "stream"
    network.causal_global_attn = False
    network._can_use_paged_kv = lambda frame: False
    network.forward = forward
    network._postprocess_stream_carry = lambda carry, spatial_hw: None
    return network


def test_streaming_dense_outputs_are_offloaded_from_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    calls = 0

    def forward(frame, **kwargs):
        nonlocal calls
        calls += 1
        device = frame.device
        return {
            "camera_poses": torch.full((1, 1, 4), calls, device=device),
            "local_points": torch.full((1, 1, 3, 2, 2), calls, device=device),
            "conf": torch.full((1, 1, 2, 2), calls, device=device),
            "past_key_values": None,
            "ref_hidden": None,
            "camera_state": None,
        }

    network = _fake_network(forward)
    frames = iter([torch.zeros(1, 1, 3, 2, 2, device="cuda") for _ in range(2)])

    output = ABotReconNetwork.inference_stream_iter(
        network,
        frames,
        num_frames=2,
        output_keys=["camera_poses", "local_points", "conf"],
    )

    assert calls == 2
    assert output["camera_poses"].device.type == "cuda"
    assert output["local_points"].shape == (1, 2, 3, 2, 2)
    assert output["local_points"].device.type == "cpu"
    assert output["conf"].shape == (1, 2, 2, 2)
    assert output["conf"].device.type == "cpu"
