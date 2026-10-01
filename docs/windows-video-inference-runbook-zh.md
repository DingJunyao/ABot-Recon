# Windows 视频推理与长视频排障 Runbook

本文是面向实际桌面/工作站环境的 **操作手册**，记录从视频文件生成点云时的检查顺序、推荐命令和已遇到的运行期问题。根目录 `README_ZH.md` 介绍通用安装和标准用法；本文补充现场排障、Windows/CUDA 环境验证、长视频显存控制以及失败后的恢复策略。

## 1. 适用场景与输入

- 输入为一个视频文件，例如 `E:/DJI_0001_min.mp4`。
- 目标是输出相机轨迹、局部点图、可选世界坐标点云和颜色。
- 主要验证环境为 Windows、PyTorch `2.5.1+cu121`、RTX 4060 Ti 16 GB。
- 视频解码依赖 OpenCV；`--video` 与 `--image-dir` 互斥。

确认 OpenCV 已安装：

```powershell
python -c "import cv2; print(cv2.__version__)"
```

如未安装：

```powershell
python -m pip install -e ".[video]"
```

## 2. 运行前环境检查

### 2.1 确认使用的是项目虚拟环境

PowerShell 提示符应类似：

```text
(ABot-Recon) PS D:\code\ABot-Recon>
```

也可以直接使用项目内 Python：

```powershell
D:\code\ABot-Recon\.venv\Scripts\python.exe --version
```

> 重要：当前环境的 CUDA 版 PyTorch 是手动按 PyTorch CUDA 索引安装的。不要随意执行 `uv run` 或 `uv sync`，否则 uv 可能按默认锁文件解析并把 `torch` 替换成 CPU 版。测试和推理请使用已激活环境中的 `python`，或显式使用 `.venv\Scripts\python.exe`。

### 2.2 确认 PyTorch 是 CUDA 版

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

期望输出包含：

```text
2.5.1+cu121 12.1 True
```

如果看到：

```text
2.5.1+cpu None False
```

说明当前环境被替换成了 CPU 版 PyTorch，推理默认 `--device cuda` 会在模型初始化时报：

```text
AssertionError: Torch not compiled with CUDA enabled
```

修复方式是重装 CUDA 12.1 版：

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 确认推理使用 RTX 4060 Ti

不同工具的 GPU 编号可能不一致，不要直接假设 `nvidia-smi` 的编号就是 PyTorch 的 `cuda:1`。先执行：

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

期望：

```text
NVIDIA GeForce RTX 4060 Ti
```

如果需要限制设备，再用 `CUDA_VISIBLE_DEVICES`，并重新用上述命令确认当前 `cuda:0` 对应的物理 GPU。

### 2.4 确认本地 checkpoint

长任务建议显式传入本地 checkpoint，避免启动时访问 Hugging Face：

```powershell
Test-Path checkpoints\abot_recon.safetensors
```

期望输出：

```text
True
```

## 3. 先做短片段冒烟测试

不要直接把数小时的长视频完整跑完。先截取前 10 帧，验证视频解码、模型加载、attention、稠密输出转 CPU 和落盘流程：

```powershell
python demo.py `
  --video E:/DJI_0001_min.mp4 `
  --checkpoint checkpoints/abot_recon.safetensors `
  --output-dir outputs/demo_smoke_10 `
  --attention-backend auto `
  --device cuda `
  --no-loop-closure `
  --end 10 `
  --dense-stride 2 `
  --no-save-confidence
```

成功标志：

```text
Processed ... frames; results saved to outputs/demo_smoke_10
```

检查输出：

```powershell
Get-ChildItem outputs\demo_smoke_10
```

至少应看到：

```text
local_points.pt
colors.pt
camera_poses.npy
metadata.json
```

## 4. 完整长视频推荐命令

### 4.1 只要局部点图

这是第一轮完整跑视频的推荐配置：

```powershell
python demo.py `
  --video E:/DJI_0001_min.mp4 `
  --checkpoint checkpoints/abot_recon.safetensors `
  --output-dir outputs/demo_dense4 `
  --attention-backend auto `
  --device cuda `
  --no-loop-closure `
  --dense-stride 4 `
  --no-save-confidence
```

参数理由：

- `--dense-stride 4`：每 4 个选中帧保存一次点图，降低 CPU/GPU 峰值和输出体积。
- `--no-save-confidence`：阈值为 `0.0` 时置信度不会参与过滤，但默认保存会显著增加输出体积。
- `--no-loop-closure`：先验证基础流式重建；回环依赖更多组件，适合在基础结果成功后再启用。

### 4.2 需要世界坐标点云

世界坐标点云会额外保存一份变换后的稠密结果。第一轮建议更保守：

```powershell
python demo.py `
  --video E:/DJI_0001_min.mp4 `
  --checkpoint checkpoints/abot_recon.safetensors `
  --output-dir outputs/demo_world_dense8 `
  --attention-backend auto `
  --device cuda `
  --no-loop-closure `
  --dense-stride 8 `
  --no-save-confidence `
  --save-world-points
```

成功后再根据点云密度和系统内存情况尝试 `--dense-stride 4`。

### 4.3 运行时间过长时降低输入帧率

若视频很长且帧率高，可以用 `--stride 2` 每两帧取一帧：

```powershell
python demo.py `
  --video E:/DJI_0001_min.mp4 `
  --checkpoint checkpoints/abot_recon.safetensors `
  --output-dir outputs/demo_stride2_dense4 `
  --attention-backend auto `
  --device cuda `
  --no-loop-closure `
  --stride 2 `
  --dense-stride 4 `
  --no-save-confidence `
  --save-world-points
```

含义：

- `--stride 2`：位姿推理输入帧数约减少一半。
- `--dense-stride 4`：每 4 个选中帧保存一次稠密输出。
- 总体大约每 8 个原始视频帧保存一次稠密点图。

代价是时间采样密度和点云密度下降。

## 5. 已遇到的问题与处理

### 5.1 `AssertionError: Torch not compiled with CUDA enabled`

根因：环境里是 CPU 版 PyTorch，而 `demo.py` 默认 `--device cuda`。

处理：

1. 按 2.2 重装 `torch==2.5.1+cu121` 和 `torchvision==0.20.1+cu121`。
2. 重新确认 `torch.cuda.is_available()` 为 `True`。
3. 重启 Python 进程后再运行 `demo.py`。

临时验证可用 `--device cpu`，但大模型 CPU 推理非常慢，不适合完整视频。

### 5.2 `RuntimeError: No available kernel. Aborting execution.`

根因：Windows 版 PyTorch 2.5.1 未编译 FlashAttention backend，而旧代码在 bf16 attention 中只允许 FlashAttention，导致无法回退。

当前代码已将 SDPA backend 改为：

```python
[
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
```

这样支持 FlashAttention 的环境仍优先使用 FlashAttention；Windows 环境会自动回退到可用 kernel。

回归测试：

```powershell
python -m pytest tests\test_attention_backend.py --basetemp .pytest_attention_tmp
```

### 5.3 长视频推理到 100% 后 CUDA OOM

典型错误：

```text
RuntimeError: CUDA error: out of memory
...
stacked[k] = torch.cat(vals, dim=1)
```

根因：旧实现把所有逐帧 `local_points` / `confidence` 留在 GPU 列表中，最后一次性 `torch.cat`。长视频会在进度条完成后汇总阶段才 OOM。

当前代码已修复：稠密输出每帧产生后立即转移到 CPU，最终拼接不再占用 CUDA 显存。

回归测试：

```powershell
python -m pytest tests\test_streaming_memory.py --basetemp .pytest_stream_tmp
```

注意：即使 GPU OOM 已修复，`--dense-stride 1` 仍会产生很大的 CPU 内存和磁盘压力。长视频继续推荐 `--dense-stride 4` 或 `8`，并使用 `--no-save-confidence`。

### 5.4 推理完成后输出目录不存在

`demo.py` 只有在 `model.infer()` 完整返回后才调用 `save_result()`。如果推理或最终汇总阶段失败，本次结果通常无法恢复，输出目录也可能不存在。

因此：

- 长任务前先做短片段冒烟测试。
- 不要在输出尚未落盘前关闭进程。
- 重要长任务建议分段运行，例如用 `--start` / `--end` 分批输出到不同目录。

### 5.5 `Warning, cannot find cuda-compiled version of RoPE2D`

这是性能警告，表示 RoPE2D 正在使用慢速 PyTorch 回退实现，不会直接导致失败。

如需编译 CUDA 版 cuRoPE2D，参见专项指南：

```text
docs/curope-windows-build-zh.md
```

编译成功并确认 `.pyd` 可导入后，可强制使用 CUDA 版：

```powershell
$env:ABOT_RECON_ROPE2D_BACKEND = "cuda"
```

如果扩展没有正确加载，该设置会让程序直接报错，而不是静默走慢速路径。

### 5.6 Hugging Face 下载缓慢或网络不可用

模型已下载到本地时，长任务应始终指定：

```text
--checkpoint checkpoints/abot_recon.safetensors
```

日志中的 `hf_xet` 提示只影响 Hugging Face 下载性能，可通过以下命令改善：

```powershell
python -m pip install hf_xet
```

它不是推理错误。

## 6. 输出检查

典型局部点图输出：

```powershell
Get-ChildItem outputs\demo_dense4
```

应包含：

```text
camera_poses.npy
camera_poses_noloop.npy
relative_poses.npy
relative_poses_noloop.npy
local_points.pt
colors.pt
metadata.json
```

如果启用 `--save-world-points`，还应包含：

```text
world_points.pt
```

如果启用置信度输出，还会包含：

```text
confidence.pt
confidence_mask.pt
```

查看 metadata：

```powershell
Get-Content outputs\demo_dense4\metadata.json
```

## 7. 运行前检查清单

在启动数小时完整任务前，逐项确认：

- [ ] `python -c "import cv2; ..."` 可以导入 OpenCV。
- [ ] PyTorch 为 `2.5.1+cu121` 且 `torch.cuda.is_available()` 为 `True`。
- [ ] `torch.cuda.get_device_name(0)` 显示目标 GPU。
- [ ] `checkpoints\abot_recon.safetensors` 存在。
- [ ] 已用 `--end 10` 或更多帧完成短片段冒烟测试。
- [ ] 输出目录中确认生成 `local_points.pt` 和 `colors.pt`。
- [ ] 长视频设置 `--dense-stride 4` 或 `8`。
- [ ] 阈值为 `0.0` 时使用 `--no-save-confidence`。
- [ ] 如已编译 cuRoPE2D，确认不再出现慢速 RoPE 警告。
- [ ] 磁盘空间足够保存完整输出。
- [ ] 长任务期间避免执行可能重装或替换 PyTorch 的环境命令。

## 8. 相关文档

- 通用安装、模型输出和 API：`README_ZH.md`
- Windows cuRoPE2D 编译细节：`docs/curope-windows-build-zh.md`
