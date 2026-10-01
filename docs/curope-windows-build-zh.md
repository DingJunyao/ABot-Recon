# Windows cuRoPE2D 编译指南（CUDA 12.1 / VS 2022）

本文是 **Windows 原生环境编译 cuRoPE2D 的专项指南**，适用于当前发布的 PyTorch `2.5.1+cu121` 组合。通用安装与快速开始请参见根目录 `README_ZH.md`；在 Windows 上编译 cuRoPE2D 时，如遇 MSVC / CUDA 版本冲突，以本文为准。

## 适用环境

- Windows 原生环境
- PyTorch：`2.5.1+cu121`
- CUDA Toolkit：`12.1`
- Visual Studio 2022 Community 及 C++ 桌面开发组件
- 目标 GPU：NVIDIA RTX 4060 Ti，compute capability `8.9`

同一台机器可以同时安装 CUDA 12.1 和 CUDA 13.0，但本扩展应使用 **CUDA 12.1** 编译，避免与 `cu121` 版 PyTorch 混用。

## 为什么 “已经使用 VS 2022” 仍可能失败

CUDA 12.1 的 `crt/host_config.h` 会检查 MSVC 编译器版本，大致条件为：

```cpp
#if _MSC_VER < 1910 || _MSC_VER >= 1940
#error -- unsupported Microsoft Visual Studio version!
#endif
```

也就是说，CUDA 12.1 接受 `_MSC_VER` 从 `1910` 到 `1939` 的编译器。

较新的 VS 2022 可能默认安装并选择 MSVC `14.44`，对应 `_MSC_VER 1944`。因此即使使用的是 VS 2022，仍可能看到：

```text
fatal error C1189: #error: -- unsupported Microsoft Visual Studio version!
```

这不是没有找到 VS 2022，而是 **VS 2022 中的 MSVC 编译器版本太新，超出 CUDA 12.1 支持范围**。

推荐安装并显式选择旧版 v143 工具集，例如：

```text
MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.38-17.8)
```

MSVC `14.38` 对应 `_MSC_VER 1938`，满足 CUDA 12.1 的检查条件。

## 1. 安装兼容的 MSVC 工具集

打开：

```text
Visual Studio Installer → Visual Studio Community 2022 → 修改
```

进入 **单个组件**，搜索：

```text
14.38
```

勾选并安装：

```text
MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.38-17.8)
```

该工具集会与现有 MSVC `14.44` 并存，不需要卸载 VS 2022 或现有编译器。

## 2. 用 MSVC 14.38 打开 VS 编译环境

以下命令必须运行在 `cmd` 中，不建议在普通 PowerShell 中编译：

```bat
cmd /k ""C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 -vcvars_ver=14.38"
```

激活项目虚拟环境：

```bat
cd /d D:\code\ABot-Recon
.venv\Scripts\activate.bat
```

如果你的环境是 conda 环境，则改用对应的 `conda activate` 命令。

## 3. 确认工具链版本

确认 MSVC：

```bat
cl
```

期望输出中的编译器版本小于 `19.40`，例如：

```text
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.xxxxx for x64
```

确认 CUDA 编译器：

```bat
nvcc --version
```

期望看到：

```text
Cuda compilation tools, release 12.1
```

如果 `nvcc` 显示 CUDA `13.0`，说明 CUDA 13.0 在 `PATH` 中优先级更高。可在同一命令行窗口中强制指向 CUDA 12.1：

```bat
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set PATH=%CUDA_PATH%\bin;%PATH%
```

然后再次执行 `nvcc --version` 确认。

确认 PyTorch 仍是 CUDA 12.1 版本：

```bat
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

期望输出：

```text
2.5.1+cu121 12.1 True
```

如果输出是 `2.5.1+cpu` 或 CUDA unavailable，先修复 PyTorch 环境，不要继续编译扩展。

## 4. 编译 cuRoPE2D

安装构建工具：

```bat
python -m pip install --upgrade setuptools wheel ninja
```

进入扩展目录并清理旧产物：

```bat
cd /d D:\code\ABot-Recon\abot_recon\modeling\pi3\models\curope
rmdir /s /q build
del /q curope*.pyd
```

针对 RTX 4060 Ti 编译：

```bat
set TORCH_CUDA_ARCH_LIST=8.9
set DISTUTILS_USE_SDK=1
python setup.py build_ext --inplace
```

如果希望生成的扩展同时支持 GTX 1660 SUPER 和 RTX 4060 Ti，可以改用：

```bat
set TORCH_CUDA_ARCH_LIST=7.5+8.9
```

只使用 RTX 4060 Ti 推理时，`8.9` 编译产物更小、编译更快。

## 5. 验证扩展

回到项目根目录：

```bat
cd /d D:\code\ABot-Recon
```

确认生成了扩展文件：

```bat
dir abot_recon\modeling\pi3\models\curope\curope*.pyd
```

确认扩展可导入：

```bat
python -c "from abot_recon.modeling.pi3.models.curope import curope; print(curope.__file__)"
```

输出应指向 `.pyd` 文件，例如：

```text
D:\code\ABot-Recon\abot_recon\modeling\pi3\models\curope\curope.cp310-win_amd64.pyd
```

运行 parity 测试：

```bat
set ABOT_RECON_REQUIRE_CUROPE=1
python -m pytest -q tests\test_curope_parity.py
```

测试通过后，可以强制后续推理使用 CUDA 版 RoPE2D：

```bat
set ABOT_RECON_ROPE2D_BACKEND=cuda
```

设置为 `cuda` 后，如果扩展加载失败，程序会直接报错，而不是静默回退到慢速 PyTorch 实现。

## 6. 重新运行推理

注意：正在运行的 Python 进程不会热加载新编译的 `.pyd`。必须结束当前推理进程并重新启动 `demo.py`。

```bat
cd /d D:\code\ABot-Recon
set ABOT_RECON_ROPE2D_BACKEND=cuda

python demo.py ^
  --video E:/DJI_0001_min.mp4 ^
  --checkpoint checkpoints/abot_recon.safetensors ^
  --output-dir outputs/demo ^
  --attention-backend auto ^
  --device cuda ^
  --no-loop-closure
```

启动后不应再出现：

```text
Warning, cannot find cuda-compiled version of RoPE2D
```

## 常见问题排查

| 现象 | 原因与处理 |
|---|---|
| `fatal error C1189: unsupported Microsoft Visual Studio version` | 当前 `cl` 是 `_MSC_VER >= 1940`，常见于默认选择 MSVC 14.44。安装 MSVC 14.38，并使用 `-vcvars_ver=14.38` 重新打开 VS 编译环境。 |
| `cl` 版本仍是 `19.44` | 没有在带 `-vcvars_ver=14.38` 的 `VsDevCmd.bat` 环境中编译。关闭当前终端，按本文命令重新打开。 |
| `nvcc --version` 显示 13.0 | CUDA 13.0 在 `PATH` 中优先。设置 `CUDA_PATH` 到 CUDA 12.1，并把 `%CUDA_PATH%\bin` 放到 `PATH` 前面。 |
| PyTorch 显示 `2.5.1+cpu` | 当前环境不是 CUDA 版 PyTorch。先安装 `torch==2.5.1+cu121` 与匹配的 torchvision，再编译扩展。 |
| 编译成功但仍出现慢速 RoPE 警告 | 新扩展不会加载到已经启动的 Python 进程；确认 `.pyd` 导入路径正确，然后重启推理进程。 |
| 想绕过 `unsupported compiler` | 不建议使用 `-allow-unsupported-compiler`。CUDA 12.1 未支持该 MSVC 版本，可能出现编译失败、运行期错误或数值问题。 |
| 想改用 CUDA 13.0 编译 | 不建议。当前 PyTorch 是 `cu121`，扩展应与 PyTorch 的 CUDA 版本保持一致。 |

## 性能说明

编译 cuRoPE2D 只会加速旋转位置编码部分。Windows 版 PyTorch 2.5.1 可能仍没有 FlashAttention backend，attention 会使用 SDPA 回退路径。因此启用 cuRoPE2D 后速度会提升，但不一定达到 README 中 Linux/H100 环境的 24 FPS。
