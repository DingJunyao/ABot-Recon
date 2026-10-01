# 重建、伪高斯与网页查看工具使用说明

本文档说明 ABot-Recon 当前用于点云导出、Open3D 交互查看、伪 Gaussian
Splat 转换、大文件分块和网页查看的脚本与工作流。

## 工具索引

| 工具 | 用途 |
|---|---|
| `scripts/export_reconstruction_ply.py` | 将 `local_points.pt`、`world_points.pt` 或 `colors.pt` 导出为 RGB PLY |
| `scripts/view_reconstruction.py` | 使用 Open3D 查看 PLY，并沿相机轨迹播放 |
| `scripts/export_point_cloud_chunks.py` | 将 RGB 点云转换为网页使用的硬点云分块与清单 |
| `scripts/export_pseudo_gaussian_ply.py` | 将 RGB 点云转换为标准 3DGS 格式的伪 Gaussian PLY |
| `scripts/export_pseudo_gaussian_chunks.py` | 将 3DGS PLY 或 RGB 点云转换为网页使用的紧凑分块 |
| `scripts/serve_pseudo_gaussian_viewer.py` | 启动项目内网页查看器的本地 HTTP 服务（同时服务硬点云与伪高斯页面） |
| `tools/point_cloud_viewer.html` | 官方风格渐进式硬点云网页查看器 |
| `tools/point_cloud_datasets.json` | 硬点云网页查看器的数据列表 |
| `tools/pseudo_gaussian_viewer.html` | 项目内伪高斯网页查看器 |
| `tools/pseudo_gaussian_datasets.json` | 伪高斯网页查看器的数据列表 |

## 推荐工作流

### 1. Open3D 检查与行车播放

先使用 `view_reconstruction.py` 检查原始 RGB 点云：

```powershell
python scripts\view_reconstruction.py `
  --ply outputs\demo\reconstruction.ply `
  --poses outputs\demo\camera_poses.npy `
  --mode orbit `
  --voxel 0.15
```

行车播放：

```powershell
python scripts\view_reconstruction.py `
  --ply outputs\demo\reconstruction.ply `
  --poses outputs\demo\camera_poses.npy `
  --mode drive `
  --start-frame 0 `
  --stop-frame 400 `
  --frame-step 2 `
  --playback-fps 12 `
  --camera-offset 0 -2 -10 `
  --zoom 2.5
```

### 2. 官方风格硬点云网页查看

对于 `demo_loop` 这类大规模重建结果，优先使用硬点云网页查看器。它直接流式渲染原始 RGB 点，适合判断重建质量；伪高斯查看器只用于合成表面预览。

先转换为硬点云分块：

```powershell
.\.venv\Scripts\python.exe scripts\export_point_cloud_chunks.py `
  --input outputs\demo_loop\reconstruction.ply `
  --poses outputs\demo_loop\camera_poses.npy `
  --output-dir outputs\demo_loop\point_cloud_chunks `
  --chunk-size 250000 `
  --max-points 0
```

然后启动仓库根目录的本地服务：

```powershell
.\.venv\Scripts\python.exe scripts\serve_pseudo_gaussian_viewer.py --no-browser
```

打开：

```text
http://127.0.0.1:8765/tools/point_cloud_viewer.html
```

操作方式与 Open3D 一致：左键拖动旋转，右键或 Shift+左键平移，滚轮缩放。请在页面中选择 `demo_loop · full RGB`，然后点击 `Load selected dataset` 开始加载；有轨迹时可用 Drive 模式沿相机路径播放、拖动进度条跳转，Reset 恢复自动取景。

编码与性能权衡：

- 硬点云直接展示原始重建点，适合判断重建质量；
- 伪高斯查看器只渲染合成的表面预览，不代表原始重建点；
- 默认 `uint16` 编码每条记录 10 字节，体积更小，通常视觉上与原坐标一致；
- `--position-encoding float32` 每条记录 16 字节，精确保留源 PLY 坐标；
- `--max-points` 可生成更轻量的目录，适合快速预览；
- 不传 `--poses` 时没有轨迹，仅支持轨道查看。

### 3. 生成标准伪 Gaussian PLY

```powershell
python scripts\export_pseudo_gaussian_ply.py `
  --input outputs\demo\reconstruction.ply `
  --output outputs\demo\pseudo_gaussian.ply `
  --voxel 0 `
  --max-points 4000000 `
  --radius 0.12 `
  --opacity 0.85
```

### 4. 生成伪高斯网页分块

从标准 3DGS PLY 生成：

```powershell
python scripts\export_pseudo_gaussian_chunks.py `
  --input outputs\demo\pseudo_gaussian.ply `
  --output-dir outputs\demo\pseudo_gaussian_chunks `
  --chunk-size 250000
```

从原始 RGB 点云直接生成，绕过 SH `f_dc` 编解码：

```powershell
python scripts\export_pseudo_gaussian_chunks.py `
  --input outputs\demo_loop\reconstruction.ply `
  --output-dir outputs\demo_loop\pseudo_gaussian_rgb_chunks `
  --chunk-size 250000 `
  --radius 0.01 `
  --opacity 0.9 `
  --flip-y
```

### 5. 启动伪高斯网页查看器

```powershell
python scripts\serve_pseudo_gaussian_viewer.py
```

浏览器默认打开：

```text
http://127.0.0.1:8765/tools/pseudo_gaussian_viewer.html
```

不自动打开浏览器：

```powershell
python scripts\serve_pseudo_gaussian_viewer.py --no-browser
```

端口冲突时：

```powershell
python scripts\serve_pseudo_gaussian_viewer.py --port 8766
```

## `export_reconstruction_ply.py`

将 ABot-Recon 的稠密输出转换为 RGB PLY。

### 基本示例

```powershell
python scripts\export_reconstruction_ply.py `
  --poses outputs\demo\camera_poses.npy `
  --points outputs\demo\local_points.pt `
  --colors outputs\demo\colors.pt `
  --metadata outputs\demo\metadata.json `
  --output outputs\demo\reconstruction.ply `
  --bev-output outputs\demo\trajectory_bev.png
```

### 关键参数

| 参数 | 说明 |
|---|---|
| `--poses` | `camera_poses*.npy`，形状为 `[N,4,4]` |
| `--points` | `local_points.pt`、`world_points.pt` 或 RGB PLY |
| `--colors` | 与点图对齐的 `colors.pt` |
| `--points-frame` | `auto`、`local` 或 `world` |
| `--metadata` | 提供 `dense_output_indices` |
| `--confidence` | 可选的 `confidence.pt` |
| `--confidence-threshold` | 低于阈值的点会被过滤 |
| `--point-stride` | 每隔 N 个像素采样，默认 4 |
| `--frame-stride` | 每隔 N 个稠密帧采样 |
| `--max-points` | 最大输出点数 |
| `--bev-output` | 输出 XZ 平面轨迹图 |

`--points` 使用 `local_points.pt` 时必须同时提供 `--poses` 和
`--metadata`。如果使用已经变换到世界坐标系的 `world_points.pt`，不需要再次
传 `--points-frame local`。

## `view_reconstruction.py`

使用 Open3D 查看点云，并在行车模式下沿 `camera_poses.npy` 播放。

### 交互查看

```powershell
python scripts\view_reconstruction.py `
  --ply outputs\demo\reconstruction.ply `
  --poses outputs\demo\camera_poses.npy `
  --mode orbit `
  --voxel 0.15
```

常用操作：

| 操作 | 效果 |
|---|---|
| 左键拖动 | 旋转 |
| 滚轮 | 缩放 |
| 右键拖动 | 平移 |

### 行车播放

```powershell
python scripts\view_reconstruction.py `
  --ply outputs\demo\reconstruction.ply `
  --poses outputs\demo\camera_poses.npy `
  --mode drive `
  --voxel 0.15 `
  --start-frame 0 `
  --stop-frame 400 `
  --frame-step 2 `
  --playback-fps 12 `
  --camera-offset 0 -2 -10 `
  --zoom 2.5
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mode` | `orbit` | `orbit` 为自由查看，`drive` 为轨迹播放 |
| `--voxel` | `0` | 体素降采样尺寸，单位为米 |
| `--max-points` | `0` | 点数预算，`0` 表示不限制 |
| `--point-size` | `1.2` | 点大小 |
| `--line-width` | `3` | 轨迹线宽 |
| `--flip-y` | 开启 | 将相机式坐标转换为 Open3D Y-up |
| `--start-frame` | `0` | 播放起始帧 |
| `--stop-frame` | 文件末尾 | 播放结束帧，不包含该帧 |
| `--frame-step` | `5` | 每次播放跳过多少帧 |
| `--playback-fps` | `24` | 播放帧率 |
| `--camera-offset` | `0 -1.5 -8` | 相机右/下/前方向偏移 |
| `--zoom` | `1` | 焦距倍数，`2` 约为原来的两倍 |
| `--camera-fov` | `70` | 垂直视场角 |
| `--invert-view` | 关闭 | 反转相机前后方向 |
| `--background` | `dark` | `dark` 或 `light` |

`--camera-offset` 的三个值分别表示：

```text
RIGHT DOWN FORWARD
```

想看第一人称：

```powershell
--camera-offset 0 0 0
```

想看更高的 chase camera：

```powershell
--camera-offset 0 -5 -15
```

## `export_pseudo_gaussian_ply.py`

将 RGB 点云转换为标准 3DGS PLY。每个点变成一个各向同性 Gaussian。

### 示例

```powershell
python scripts\export_pseudo_gaussian_ply.py `
  --input outputs\demo_loop\reconstruction.ply `
  --output outputs\demo_loop\pseudo_gaussian.ply `
  --voxel 0 `
  --max-points 4000000 `
  --radius 0.12 `
  --opacity 0.85
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--input` | 必填 | RGB 点云 PLY |
| `--output` | 必填 | 标准 3DGS PLY |
| `--voxel` | `0.2` | 先进行体素降采样 |
| `--max-points` | `1500000` | 最大 Gaussian 数量，`0` 表示不限制 |
| `--radius` | `0.18` | 各向同性 Gaussian 半径 |
| `--opacity` | `0.85` | 透明度，取值必须在 0 和 1 之间 |
| `--flip-y` | 开启 | 转换为 Y-up 坐标 |

质量建议：

- 追求清晰度时使用 `--voxel 0`
- 点太少时会显得稀疏
- `radius` 太大会显得糊
- `opacity` 太低会导致前后颜色互相渗透

## `export_pseudo_gaussian_chunks.py`

将点云转换为网页查看器使用的紧凑分块格式。每个 Gaussian 固定使用 24 字节：

```text
position x/y/z  12 bytes
RGB              3 bytes
padding          1 byte
radius           4 bytes
opacity          4 bytes
```

相比原始 3DGS PLY 的 104 字节记录，适合浏览器按 chunk 加载。

### 从标准 3DGS PLY 生成

```powershell
python scripts\export_pseudo_gaussian_chunks.py `
  --input outputs\demo_loop\pseudo_gaussian.ply `
  --output-dir outputs\demo_loop\pseudo_gaussian_chunks `
  --chunk-size 250000
```

### 从原始 RGB 点云生成

该模式直接保留 PLY 中的 RGB 字节，不经过 `f_dc` 编解码：

```powershell
python scripts\export_pseudo_gaussian_chunks.py `
  --input outputs\demo_loop\reconstruction.ply `
  --output-dir outputs\demo_loop\pseudo_gaussian_rgb_chunks `
  --chunk-size 250000 `
  --radius 0.01 `
  --opacity 0.9 `
  --flip-y
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--input` | 必填 | 3DGS PLY 或 RGB 点云 PLY |
| `--output-dir` | 必填 | 输出 chunk 和 `manifest.json` |
| `--chunk-size` | `250000` | 每个 chunk 的 Gaussian 数量 |
| `--radius` | 无 | 覆盖半径，RGB 点云模式通常需要 |
| `--opacity` | 无 | 覆盖透明度，RGB 点云模式通常需要 |
| `--flip-y` | 关闭 | 将相机式 Y-down/Z-forward 转换为 Y-up |

输出目录：

```text
outputs/demo_loop/pseudo_gaussian_rgb_chunks/
├── manifest.json
├── chunk-00000.pbin
├── chunk-00001.pbin
└── ...
```

## `serve_pseudo_gaussian_viewer.py`

```powershell
python scripts\serve_pseudo_gaussian_viewer.py
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8765` | 服务端口 |
| `--no-browser` | 关闭 | 不自动打开浏览器 |

## 项目内网页查看器

主页面：

```text
tools/pseudo_gaussian_viewer.html
```

### 点云选择

数据列表：

```text
tools/pseudo_gaussian_datasets.json
```

格式：

```json
[
  {
    "name": "显示名称",
    "manifest": "/outputs/path/manifest.json",
    "poses": "/outputs/path/camera_poses.npy"
  }
]
```

也可以在页面的 `Manifest` 和 `Pose NPY` 输入框中填写其他路径。

### 交互查看

- 左键拖动：旋转
- 滚轮：缩放
- 右键拖动：平移

### 行车播放

- `行车播放`：进入轨迹播放
- `暂停 / 继续`：暂停或继续
- 空格键：暂停或继续
- `播放进度`：点击或拖动跳转到任意帧
- `播放速度`：调整播放速率

相机控制：

| 控件 | 说明 |
|---|---|
| 相机高度 | 沿世界竖直方向移动 |
| 相机后退 | 调整 chase camera 距离 |
| 水平视角 | 左右转头 |
| 俯仰视角 | 抬头或低头 |
| 画面缩放 | 调整焦距 |

渲染控制：

| 控件 | 说明 |
|---|---|
| 天空背景 | 在缺失天空几何时显示蓝色渐变背景 |
| 透明混合 | 开启时按距离排序并做 alpha 混合 |
| 整体透明 | 调整 Gaussian 混合强度 |
| 泼溅大小 | 调整屏幕上的 Gaussian 尺寸 |

## 当前推荐数据

### 官方风格硬点云全量版本

```text
outputs/demo_loop/point_cloud_chunks
```

特征：

```text
25,596,210 points
约 256 MB compact data
uint16 坐标 + 原始 RGB
```

这是优先用于判断重建质量的网页版本，直接显示原始 RGB 点，不引入合成 Gaussian 表面。

### 伪高斯精确 RGB 全量版本

```text
outputs/demo_loop/pseudo_gaussian_rgb_chunks
```

特征：

```text
25,596,210 Gaussians
约 614 MB compact data
RGB 直接来自 reconstruction.ply
```

这是与 `view_reconstruction.py` 颜色最接近的网页版本。

### 标准全量伪高斯版本

```text
outputs/demo_loop/pseudo_gaussian_chunks
```

特征：

```text
23,664,527 Gaussians
约 568 MB compact data
RGB 来自 3DGS f_dc 解码
```

### 小规模演示版本

```text
outputs/demo/pseudo_gaussian_chunks
```

特征：

```text
1,180,405 Gaussians
约 28 MB compact data
```

## 常见问题

### 页面提示 `Failed to fetch`

通常是浏览器尝试一次性加载超过约 2 GB 的单个文件。不要直接 `fetch` 完整
PLY，应该先转换为 chunk：

```powershell
python scripts\export_pseudo_gaussian_chunks.py `
  --input outputs\demo_loop\pseudo_gaussian.ply `
  --output-dir outputs\demo_loop\pseudo_gaussian_chunks
```

### 天空或蓝色远处物体显示成土黄色

该问题由分块颜色的 RGB/BGR 字节序错误导致。当前版本已经修正。

如果使用旧分块，请重新生成：

```powershell
python scripts\export_pseudo_gaussian_chunks.py `
  --input outputs\demo_loop\reconstruction.ply `
  --output-dir outputs\demo_loop\pseudo_gaussian_rgb_chunks `
  --radius 0.01 `
  --opacity 0.9 `
  --flip-y
```

### 行车视角没有画面

常见原因是 `.npy` 位姿 dtype 解析错误或相机 forward 轴方向错误。查看器现在
会自动解析 `float32` 和 `float64` 位姿，并处理 ABot 的 `+Z` forward 与
WebGL 的 `-Z` forward 差异。

### 进度条无法调整

进度条统一使用 `0～1` 内部比例：

```text
0.0 = 第 1 帧
0.5 = 中间帧
1.0 = 最后一帧
```

当前版本支持点击和拖动跳转。

### 天空几何很少

单目重建对天空的置信度通常很低。模型可能只给出相机附近的不稳定点，而不是
远处天空穹顶。网页查看器提供可选的天空背景来弥补展示效果；真正的照片级天空
需要训练 3D Gaussian Splatting 或 NeRF。

## 验证

相关测试：

```powershell
python -m pytest tests\test_view_reconstruction.py tests\test_pseudo_gaussian_ply.py tests\test_pseudo_gaussian_chunks.py tests\test_export_point_cloud_chunks.py tests\test_point_cloud_viewer_assets.py tests\test_serve_pseudo_gaussian_viewer.py -q
```

代码检查：

```powershell
ruff check scripts tests
```
