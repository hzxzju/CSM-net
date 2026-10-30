# FOV OCR：Tesseract baseline

当前 OCR 模型的 baseline 为 **Tesseract**，代码入口是 `TesseractFOVBaseline`，
模型标识为 `tesseract_baseline`。OpenCV 负责图像预处理，Tesseract 负责文字识别，
再通过规则解析 FOV 数值和单位。

使用 OpenCV 放大、灰度化、二值化及白色文字筛选，再通过 Tesseract 读取文本并提取
`FOV: 4.00um` 中的数值及单位。无需训练，全部在本地运行。

默认输入是**手动裁剪好的右上角区域**，不会再次自动裁剪。
支持路径（包括中文路径）和 OpenCV `uint8` 灰度/BGR/BGRA 数组。

## Baseline 默认配置

- 使用 Tesseract 的预训练英文语言数据 `eng`，不进行微调。
- 输入区域放大 3 倍，使用三次插值；边缘补白 12 像素。
- 预处理包含灰度、Otsu 二值化、亮度阈值 200；彩色输入额外按 BGR 各通道
  均不小于 190 筛选白色文字。
- 每种预处理分别使用 `PSM 6`、`PSM 11`，按既有规则检查结果冲突。
- 每次 Tesseract 调用的超时为 10 秒。

后续比较其他 OCR 模型时，应使用相同的裁剪区域和人工标注，记录 OCR 引擎版本、
语言数据版本及任何参数变更。现有样例用于功能验证，尚无全数据集 baseline 准确率。

## 安装

```bash
conda activate hzx
python -m pip install -e '.[ocr]'
```

Tesseract 是额外的系统程序，`pip install pytesseract` 不会安装它。
Ubuntu 可安装：

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

也可安装到 Conda 环境中：

```bash
conda install -n hzx -c conda-forge tesseract pytesseract
```

如果环境已有 `opencv-python` 或 `opencv-contrib-python`，请继续使用已有的
OpenCV，只安装 `pytesseract` 并使用 `pip install -e . --no-deps`，避免同时安装
多个提供 `cv2` 的包。

## Python

```python
from ocr_model import TesseractFOVBaseline

model = TesseractFOVBaseline()
result = model.predict("右上角裁剪.png")
if result is not None:
    print(result.value)     # 4.0
    print(result.unit)      # "um"，µm/μm 会统一为 um
    print(result.value_um)  # 4.0，换算为微米；未识别单位时为 None
    print(result.raw_text)  # 原始 OCR 文本，便于复核
```

原有 `FOVRecognizer` 名称保留为 `TesseractFOVBaseline` 的别名，已有调用可继续使用。

对完整图像，可显式传入像素坐标 `(x, y, width, height)`：

```python
result = model.predict("image/hole0.jpg", roi=(285, 0, 195, 80))
```

该坐标适用于本地查看的 480×480 样例，不应直接套用到不同布局或尺寸的图片。

## 命令行

在项目根目录运行：

```bash
PYTHONPATH=src python -m ocr_model 右上角裁剪.png
PYTHONPATH=src python -m ocr_model image/hole0.jpg --roi 285 0 195 80
```

输出 JSON，包含 `model: "tesseract_baseline"`；退出码 0 表示成功，
1 表示未读到明确 FOV，2 表示输入、依赖或执行错误。

## 结果约定

- 必须读到 `FOV` 标签（兼容 OCR 误读 `F0V`），不会把 Location、Rotation 等数字当作 FOV。
- 支持正整数、小数、千分位及 nm、um、µm、μm、mm、m；小数使用英文句点。
- 没读到单位时 `unit` 和 `value_um` 为 `None`，不默认假设为微米。
- 灰度输入使用三种预处理，彩色输入额外筛选白色文字；各用两种文本布局识别，
  成功结果的数值或单位发生冲突时返回 `None`。
- 结果不是准确率保证，尤其需复核小数点；真实批量准确率需用人工标注样本评估。
- 依赖缺失、无效图像、Tesseract 执行失败或超时会抛出异常。默认每次 OCR 超时 10 秒。

## 验证

`tests/test_ocr_model.py` 的 26 项测试已通过，包含真实 Tesseract 识别、中文路径、
输入校验和三个本地 SEM 样例：`hole0.jpg` → `4.00 um`，
`hole100.jpg`、`hole104.jpg` → `1.00 um`。这些样例不代表全数据集准确率。

本次测试使用 OpenCV 4.11.0、pytesseract 0.3.13 和 Tesseract 4.1.1。
`pytesseract` 已安装到 `hzx`；Tesseract 仅解压在 `/tmp/sem-fov-tesseract` 中，
长期使用请按上面的安装说明配置。临时目录尚在时，可从项目根目录复现：

```bash
PATH=/tmp/sem-fov-tesseract/usr/bin:$PATH \
LD_LIBRARY_PATH=/tmp/sem-fov-tesseract/usr/lib/x86_64-linux-gnu \
TESSDATA_PREFIX=/tmp/sem-fov-tesseract/usr/share/tesseract-ocr/4.00/tessdata \
PYTHONPATH=src conda run -n hzx python -m ocr_model image/hole100.jpg --roi 285 0 195 80
```
