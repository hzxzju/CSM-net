# DINOv3 + LazyStrike SEM classification

For FOV recognition from a cropped SEM top-right annotation region, see
[`src/ocr_model/README.md`](src/ocr_model/README.md). This optional module uses
OpenCV preprocessing and Tesseract OCR as the current OCR baseline
(`TesseractFOVBaseline`).

This project implements image-level classification for the 1,475 local SEM
images. Labels are inferred from filenames and cover seven defect classes:
`hole`, `infilm`, `metal`, `partical`, `pattern_deform`, `puddle`, and
`scratch`.

The downstream task is seven-class image classification. The model pipeline is:

1. DINOv3 ViT-S/16 produces the final patch feature map `F_last`.
2. LazyStrike applies the official shifted-FFT Gaussian filter, repository
   stability scoring, channel-wise Top-K selection, and patch voting.
3. The selected pseudo foreground mask generates a learnable spatial gate
   `G_pos`.
4. Residual positional attention forms
   `F_cls = F_last * (1 + G_pos)`.
5. Global average pooling, LayerNorm, and a linear head produce seven image-level
   class logits.

The pseudo mask is an internal attention prior and an interpretable diagnostic;
it is not treated as a pixel-level training target.

## Install and test

```bash
conda activate hzx
python -m pip install -r requirements.txt
python -m pip install -e '.[test]' --no-deps
pytest -q
```

The pretrained DINOv3 checkpoint is downloaded from Hugging Face on first use.
Use `local_files_only=True` when the model is already cached on an offline
machine.

## Minimal classification use

```python
import torch

from lazystrike_cls import DinoV3LazyStrikeClassifier, SEMClassificationDataset

dataset = SEMClassificationDataset("image")
sample = dataset[0]

model = DinoV3LazyStrikeClassifier(
    model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    num_classes=len(dataset.class_to_idx),
    topk_ratio=0.10,
    gaussian_sigma=None,  # official default: sqrt(feature channels) in FFT bins
    mask_source="vote",  # use "patch_score" for the score-threshold pseudo mask
    freeze_encoder=True,
).eval()

with torch.inference_mode():
    output = model(sample["pixel_values"].unsqueeze(0))

print(output.logits.shape)                   # [1, 7]
print(output.probabilities.shape)            # [1, 7]
print(output.prediction.shape)               # [1]
print(output.positional_gate.shape)          # [1, 1, 30, 30]
print(output.pseudo_foreground_mask.shape)   # [1, 480, 480], diagnostic only
```

The Gaussian kernel follows the official visualization code: it is constructed
in shifted FFT-bin coordinates and defaults to `sigma = sqrt(C)`, where `C` is
the feature dimension. The stability score is `patch / abs(low_pass - patch)`.

`patch_score` is the cosine similarity between every final-layer patch token and
the encoder CLS token, matching the repository's `visualization/patch_score.py`.
`vote_foreground_patch_mask` thresholds the LazyStrike vote count, while
`patch_score_foreground_patch_mask` thresholds this CLS-patch score at its
per-image mean plus one population standard deviation.
`selected_foreground_patch_mask` is chosen by `mask_source` and used to build
the positional gate.

At initialization, pseudo-foreground and pseudo-background gate values are 0.75
and 0.25. `foreground_attention_prior` changes this prior. The refinement head
then learns task-specific spatial corrections from image-level classification
loss.

## Classification training target

The dataset already returns an integer `class_label`; no mask directory is
required:

```python
output = model(batch["pixel_values"])
loss = torch.nn.functional.cross_entropy(output.logits, batch["class_label"])
```

With `freeze_encoder=True`, classification loss trains the positional-attention
refinement and classification head while keeping DINOv3 frozen. Set
`freeze_encoder=False` for end-to-end fine-tuning.

## Single-image diagnostic

After training, pass the saved state dictionary with `--checkpoint`:

```bash
python -m lazystrike_cls.demo image/hole0.jpg \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs/hole0
```

The command prints seven class probabilities and saves the selected pseudo mask
and positional gate. Without a checkpoint, the classification head is random and
its prediction is not meaningful.
