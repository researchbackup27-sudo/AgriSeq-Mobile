# AgriSeq-Mobile: reference implementation

Reference code for:

> **AgriSeq-Mobile: cold-start-aware detection of black gram and rubber leaf diseases.**

The study trains one Dense classification layer on a frozen ImageNet backbone (NASNetMobile,
MobileNetV2 or EfficientNet-B0). It compares six optimizer schedules under two training protocols
over five seeds, exports the models to TensorFlow Lite and times each start-up stage on a CPU. It
ranks the backbones with an Edge-Efficiency Score (EES) that covers a fixed number of diagnoses.

## What is here

| File | Contents |
|---|---|
| `agriseq_mobile.py` | The methods of the paper as functions: the data split, the task model (frozen backbone with one Dense layer, Table 5), the six schedules and the training settings (Table 6), the Protocol 1 early-stopping rule and the Protocol 2 lowest-validation-loss epoch, export to Keras and TensorFlow Lite (float32 and INT8), staged start-up timing for Keras and TensorFlow Lite, the metrics, the paired bootstrap, the exact McNemar test, the Holm adjustment, the seed-level paired t-test and the Edge-Efficiency Score (Eq 8). |
| `reproduce.py` | Recomputes the paper's statistics from the deposited per-image predictions and timing observations. It checks every value against the deposited results and prints the headline numbers next to their published values. |

The paper timed each model in a new Python process; the timing functions measure the stages inside
the process that calls them.

## Data

This repository does **not** redistribute data.

- Images: IDBGL (black gram, Shoib et al., doi:10.17632/z55yrbmn2d.3), redistributed at
  https://www.kaggle.com/datasets/miitdaga/black-gram-disease, and BDRubberLeaf V2 (rubber,
  Deb Nath et al., doi:10.17632/4kjz78m7x5.2), redistributed at
  https://www.kaggle.com/datasets/miitdaga/rubber-leaf-disease-dataset. Both are under CC BY 4.0.
- Results: https://doi.org/10.34740/kaggle/ds/12896074
  (https://www.kaggle.com/datasets/miitdaga/agriseq-mobile-results). This deposit holds the split
  file lists, the settings, training histories and per-image predictions of all 200 runs, and the
  raw CPU timing observations. `reproduce.py` reads it.

The license below covers this code only.

## Usage

```bash
pip install -r requirements.txt
kaggle datasets download -d miitdaga/agriseq-mobile-results --unzip -p agriseq-mobile-results
python reproduce.py --results-root agriseq-mobile-results --out reproduced.json
```

`reproduce.py` needs only NumPy, pandas and SciPy. It runs on a CPU in a few seconds, is
deterministic, and makes no network calls. It exits with status 1 if any recomputed value differs
from the deposited one.

To train a model with the reference code, install TensorFlow and scikit-learn as well:

```python
import numpy as np
import agriseq_mobile as am

paths, classes, labels = am.dataset_listing("/path/to/black-gram-disease")
tr, va, te = am.split(paths, labels)                     # 70/20/10, random_state=1
order = tr + va + te
label_of = dict(zip(paths, labels))
X = am.load_images(order)                                # about 2.4 GB of RAM for IDBGL
y = np.array([label_of[p] for p in order])
splits = {"train": np.arange(len(tr)), "val": np.arange(len(tr), len(tr) + len(va)),
          "test": np.arange(len(tr) + len(va), len(order))}
preds, model = am.train_run("mobilenetv2", "adam_sgd", "orig", seed=1, X=X, y=y,
                            splits=splits, n_classes=len(classes))
files = am.export_models(model, "export/mobilenetv2")   # Keras, TFLite float32, TFLite INT8
x = X[splits["test"][:1]]
print(am.time_tflite_stages(files["fp32"], x, threads=4))
```

## What `reproduce.py` checks

1. The test accuracy, macro and weighted F1, precision and recall of every run, the accuracy on
   the copy-free test images (from the deposited duplicate pairs), and the epoch that Protocol 2
   selects, recomputed from the logits.
2. The bootstrap intervals of the five-seed means (10,000 resamples of the test images), the paired
   differences between schedules, between switching and restarting, and between backbones, with
   their bootstrap p-values, Holm-adjusted p-values, per-seed differences and exact McNemar tests
   for every seed.
3. The schedule chosen for each backbone by mean validation accuracy (a tie goes to the first
   schedule in alphabetical order, as in the paper), the EES at 1, 10 and 100 diagnoses with its
   bootstrap intervals and the share of resamples in which each backbone ranks first, and the
   sensitivity checks (a) to (e) with their rankings.
4. The median and interquartile range of the start-up and warm prediction times of all 30 timing
   settings.
5. The headline numbers below, at the precision printed in the paper.

It does not repeat the duplicate audit itself, which needs the images; it uses the deposited list
of duplicate pairs.

## Expected output

| Quantity | Published |
|---|---|
| IDBGL five-seed mean accuracy, range over the four schedules, EfficientNet-B0 | 95.45% to 96.24% |
| Same, NASNetMobile with early stopping | 91.73% to 93.32% |
| Same, MobileNetV2 | 94.11% to 95.25% |
| Paired seeds in which MobileNetV2 / EfficientNet-B0 beat NASNetMobile on IDBGL | 17 of 20 / 20 of 20 |
| Spread of the six schedules at the lowest-validation-loss epoch, IDBGL / BDRubberLeaf | within 0.6 / 0.4 points |
| Smallest Holm-adjusted seed-level p between IDBGL schedules | 0.249 |
| Keras start-up, NASNetMobile / MobileNetV2 (IDBGL) | 16.8 s / 3.7 s |
| TensorFlow Lite float32 start-up, 4 threads / 1 thread | 26 to 84 ms / 33 to 106 ms |
| Highest EES at one diagnosis, both datasets | MobileNetV2 |

## Citation

Citation details will be added when the paper is published.
