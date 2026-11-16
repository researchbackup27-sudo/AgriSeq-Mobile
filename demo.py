"""Demonstration of the AgriSeq-Mobile evaluation methods on synthetic data.

Runs, on a synthetic imbalanced 5-class test set and synthetic runs with five seeds:
  [1/6] per-class recall and balanced accuracy,
  [2/6] the learning-rate selection rule (lowest mean validation loss, ties to the smaller rate,
        grid-edge flag),
  [3/6] paired comparison of two configurations: bootstrap over test images, seed-level paired t-test,
        Welch's unpaired t-test, exact sign-flip test, per-seed McNemar tests and Holm adjustment,
  [4/6] the smallest difference five seeds detect with 80% power,
  [5/6] the operating-system file cache: residency of a file before and after eviction,
  [6/6] implementation checks against independent references.

It needs none of the paper's datasets, no TensorFlow and no network, downloads no weights, and runs in a
few seconds on a laptop CPU. Every printed number is computed from the synthetic runs with the functions in
agriseq_mobile.py and NumPy (the paired bootstrap interval is assembled from resample_weights).
The training functions (train_run, tune_learning_rate) need TensorFlow and the image datasets; see README.
"""
import itertools
import os
import sys
import tempfile

import numpy as np
from scipy import stats

import agriseq_mobile as am

rng = np.random.default_rng(0)
CLASSES = ["Cercospora", "Healthy", "Insect", "Leaf Crinkle", "Yellow Mosaic"]
COUNTS = [60, 54, 41, 81, 168]             # test-set class sizes of IDBGL (imbalanced, 4.1 to 1)
K, SEEDS = len(CLASSES), am.SEEDS
y_test = np.repeat(np.arange(K), COUNTS)
n_test = len(y_test)


def synthetic_run(recall_by_class, seed):
    """Predictions of one synthetic run: each image of class c is correct with probability recall[c]."""
    r = np.random.default_rng(seed)
    pred = y_test.copy()
    wrong = r.random(n_test) > np.asarray(recall_by_class)[y_test]
    pred[wrong] = (y_test[wrong] + r.integers(1, K, wrong.sum())) % K
    return pred


# Configuration B has a weaker rare class (Insect) than configuration A.
REC_A = [0.93, 0.99, 0.90, 0.94, 0.96]
REC_B = [0.92, 0.99, 0.84, 0.93, 0.96]
runs = {"A": [synthetic_run(REC_A, 100 + s) for s in SEEDS], "B": [synthetic_run(REC_B, 200 + s) for s in SEEDS]}

print("[1/6] Per-class recall and balanced accuracy (five-seed means)")
for name, preds in runs.items():
    cms = [am.cm_of(y_test, p, K) for p in preds]
    rec = np.mean([am.per_class_recall(cm) for cm in cms], 0) * 100
    m = am.metrics_from_cm(np.stack(cms))
    print(f"  {name}: accuracy {100 * m['accuracy'].mean():.2f}%  balanced accuracy "
          f"{100 * m['macro_recall'].mean():.2f}%  recall " + ", ".join(f"{c} {v:.1f}" for c, v in zip(CLASSES, rec)))

print("[2/6] Learning-rate selection on the validation set (two tuning seeds per rate)")
grid = am.LR_GRID["adam"]
true_curve = {r: 0.20 + 0.05 * (np.log10(r) + 2.5) ** 2 for r in grid}       # minimum near 3e-3
table = {r: [true_curve[r] + rng.normal(0, 0.005) for _ in am.TUNE_SEEDS] for r in grid}
rate, edge = am.select_learning_rate(table)
print("  mean lowest validation loss: " + ", ".join(f"{r:g}: {np.mean(v):.4f}" for r, v in table.items()))
print(f"  selected rate {rate:g}, at the edge of the grid: {edge}")
edge_table = {r: [1.0 / (1 + 100 * r)] * 2 for r in grid}                     # keeps falling with the rate
print(f"  a loss that keeps falling with the rate selects {am.select_learning_rate(edge_table)}")
tie_table = {r: [0.30, 0.30] for r in grid}
print(f"  equal losses for every rate select the smallest: {am.select_learning_rate(tie_table)}")

print("[3/6] Paired comparison of A and B (same test images, paired by seed)")
ok_a = np.stack([p == y_test for p in runs["A"]])
ok_b = np.stack([p == y_test for p in runs["B"]])
acc_a, acc_b = ok_a.mean(1) * 100, ok_b.mean(1) * 100
d = acc_a - acc_b
W = am.resample_weights(np.random.default_rng(20261116), n_test, 10_000)
boot = ((W @ ok_a.T.astype(np.float32) / n_test).mean(1) - (W @ ok_b.T.astype(np.float32) / n_test).mean(1)) * 100
p_boot = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
p_t, p_w, p_sf = am.seed_level_t(d), am.welch_t(acc_a, acc_b), am.signflip_exact(d)
mc = [am.mcnemar_exact(a, b) for a, b in zip(ok_a, ok_b)]
print(f"  difference {d.mean():.2f} points, 95% interval [{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}], "
      f"bootstrap p = {p_boot:.4f}")
print(f"  seed-level paired t p = {p_t:.4f}, Welch p = {p_w:.4f}, exact sign-flip p = {p_sf:.4f} "
      f"(smallest possible with 5 seeds: {2 / 2 ** len(SEEDS):.4f})")
print("  per-seed McNemar (b, c, p): " + "; ".join(f"({b}, {c}, {p:.3f})" for b, c, p in mc))
print(f"  Holm-adjusted [bootstrap, paired t, sign-flip]: {np.round(am.holm([p_boot, p_t, p_sf]), 4).tolist()}")

print("[4/6] Smallest difference detected with 80% power by the seed-level paired t-test (5 seeds)")
sd = float(np.std(d, ddof=1))
print(f"  sd of per-seed differences {sd:.3f} points -> {am.mde_paired_t(sd):.2f} points")

print("[5/6] Operating-system file cache")
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model.bin")
    with open(path, "wb") as f:
        f.write(np.random.default_rng(1).bytes(8 << 20))
    with open(path, "rb") as f:
        f.read()
    print(f"  after reading the file: {100 * am.file_cache_residency(path):.0f}% of its pages are cached")
    if hasattr(os, "posix_fadvise"):
        am.evict_from_file_cache(path)
        print(f"  after evict_from_file_cache: {100 * am.file_cache_residency(path):.0f}% cached")
    else:
        print("  posix_fadvise is not available on this operating system; eviction needs Linux")

print("[6/6] Implementation checks")
checks = []
ref_sf = stats.permutation_test((d,), np.mean, permutation_type="samples", n_resamples=np.inf,
                                alternative="two-sided").pvalue
checks.append(("exact sign-flip vs scipy.stats.permutation_test", np.isclose(p_sf, ref_sf)))
ps = rng.random(7)
naive = np.empty(7)
for i in range(7):                       # Holm by its definition: max over smaller p of (m - rank) * p
    order = np.argsort(ps)
    rank = int(np.where(order == i)[0][0])
    naive[i] = min(1.0, max((7 - r) * ps[order[r]] for r in range(rank + 1)))
checks.append(("Holm vs its definition", np.allclose(am.holm(ps), naive)))
checks.append(("exact McNemar vs scipy.stats.binomtest",
               all(np.isclose(p, 1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue) for b, c, p in mc)))
cm = am.cm_of(y_test, runs["B"][0], K)
loop_rec = [sum(1 for t, p in zip(y_test, runs["B"][0]) if t == p == c) / COUNTS[c] for c in range(K)]
checks.append(("per-class recall vs a loop over images", np.allclose(am.per_class_recall(cm), loop_rec)))
checks.append(("balanced accuracy vs mean of per-class recall",
               np.isclose(am.metrics_from_cm(cm)["macro_recall"], np.mean(loop_rec))))
idx = np.random.default_rng(5).integers(0, n_test, size=(20, n_test))
Wsmall = np.stack([np.bincount(i, minlength=n_test) for i in idx]).astype(np.float32)
direct = [(runs["A"][0][i] == y_test[i]).mean() for i in idx]
checks.append(("weighted bootstrap accuracy vs direct resampling",
               np.allclose(am.boot_metrics(Wsmall, y_test, runs["A"][0], K)["accuracy"], direct)))
eff = am.mde_paired_t(1.0)
sim = np.random.default_rng(7).normal(eff, 1.0, size=(200_000, len(SEEDS)))
tstat = sim.mean(1) / (sim.std(1, ddof=1) / np.sqrt(len(SEEDS)))
power = np.mean(np.abs(tstat) > stats.t.ppf(0.975, len(SEEDS) - 1))
checks.append((f"power at the detectable difference by simulation ({power:.3f}) is 0.80 +- 0.01 (tolerance)", abs(power - 0.8) < 0.01))
checks.append(("seed-level paired t vs scipy.stats.ttest_rel", np.isclose(p_t, stats.ttest_rel(acc_a, acc_b).pvalue)))
checks.append(("tie in the selection rule goes to the smaller rate", am.select_learning_rate(tie_table)[0] == min(grid)))
checks.append(("Welch vs its formula",
               np.isclose(p_w, 2 * stats.t.sf(abs(d.mean()) / np.sqrt(acc_a.var(ddof=1) / 5 + acc_b.var(ddof=1) / 5),
                                              (acc_a.var(ddof=1) / 5 + acc_b.var(ddof=1) / 5) ** 2 /
                                              ((acc_a.var(ddof=1) / 5) ** 2 / 4 + (acc_b.var(ddof=1) / 5) ** 2 / 4)))))
for name, ok in checks:
    label = "AGREES" if "tolerance" in name else "IDENTICAL"
    print(f"  {name}: {label if ok else '*** DIFFERS ***'}")
sys.exit(0 if all(ok for _, ok in checks) else 1)
