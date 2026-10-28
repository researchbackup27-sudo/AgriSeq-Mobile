"""Recompute the paper's statistics from the deposited per-image predictions and timing observations.

    python reproduce.py --results-root /path/to/agriseq-mobile-results --out reproduced.json

--results-root is the downloaded Kaggle dataset https://doi.org/10.34740/kaggle/ds/12896074
(per_run_predictions/, nb1/, nb2/, analysis/). No images, GPU or TensorFlow are needed.

Each step recomputes a quantity from the raw deposit and compares it with the value the paper
reports (the deposited stats and analysis files). The script stops with an error if any value
differs by more than a rounding tolerance. Headline numbers are printed next to the published ones.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from agriseq_mobile import (best_val_epoch, boot_metrics, cm_of, ees, holm, mcnemar_exact,
                            metrics_from_cm, resample_weights, seed_level_t)

DATASETS = ["idbgl", "rubber"]
ARCHS = ["nasnetmobile", "mobilenetv2", "efficientnetb0"]
MAIN = ["adam", "sgd", "adam_sgd", "sgd_adam"]
SIX = MAIN + ["adam_adam", "sgd_sgd"]
SEEDS = [1, 42, 123, 270, 789]
B = 10_000
CHECKS = []


def check(name, got, ref, tol=1e-6):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    ok = got.shape == ref.shape and bool(np.all(np.abs(got - ref) <= tol))
    CHECKS.append((name, ok))
    if not ok:
        print(f"  MISMATCH {name}: {got} vs {ref}")
    return ok


def load_runs(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "per_run_predictions", "*"))):
        r = json.load(open(os.path.join(d, "result.json")))
        z = np.load(os.path.join(d, "preds.npz"))
        runs[r["job"]["tag"]] = (r, {k: z[k] for k in z.files})
    return runs


def final_pred(P, rule):
    if rule == "best_val":
        e = best_val_epoch(P["ep_val_logits"], P["val_true"])
        return P["ep_test_logits"][e].argmax(1), e + 1
    return P["test_logits"].argmax(1), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", default="reproduced.json")
    A = ap.parse_args()
    root = A.results_root
    nb1, nb2, an = (os.path.join(root, s) for s in ("nb1", "nb2", "analysis"))
    out = {}

    # 1. Runs and per-run metrics -------------------------------------------------------------
    runs = load_runs(root)
    print(f"{len(runs)} training runs loaded")
    check("number of runs", len(runs), 200, 0)
    K = {ds: len(json.load(open(os.path.join(nb1, "data", "prep_summary.json")))[ds]["classes"]) for ds in DATASETS}
    # Copy-free test images: no exact (MD5) or near (thumbnail correlation >= 0.98) copy in the training
    # or validation set, read from the deposited duplicate pairs (the audit itself needs the images).
    leaked = {}
    for ds in DATASETS:
        dp = pd.read_csv(os.path.join(nb1, "data", f"{ds}_duplicate_pairs_exact_or_corr0.98.csv"))
        leaked[ds] = set(dp.i[(dp.split_i == "test") & (dp.split_j != "test")]) | set(dp.j[(dp.split_j == "test") & (dp.split_i != "test")])
    rows = []
    for tag, (r, P) in runs.items():
        j = r["job"]
        for rule in ["final"] + (["best_val"] if j["protocol"] == "fixed10" else []):
            pred, ep = final_pred(P, rule)
            m = metrics_from_cm(cm_of(P["test_true"], pred, K[j["dataset"]]))
            clean = ~np.isin(P["test_idx"], list(leaked[j["dataset"]]))
            rows.append(dict(**{k: j[k] for k in ("dataset", "arch", "protocol", "schedule", "seed")}, rule=rule,
                             selected_epoch=ep, n_clean_test=int(clean.sum()),
                             acc_clean_test=float((pred[clean] == P["test_true"][clean]).mean()) * 100, val_acc=float((P["val_logits"].argmax(1) == P["val_true"]).mean()) * 100,
                             **{k: float(v) * 100 for k, v in m.items()}))
    PR = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(nb1, "stats", "per_run_metrics.csv"))
    key = ["dataset", "arch", "protocol", "schedule", "seed", "rule"]
    mg = PR.merge(ref, on=key, suffixes=("", "_ref"))
    check("per-run rows matched", len(mg), len(ref), 0)
    for mname in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
                  "weighted_precision", "weighted_recall", "acc_clean_test"):
        check(f"per-run {mname}", mg[mname], mg[f"{mname}_ref"], 1e-6)
    check("copy-free test images", mg.n_clean_test, mg.n_clean_test_ref, 0)
    check("Protocol 2 selected epoch", mg.selected_epoch.fillna(0), mg.selected_epoch_ref.fillna(0), 0)
    print("1. per-run accuracy, F1, precision, recall, copy-free accuracy and Protocol 2 epoch selection: recomputed")

    # 2. Bootstrap intervals and paired comparisons (paired over test images, B = 10,000) ---------
    rng = np.random.default_rng(20261027)
    W_OF = {}
    for ds in DATASETS:
        yt = next(P["test_true"] for r, P in runs.values() if r["job"]["dataset"] == ds)
        W_OF[ds] = (resample_weights(rng, len(yt), B), yt)
    BOOT, ci_rows = {}, []
    for (ds, arch, prot, rule, s), g in PR.groupby(["dataset", "arch", "protocol", "rule", "schedule"]):
        W, yt = W_OF[ds]
        preds = {sd: final_pred(runs[f"{ds}__{arch}__{prot}__{s}__s{sd}"][1], rule)[0] for sd in sorted(g.seed)}
        per_seed = {sd: boot_metrics(W, yt, p, K[ds]) for sd, p in preds.items()}
        BOOT[(ds, arch, prot, rule, s)] = (per_seed, preds)
        for mname in ("accuracy", "macro_f1", "weighted_f1"):
            dist = np.mean([per_seed[sd][mname] for sd in preds], axis=0) * 100
            lo, hi = np.percentile(dist, [2.5, 97.5])
            ci_rows.append(dict(dataset=ds, arch=arch, protocol=prot, rule=rule, schedule=s, metric=mname, lo=lo, hi=hi))
    CI = pd.DataFrame(ci_rows)
    ref = pd.read_csv(os.path.join(nb1, "stats", "bootstrap_ci.csv"))
    ref = ref[ref.seed.isna()]
    mg = CI.merge(ref, on=["dataset", "arch", "protocol", "rule", "schedule", "metric"])
    check("bootstrap interval rows", len(mg), len(ref), 0)
    check("bootstrap interval bounds", mg[["lo", "hi"]].to_numpy(), mg[["boot_ci_low", "boot_ci_high"]].to_numpy(), 1e-6)

    def compare(family, a, b):
        (ds, *_), (pa_seed, pa), (pb_seed, pb) = a, BOOT[a], BOOT[b]
        W, yt = W_OF[ds]
        diff = np.mean([pa_seed[sd]["accuracy"] - pb_seed[sd]["accuracy"] for sd in SEEDS], axis=0) * 100
        sd_diff = [((pa[sd] == yt).mean() - (pb[sd] == yt).mean()) * 100 for sd in SEEDS]
        mc = [mcnemar_exact(pa[sd] == yt, pb[sd] == yt) for sd in SEEDS]
        return dict(family=family, dataset=ds, a="/".join([a[1], a[2], a[3], a[4]]), b="/".join([b[1], b[2], b[3], b[4]]),
                    diff=float(np.mean(sd_diff)), ci_low=np.percentile(diff, 2.5), ci_high=np.percentile(diff, 97.5),
                    boot_p=float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))),
                    seed_diffs=sd_diff, seed_t_p=seed_level_t(sd_diff), mc_min_p=min(p for *_, p in mc),
                    mcnemar=mc)

    pairs = []
    for ds in DATASETS:
        for prot, rule in (("orig", "final"), ("fixed10", "final"), ("fixed10", "best_val")):
            for i, a in enumerate(MAIN):
                for b in MAIN[i + 1:]:
                    pairs.append(compare(f"schedules/{prot}/{rule}", (ds, "nasnetmobile", prot, rule, a), (ds, "nasnetmobile", prot, rule, b)))
            for a, b in (("adam_sgd", "adam_adam"), ("sgd_adam", "sgd_sgd")):
                pairs.append(compare(f"switch_vs_restart/{prot}/{rule}", (ds, "nasnetmobile", prot, rule, a), (ds, "nasnetmobile", prot, rule, b)))
        for s in MAIN:
            for other in ("mobilenetv2", "efficientnetb0"):
                pairs.append(compare("backbones/orig/final", (ds, "nasnetmobile", "orig", "final", s), (ds, other, "orig", "final", s)))
    PC = pd.DataFrame(pairs)
    for pcol in ("boot_p", "seed_t_p"):                  # Holm within each family and dataset
        col = f"{pcol}_holm"
        PC[col] = np.nan
        for _, g in PC.groupby(["family", "dataset"]):
            ok = g[pcol].notna()
            PC.loc[g.index[ok], col] = holm(g.loc[ok, pcol].to_numpy())
    ref = pd.read_csv(os.path.join(nb1, "stats", "paired_comparisons.csv"))
    mg = PC.merge(ref, on=["family", "dataset", "a", "b"])
    check("paired comparison rows", len(mg), len(ref), 0)
    check("paired accuracy difference", mg["diff"], mg.accuracy_diff, 1e-6)
    check("paired difference interval", mg[["ci_low", "ci_high"]].to_numpy(), mg[["accuracy_diff_ci_low", "accuracy_diff_ci_high"]].to_numpy(), 1e-6)
    check("bootstrap p and Holm-adjusted p", mg[["boot_p", "boot_p_holm"]].to_numpy(), mg[["accuracy_boot_p", "accuracy_boot_p_holm"]].to_numpy(), 1e-9)
    check("McNemar minimum p", mg.mc_min_p, mg.mcnemar_min_p, 1e-9)
    # the deposit rounds per-seed differences to 3 decimals and per-seed McNemar p to 4
    check("per-seed accuracy differences", np.stack(mg.seed_diffs), np.stack(mg.acc_diff_per_seed.map(json.loads)), 5e-4 + 1e-9)
    check("seeds favouring each side", np.stack([[sum(d > 0 for d in x), sum(d < 0 for d in x)] for x in mg.seed_diffs]),
          mg[["seeds_a_better", "seeds_b_better"]].to_numpy(), 0)
    got = np.array([[[b_, c_, p_] for b_, c_, p_ in x] for x in mg.mcnemar], float)
    ref_mc = np.array([json.loads(x) for x in mg.mcnemar_b_c_p_per_seed], float)
    check("per-seed McNemar discordant counts", got[..., :2], ref_mc[..., :2], 0)
    check("per-seed McNemar p", got[..., 2], ref_mc[..., 2], 5e-5 + 1e-9)
    print("2. bootstrap intervals, paired differences, bootstrap and Holm p, per-seed differences, McNemar tests, "
          "seed-level t-tests: recomputed")

    # 3. Validation-based schedule selection and EES (Eq 8) -------------------------------------
    MM = pd.read_csv(os.path.join(nb1, "data", "model_meta.csv"))
    TR = pd.read_csv(os.path.join(nb2, "stats", "timings_raw.csv"))
    TR["allocate_s"] = TR["allocate_s"].fillna(0.0)
    TR["start"] = TR.load_s + TR.allocate_s + TR.first_call_s
    p1 = PR[(PR.protocol == "orig") & (PR.rule == "final") & PR.schedule.isin(MAIN)]
    # Highest mean validation accuracy over the five seeds; a tie goes to the first schedule in alphabetical
    # order, as in the paper (Adam for MobileNetV2 on BDRubberLeaf, where three schedules tie).
    sel = {(ds, a): g.groupby("schedule").val_acc.mean().round(9).idxmax() for (ds, a), g in p1.groupby(["dataset", "arch"])}
    ref_ees = json.load(open(os.path.join(an, "ees_v2.json")))["datasets"]
    rng = np.random.default_rng(1)

    def times(ds, arch, runtime, threads):
        g = TR[(TR.dataset == ds) & (TR.arch == arch) & (TR.runtime == runtime) & (TR.threads.astype(str) == threads)]
        s, w = g.start.to_numpy(), g.warm_call_median_s.to_numpy()
        idx = rng.integers(0, len(s), size=(B, len(s)))
        return np.median(s), np.median(w), np.median(s[idx], axis=1), np.median(w[idx], axis=1)

    out["ees"] = {}
    for ds in DATASETS:
        W, res = None, {}
        for arch in ARCHS:
            s = sel[(ds, arch)]
            check(f"selected schedule {ds}/{arch}", 0 if s == ref_ees[ds]["models"][arch]["schedule"] else 1, 0, 0)
            P = [runs[f"{ds}__{arch}__orig__{s}__s{sd}"][1] for sd in SEEDS]
            n = len(P[0]["test_true"])
            if W is None:
                W = resample_weights(rng, n, B)
            ok = np.stack([(p["test_logits"].argmax(1) == p["test_true"]) for p in P]).astype(np.float32)
            acc, acc_b = ok.mean() * 100, (W @ ok.T / n).mean(1) * 100
            g = float(MM[(MM.arch == arch) & (MM.dataset == ds)].gflops_per_image.iloc[0])
            # Keras and INT8 timings feed sensitivity checks (c) and (d); the call order also fixes the random stream
            T = {k: times(ds, arch, *v) for k, v in {"fp32_4": ("tflite_fp32", "4"), "keras": ("keras", "tf-default"),
                                                      "int8_4": ("tflite_int8dr", "4")}.items()}
            ts, tw, tsb, twb = T["fp32_4"]
            r = dict(schedule=s, acc=acc, gflops=g, t_start_ms=ts * 1000, t_warm_ms=tw * 1000)
            for N in (1, 10, 100):
                r[f"ees_{N}"] = ees(acc, g, ts, tw, N)
                r[f"_b{N}"] = ees(acc_b, g, tsb, twb, N)
                r[f"ees_{N}_ci"] = np.percentile(r[f"_b{N}"], [2.5, 97.5]).tolist()
            best = PR[(PR.dataset == ds) & (PR.arch == arch) & (PR.protocol == "orig") & (PR.rule == "final")
                      & PR.schedule.isin(MAIN)].accuracy.max()          # (e) best single run, selected on test data
            r["sens"] = {"a_linear_gflops": ees(acc, g, ts, tw, 1, "linear"), "b_no_compute": ees(acc, g, ts, tw, 1, "none"),
                         "c_keras": ees(acc, g, T["keras"][0], T["keras"][1], 1), "d_int8": ees(acc, g, T["int8_4"][0], T["int8_4"][1], 1),
                         "e_best_case": ees(best, g, ts, tw, 1)}
            res[arch] = r
        for N in (1, 10, 100):
            first = np.bincount(np.stack([res[a][f"_b{N}"] for a in ARCHS]).argmax(0), minlength=3) / B
            for i, a in enumerate(ARCHS):
                res[a][f"p_rank1_{N}"] = float(first[i])
                del res[a][f"_b{N}"]
        for a in ARCHS:
            rf = ref_ees[ds]["models"][a]
            for k in ("acc", "t_start_ms", "t_warm_ms", "ees_1", "ees_10", "ees_100", "p_rank1_1", "p_rank1_10", "p_rank1_100"):
                check(f"EES {ds}/{a} {k}", res[a][k], rf[k], 1e-6)
            for N in (1, 10, 100):
                check(f"EES {ds}/{a} interval N={N}", res[a][f"ees_{N}_ci"], rf[f"ees_{N}_ci"], 1e-6)
            for k, v in rf["sens"].items():
                check(f"EES {ds}/{a} sensitivity {k}", res[a]["sens"][k], v, 1e-6)
        for k, order in ref_ees[ds]["ranking"].items():
            got = sorted(ARCHS, key=lambda a: -(res[a][k] if k in res[a] else res[a]["sens"][k]))
            check(f"EES ranking {ds} {k}", 0 if got == order else 1, 0, 0)
        out["ees"][ds] = res
    print("3. validation-based schedule choice, EES at N = 1, 10, 100 with intervals, P(rank 1), "
          "sensitivity checks (a) to (e) and rankings: recomputed")

    # 4. Timing medians -------------------------------------------------------------------------
    ref_t = json.load(open(os.path.join(an, "timing_tables.json")))["cells"]
    for (ds, arch, rt, th), g in TR.groupby(["dataset", "arch", "runtime", "threads"]):
        rf = ref_t[f"{ds}/{arch}/{rt}/{th}"]
        check(f"start-up {ds}/{arch}/{rt}/{th}", g.start.median(), rf["startup_s"], 1e-9)
        check(f"warm {ds}/{arch}/{rt}/{th}", g.warm_call_median_s.median() * 1000, rf["warm_ms"], 1e-9)
        check(f"start-up IQR {ds}/{arch}/{rt}/{th}", np.percentile(g.start, [25, 75]), rf["startup_iqr"], 1e-9)
        check(f"warm IQR {ds}/{arch}/{rt}/{th}", np.percentile(g.warm_call_median_s, [25, 75]) * 1000, rf["warm_ms_iqr"], 1e-9)
    print("4. start-up and warm-prediction medians and interquartile ranges for all 30 timing settings: recomputed")

    # Headline numbers ---------------------------------------------------------------------------
    def rng_of(ds, arch, s=None, prot="orig", rule="final"):
        g = PR[(PR.dataset == ds) & (PR.arch == arch) & (PR.protocol == prot) & (PR.rule == rule)]
        m = g[g.schedule.isin(MAIN if s is None else [s])].groupby("schedule").accuracy.mean()
        return m.min(), m.max()                       # range of the five-seed means of the four schedules

    bv = PR[(PR.arch == "nasnetmobile") & (PR.protocol == "fixed10") & (PR.rule == "best_val")].groupby(["dataset", "schedule"]).accuracy.mean()
    spread = {ds: bv[ds].max() - bv[ds].min() for ds in DATASETS}
    sched_fam = PC[PC.family.str.startswith(("schedules", "switch_vs_restart")) & (PC.dataset == "idbgl")]
    bb = PC[(PC.family == "backbones/orig/final") & (PC.dataset == "idbgl")]
    seeds_better = lambda other: int(sum(d < 0 for r in bb[bb.b.str.startswith(other)].itertuples() for d in r.seed_diffs))
    k = lambda ds, a, rt, th: TR[(TR.dataset == ds) & (TR.arch == a) & (TR.runtime == rt) & (TR.threads.astype(str) == th)].start.median()
    tfl = {th: [k(ds, a, "tflite_fp32", th) * 1000 for ds in DATASETS for a in ARCHS] for th in ("1", "4")}
    rank = {ds: max(ARCHS, key=lambda a: out["ees"][ds][a]["ees_1"]) for ds in DATASETS}
    head = [
        ("IDBGL mean accuracy range over schedules, EfficientNet-B0 (%)", "%.2f to %.2f" % rng_of("idbgl", "efficientnetb0"), "95.45 to 96.24"),
        ("Same, NASNetMobile, early stopping (%)", "%.2f to %.2f" % rng_of("idbgl", "nasnetmobile"), "91.73 to 93.32"),
        ("Same, MobileNetV2 (%)", "%.2f to %.2f" % rng_of("idbgl", "mobilenetv2"), "94.11 to 95.25"),
        ("Paired seeds, MobileNetV2 above NASNetMobile on IDBGL", f"{seeds_better('mobilenetv2')} of 20", "17 of 20"),
        ("Paired seeds, EfficientNet-B0 above NASNetMobile on IDBGL", f"{seeds_better('efficientnetb0')} of 20", "20 of 20"),
        ("Spread of six schedules, lowest-val-loss epoch (points)", "%.2f / %.2f" % (spread["idbgl"], spread["rubber"]), "within 0.6 / 0.4"),
        ("Smallest Holm-adjusted seed-level p, IDBGL schedules", "%.3f" % sched_fam.seed_t_p_holm.min(), "0.249 (> 0.24)"),
        ("Keras start-up, NASNetMobile / MobileNetV2, IDBGL (s)", "%.1f / %.1f" % (k("idbgl", "nasnetmobile", "keras", "tf-default"), k("idbgl", "mobilenetv2", "keras", "tf-default")), "16.8 / 3.7"),
        ("TFLite float32 start-up, 4 threads (ms)", "%.0f to %.0f" % (min(tfl["4"]), max(tfl["4"])), "26 to 84"),
        ("TFLite float32 start-up, 1 thread (ms)", "%.0f to %.0f" % (min(tfl["1"]), max(tfl["1"])), "33 to 106"),
        ("Highest EES_1, IDBGL / BDRubberLeaf", f"{rank['idbgl']} / {rank['rubber']}", "mobilenetv2 / mobilenetv2"),
    ]
    # each headline value must equal the published one at the printed precision
    r2 = lambda v, d: np.round(np.asarray(v, float), d)
    check("headline: IDBGL accuracy ranges", r2([rng_of("idbgl", a) for a in ("efficientnetb0", "nasnetmobile", "mobilenetv2")], 2),
          [[95.45, 96.24], [91.73, 93.32], [94.11, 95.25]], 1e-9)
    check("headline: paired seeds", [seeds_better("mobilenetv2"), seeds_better("efficientnetb0")], [17, 20], 0)
    check("headline: schedule spread within 0.6 / 0.4", [spread["idbgl"] <= 0.6, spread["rubber"] <= 0.4], [1, 1], 0)
    check("headline: smallest Holm seed-level p", r2(sched_fam.seed_t_p_holm.min(), 3), 0.249, 1e-9)
    check("headline: Keras start-up", r2([k("idbgl", "nasnetmobile", "keras", "tf-default"), k("idbgl", "mobilenetv2", "keras", "tf-default")], 1),
          [16.8, 3.7], 1e-9)
    check("headline: TFLite start-up ranges", r2([min(tfl["4"]), max(tfl["4"]), min(tfl["1"]), max(tfl["1"])], 0), [26, 84, 33, 106], 1e-9)
    check("headline: highest EES_1", [rank["idbgl"] == "mobilenetv2", rank["rubber"] == "mobilenetv2"], [1, 1], 0)

    print("\nHeadline numbers")
    w = max(len(h[0]) for h in head)
    print(f"{'Quantity':<{w}}  {'Recomputed':<28}  Published")
    for name, got, pub in head:
        print(f"{name:<{w}}  {got:<28}  {pub}")
    out["headline"] = [dict(quantity=a, recomputed=b, published=c) for a, b, c in head]
    out["checks"] = {n: ok for n, ok in CHECKS}
    json.dump(out, open(A.out, "w"), indent=1, default=float)
    bad = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(bad)} of {len(CHECKS)} checks match the deposited values. Wrote {A.out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
