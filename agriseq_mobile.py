"""AgriSeq-Mobile: reference implementation of the methods in

    AgriSeq-Mobile: cold-start-aware detection of black gram and rubber leaf diseases.

Sections follow the Materials and methods of the paper:

    data split ............ dataset_listing, split
    task model ............ build_model (frozen ImageNet backbone + one Dense layer, Table 5)
    training .............. SCHEDULES, schedule_phases, make_optimizer, protocol1_early_stopping,
                            train_run (Table 6)
    learning-rate tuning .. LR_GRID, TUNE_SEEDS, select_learning_rate, tune_learning_rate
                            (second experiment: tuned rates and training until early stopping)
    statistics ............ metrics_from_cm (balanced accuracy = macro recall), per_class_recall,
                            resample_weights, boot_metrics, mcnemar_exact, holm, seed_level_t,
                            welch_t, signflip_exact, mde_paired_t, best_val_epoch
    export ................ export_models (Keras file, TensorFlow Lite float32 and INT8)
    timing ................ time_keras_stages, time_tflite_stages (staged start-up timing),
                            evict_from_file_cache, file_cache_residency (timing with the model file
                            outside the operating-system file cache)
    Edge-Efficiency Score . ees (Eq 8)

TensorFlow is imported only inside the functions that need it, so the statistics, the learning-rate
selection rule and the score can be used (and reproduce.py and demo.py run) without it.
"""
import glob
import os
import time

import numpy as np

# ---------------------------------------------------------------- settings (Table 6)
IMG = 224
BATCH = 32
SPLIT_SEED = 1
SEEDS = (1, 42, 123, 270, 789)
LR = {"adam": 1e-3, "sgd": 1e-2}
SGD_MOMENTUM = 0.9
E_TOTAL, E_PHASE, PATIENCE = 10, 5, 3

# Second experiment: learning rates tuned on the validation set, then training until early stopping
# with a 50-epoch limit (25 + 25 for two-phase schedules), with the default and with the selected rates.
LONG_E_TOTAL, LONG_E_PHASE = 50, 25
LR_GRID = {"adam": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2), "sgd": (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)}
TUNE_SEEDS = (7, 2027)                    # not among the evaluation SEEDS

# Schedules 1 to 6. Each phase is (optimizer, epochs). A new optimizer is created at every phase,
# so the optimizer state is reset at the switch (and at the restart in the two controls).
def schedule_phases(name, e_total=E_TOTAL, e_phase=E_PHASE):
    """Phases of schedule `name` as (optimizer, epochs): one phase of e_total epochs for Adam and SGD,
    two phases of e_phase epochs for the switching schedules and the restart controls."""
    one = {"adam": "adam", "sgd": "sgd"}
    if name in one:
        return [(one[name], e_total)]
    first, second = name.split("_")
    return [(first, e_phase), (second, e_phase)]


SCHEDULES = {n: schedule_phases(n) for n in ("adam", "sgd", "adam_sgd", "sgd_adam", "adam_adam", "sgd_sgd")}

# Keras application name, and whether the input is scaled to [-1, 1] outside the network.
# EfficientNetB0 rescales internally.
ARCH = {"nasnetmobile": ("NASNetMobile", True), "mobilenetv2": ("MobileNetV2", True),
        "efficientnetb0": ("EfficientNetB0", False)}


# ---------------------------------------------------------------- data split
def dataset_listing(root):
    """Sorted image paths, sorted class names, and labels from the class-folder names."""
    paths = sorted(glob.glob(os.path.join(root, "*/*.*")))
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    lab = {c: i for i, c in enumerate(classes)}
    labels = [lab[os.path.basename(os.path.dirname(p))] for p in paths]
    return paths, classes, labels


def split(paths, labels, seed=SPLIT_SEED):
    """70/20/10 stratified split: the test set is drawn first, validation from the remainder."""
    from sklearn.model_selection import train_test_split
    tv_p, te_p, tv_l, _ = train_test_split(paths, labels, test_size=0.10, random_state=seed, stratify=labels)
    tr_p, va_p, _, _ = train_test_split(tv_p, tv_l, test_size=(0.20 / 0.90), random_state=seed, stratify=tv_l)
    return tr_p, va_p, te_p


def load_images(paths):
    """Decode and resize to 224 x 224 (bilinear, no antialiasing, aspect ratio not kept), float32 in [0, 255]."""
    import tensorflow as tf
    out = np.empty((len(paths), IMG, IMG, 3), np.float32)
    for i, p in enumerate(paths):
        x = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        out[i] = tf.image.resize(x, (IMG, IMG), method="bilinear", antialias=False).numpy()
    return out


# ---------------------------------------------------------------- task model (Table 5)
def build_model(arch, n_classes, with_aug=True):
    """Frozen ImageNet backbone with global average pooling and one Dense layer that outputs logits.

    Returns (train_model, infer_model). The two share weights; only train_model applies augmentation.
    The backbone is called with training=False, so its BatchNorm layers stay in inference mode.
    """
    import keras
    name, scale = ARCH[arch]
    base = getattr(keras.applications, name)(include_top=False, weights="imagenet",
                                             input_shape=(IMG, IMG, 3), pooling="avg")
    base.trainable = False
    prep = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess") if scale else None
    head = keras.layers.Dense(n_classes, name="head")
    aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomContrast(0.2),
    ], name="augmentation")

    def chain(x, train_graph):
        if train_graph:
            x = aug(x)
        if prep is not None:
            x = prep(x)
        return head(base(x, training=False))

    inp = keras.Input((IMG, IMG, 3))
    train_model = keras.Model(inp, chain(inp, with_aug), name=f"{arch}_train")
    inp2 = keras.Input((IMG, IMG, 3))
    infer_model = keras.Model(inp2, chain(inp2, False), name=f"{arch}_infer")
    return train_model, infer_model


# ---------------------------------------------------------------- training
def make_optimizer(kind, lr=None):
    """Adam or SGD with momentum 0.9. lr: dict of learning rates by optimizer; missing optimizers use LR."""
    import keras
    rate = {**LR, **(lr or {})}[kind]
    if kind == "adam":
        return keras.optimizers.Adam(learning_rate=rate)
    return keras.optimizers.SGD(learning_rate=rate, momentum=SGD_MOMENTUM)


def protocol1_early_stopping(patience=PATIENCE):
    """Protocol 1 stopping rule, applied separately in each phase (one fit() call per phase).

    Monitors validation loss with patience 3. When stopping fires, the weights of the phase's
    lowest-validation-loss epoch are restored; when it does not fire, the last epoch is kept.
    This is the behaviour of EarlyStopping(restore_best_weights=True) in Keras 2, written out so
    that it does not depend on the installed Keras version."""
    import keras

    class Protocol1EarlyStopping(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.patience = patience
            self.events = []

        def on_train_begin(self, logs=None):
            self.wait, self.best, self.best_weights, self.best_epoch = 0, np.inf, None, -1

        def on_epoch_end(self, epoch, logs=None):
            cur = float(logs["val_loss"])
            if self.best_weights is None:
                self.best_weights = self.model.get_weights()
            self.wait += 1
            if cur < self.best:
                self.best, self.best_epoch, self.wait = cur, epoch, 0
                self.best_weights = self.model.get_weights()
                return
            if self.wait >= self.patience and epoch > 0:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
                self.events.append({"stopped_at_epoch_in_phase": epoch + 1,
                                    "restored_epoch_in_phase": self.best_epoch + 1})

    return Protocol1EarlyStopping()


def seeded_batches(X, y, idx, seed, shuffle=True):
    """Training batches of BATCH images in an order drawn from one NumPy generator seeded with `seed`.

    The generator is created once per training run and reshuffles after every epoch, so the order
    continues across the phases of a two-phase schedule (each phase is one fit() call on the same
    object) and is the same for every schedule trained with the same seed."""
    import keras

    class SeededBatches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.idx = np.asarray(idx)
            self.rng = np.random.default_rng(seed)
            self.order = self.idx.copy()
            if shuffle:
                self.rng.shuffle(self.order)

        def __len__(self):
            return int(np.ceil(len(self.idx) / BATCH))

        def __getitem__(self, i):
            b = self.order[i * BATCH:(i + 1) * BATCH]
            return X[b], y[b]

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.order)

    return SeededBatches()


def train_run(arch, schedule, protocol, seed, X, y, splits, n_classes, lr=None, e_total=E_TOTAL,
              e_phase=E_PHASE, keep_epoch_logits=None):
    """Train one configuration and return logits for the validation and test images.

    X: float32 images (N, 224, 224, 3) in [0, 255]; y: int labels; splits: dict of index arrays
    for "train", "val" and "test". protocol: "orig" (Protocol 1, early stopping) or "fixed10"
    (Protocol 2, all 10 epochs; per-epoch logits are returned so that the lowest-validation-loss
    epoch can be selected afterwards with best_val_epoch). For Protocol 1 the returned dict also lists
    the phases in which early stopping fired ("early_stop_events").
    lr, e_total, e_phase: learning rates and epoch limits. The main analysis uses the defaults (LR, at
    most 10 epochs); the second experiment uses Protocol 1 with e_total=LONG_E_TOTAL and
    e_phase=LONG_E_PHASE, with the default rates and with the rates chosen by select_learning_rate.
    The validation loss after every epoch is returned as "val_loss_history" (used for tuning).
    keep_epoch_logits: also return the validation and test logits after every epoch ("ep_val_logits",
    "ep_test_logits"), so that the lowest-validation-loss epoch can be evaluated with best_val_epoch;
    always on for Protocol 2, optional for Protocol 1 (used in the second experiment).
    The training batches come from seeded_batches(..., seed): one order per run, continued across phases.
    """
    import keras
    import tensorflow as tf
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    train_model, infer_model = build_model(arch, n_classes, with_aug=True)

    def logits(idx):
        return np.concatenate([infer_model(tf.constant(X[idx[i:i + 64]]), training=False).numpy()
                               for i in range(0, len(idx), 64)])

    ep_logits = {"val": [], "test": []}

    class KeepEpochLogits(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            for s in ("val", "test"):
                ep_logits[s].append(logits(splits[s]))

    keep = (protocol == "fixed10") if keep_epoch_logits is None else (keep_epoch_logits or protocol == "fixed10")
    va = splits["val"]
    train_batches = seeded_batches(X, y, splits["train"], seed)  # created once: order continues across phases
    events, val_loss = [], []                                # Protocol 1: where stopping fired
    for ph, (kind, epochs) in enumerate(schedule_phases(schedule, e_total, e_phase)):
        train_model.compile(optimizer=make_optimizer(kind, lr),  # new optimizer: state reset at the switch
                            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=["acc"])
        es = protocol1_early_stopping()
        cbs = ([KeepEpochLogits()] if keep else []) + ([es] if protocol == "orig" else [])
        hist = train_model.fit(train_batches, epochs=epochs, validation_data=(X[va], y[va]),
                               callbacks=cbs, verbose=2)
        val_loss += [float(v) for v in hist.history["val_loss"]]
        if protocol == "orig":
            events += [dict(e, phase=ph) for e in es.events]
    out = {f"{s}_logits": logits(splits[s]) for s in ("val", "test")}
    out.update({f"{s}_true": y[splits[s]] for s in ("val", "test")})
    out["val_loss_history"] = val_loss
    if keep:
        out["ep_val_logits"] = np.stack(ep_logits["val"])
        out["ep_test_logits"] = np.stack(ep_logits["test"])
    if protocol == "orig":
        out["early_stop_events"] = events
    return out, infer_model


# ---------------------------------------------------------------- learning-rate tuning
def select_learning_rate(min_val_loss):
    """Selection rule of the second experiment.

    min_val_loss: dict {learning rate: list of the lowest validation loss of each tuning run (one per
    tuning seed)}. The rate with the lowest mean is selected; a tie goes to the smaller rate. Test
    data never enter the choice. Returns (rate, at_grid_edge), where at_grid_edge is True when the
    selected rate is the smallest or largest rate of the grid (reported as such; the grid is not
    extended afterwards)."""
    rates = sorted(min_val_loss)
    means = {r: round(float(np.mean(min_val_loss[r])), 10) for r in rates}   # round so that exact ties go to the smaller rate
    best = min(rates, key=lambda r: (means[r], r))
    return best, best in (rates[0], rates[-1])


def tune_learning_rate(arch, optimizer, X, y, splits, n_classes, grid=None, seeds=TUNE_SEEDS,
                       e_total=LONG_E_TOTAL):
    """Tune the learning rate of one optimizer ("adam" or "sgd") for one backbone and dataset.

    Every rate of the grid (LR_GRID by default) is trained with the single-optimizer schedule,
    Protocol 1 and an e_total-epoch limit, once per tuning seed; each run contributes its lowest
    validation loss. Returns (selected rate, at_grid_edge, {rate: [lowest validation loss per seed]})."""
    grid = grid or LR_GRID[optimizer]
    table = {}
    for rate in grid:
        table[rate] = []
        for seed in seeds:
            out, _ = train_run(arch, optimizer, "orig", seed, X, y, splits, n_classes,
                               lr={optimizer: rate}, e_total=e_total, e_phase=e_total)
            table[rate].append(min(out["val_loss_history"]))
    rate, edge = select_learning_rate(table)
    return rate, edge, table


def export_models(infer_model, out_dir):
    """Save the inference model as a Keras file and export it, through a SavedModel, to TensorFlow Lite
    with float32 weights and with dynamic-range INT8 weight quantization. Returns the three paths."""
    import tensorflow as tf
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, f) for k, f in
             (("keras", "model.keras"), ("fp32", "model_fp32.tflite"), ("int8", "model_int8dr.tflite"))}
    infer_model.save(paths["keras"])
    sm = os.path.join(out_dir, "saved_model")
    infer_model.export(sm, verbose=False)
    for kind in ("fp32", "int8"):
        conv = tf.lite.TFLiteConverter.from_saved_model(sm)
        if kind == "int8":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        with open(paths[kind], "wb") as f:
            f.write(conv.convert())
    return paths


# ---------------------------------------------------------------- statistics
def softmax_ce(logits, y):
    m = logits.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(1))
    return float((lse - logits[np.arange(len(y)), y]).mean())


def best_val_epoch(ep_val_logits, val_true):
    """Protocol 2: index of the epoch with the lowest validation cross-entropy."""
    return int(np.argmin([softmax_ce(L, val_true) for L in ep_val_logits]))


def cm_of(y, p, K):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def metrics_from_cm(cm):
    """cm: (..., K, K), rows true, columns predicted. Same values as scikit-learn with zero_division=0."""
    cm = cm.astype(float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    true_n, pred_n = cm.sum(-1), cm.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_n > 0, tp / pred_n, 0.0)
        rec = np.where(true_n > 0, tp / true_n, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    n = true_n.sum(-1)
    w = true_n / n[..., None]
    return {"accuracy": tp.sum(-1) / n, "macro_precision": prec.mean(-1), "macro_recall": rec.mean(-1),
            "macro_f1": f1.mean(-1), "weighted_precision": (prec * w).sum(-1),
            "weighted_recall": (rec * w).sum(-1), "weighted_f1": (f1 * w).sum(-1)}


def resample_weights(rng, n, B):
    """B bootstrap resamples of n test images, as multiplicity weights (B x n).

    All runs of a dataset share the same test images, so one weight matrix is applied to every run:
    this pairs the resamples across seeds and across the configurations being compared."""
    idx = rng.integers(0, n, size=(B, n))
    return np.bincount((np.arange(B)[:, None] * n + idx).ravel(), minlength=B * n).reshape(B, n).astype(np.float32)


def boot_metrics(W, y, p, K):
    onehot = np.zeros((len(y), K * K), np.float32)
    onehot[np.arange(len(y)), y * K + p] = 1
    return metrics_from_cm((W @ onehot).reshape(-1, K, K))


def mcnemar_exact(a_ok, b_ok):
    """Exact two-sided McNemar test on the discordant test images of two runs."""
    from scipy.stats import binom
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = 1.0 if b + c == 0 else min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))
    return b, c, p


def holm(ps):
    """Holm step-down adjustment."""
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def seed_level_t(diffs):
    """Paired t-test across seeds: the training run is the unit (df = number of seeds - 1).

    Runs are paired by seed: they share the training batch order and the starting random state, and
    within a backbone also the initial weights of the classification layer. When every per-seed
    difference is the same, the t statistic is undefined; use signflip_exact for such comparisons."""
    from scipy import stats
    d = np.asarray(diffs, float)
    if np.allclose(d, d[0]):                # zero variance: 1.0 if all differences are 0, else undefined
        return 1.0 if np.allclose(d, 0) else float("nan")
    return float(stats.ttest_1samp(d, 0.0).pvalue)


def welch_t(acc_a, acc_b):
    """Welch's unpaired t-test on the per-seed accuracies of two configurations (ignores the pairing)."""
    from scipy import stats
    a, b = np.asarray(acc_a, float), np.asarray(acc_b, float)
    if a.std() + b.std() == 0:              # both arms constant: 1.0 if equal, else undefined
        return 1.0 if np.isclose(a.mean(), b.mean()) else float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def signflip_exact(diffs):
    """Exact two-sided sign-flip permutation test of the mean per-seed difference.

    All 2^n sign patterns are enumerated. With n = 5 seeds the smallest attainable p is 2/32 = 0.0625,
    so this test cannot reach p < 0.05 with five seeds."""
    import itertools
    d = np.asarray(diffs, float)
    if np.allclose(d, 0):
        return 1.0
    null = [abs((d * np.array(s)).mean()) for s in itertools.product((1.0, -1.0), repeat=len(d))]
    return float(np.mean(np.array(null) >= abs(d.mean()) - 1e-12))


def mde_paired_t(sd_diff, n=len(SEEDS), alpha=0.05, power=0.8):
    """Smallest mean difference that a two-sided paired t-test with n pairs detects with the given power,
    from the standard deviation of the per-seed differences (noncentral t distribution, df = n - 1).
    With n = 5, alpha = 0.05 and power = 0.8 the required noncentrality is about 3.76."""
    from scipy import stats
    from scipy.optimize import brentq
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)

    def pw(nc):
        return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)

    if not sd_diff > 0:
        return float("nan")
    nc = brentq(lambda x: pw(x) - power, 0.5, 8.0)
    return float(nc * sd_diff / np.sqrt(n))


def per_class_recall(cm):
    """Recall of each class from a confusion matrix (rows true, columns predicted). Their mean is the
    balanced accuracy ("macro_recall" in metrics_from_cm)."""
    cm = np.asarray(cm, float)
    return np.diag(cm) / cm.sum(1)


# ---------------------------------------------------------------- timing
def time_keras_stages(model_path, x, warm_calls=15):
    """Staged start-up timing of a saved Keras model in the current process.

    x: one preprocessed image, float32 array of shape (1, 224, 224, 3) with values in [0, 255].
    Returns seconds for loading the file, the first model.predict call, and the median of
    warm_calls further calls. Start-up time = load + first. In the paper every measurement ran in
    a new Python process, so the import of TensorFlow (timed separately, excluded) and the model
    were initialized from scratch each time; call this as the first thing in a fresh process."""
    import keras
    t = {}
    t0 = time.perf_counter()
    m = keras.models.load_model(model_path)
    t["load_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.predict(x, verbose=0)
    t["first_call_s"] = time.perf_counter() - t0
    warm = []
    for _ in range(warm_calls):
        t0 = time.perf_counter()
        m.predict(x, verbose=0)
        warm.append(time.perf_counter() - t0)
    t["warm_call_median_s"] = float(np.median(warm))
    t["startup_s"] = t["load_s"] + t["first_call_s"]
    return t


def time_tflite_stages(model_path, x, threads=4, warm_calls=15):
    """Staged start-up timing of a TensorFlow Lite file in the current process.

    x: one preprocessed image, float32 array of shape (1, 224, 224, 3) with values in [0, 255].
    Returns seconds for: load, allocate, first prediction, and the median of warm_calls further
    predictions. Start-up time = load + allocate + first. The paper used the LiteRT interpreter
    (ai-edge-litert); tf.lite.Interpreter is used only if LiteRT is not installed. Every measurement
    in the paper ran in a new Python process, and the runtime import was timed separately and excluded."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    t = {}
    t0 = time.perf_counter()
    it = Interpreter(model_path=model_path, num_threads=threads)
    t["load_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    it.allocate_tensors()
    t["allocate_s"] = time.perf_counter() - t0
    inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

    def call():
        it.set_tensor(inp, x)
        it.invoke()
        return it.get_tensor(out)

    t0 = time.perf_counter()
    call()
    t["first_call_s"] = time.perf_counter() - t0
    warm = []
    for _ in range(warm_calls):
        t0 = time.perf_counter()
        call()
        warm.append(time.perf_counter() - t0)
    t["warm_call_median_s"] = float(np.median(warm))
    t["startup_s"] = t["load_s"] + t["allocate_s"] + t["first_call_s"]
    return t


def evict_from_file_cache(path):
    """Remove a file's pages from the operating-system file cache (Linux), so that the next process that
    loads the file reads it from storage. The file is synced first, then posix_fadvise with
    POSIX_FADV_DONTNEED is applied. Check the result with file_cache_residency. The runtime libraries
    themselves stay cached."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def file_cache_residency(path):
    """Share (0 to 1) of a file's pages that are held in the operating-system file cache, measured with
    mincore() on a read-only memory mapping (creating the mapping reads nothing from the file)."""
    import ctypes
    import ctypes.util
    import mmap
    size = os.path.getsize(path)
    if size == 0:
        return 1.0
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    page = mmap.PAGESIZE
    n_pages = (size + page - 1) // page
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
        view = np.frombuffer(mm, dtype=np.uint8)
        try:
            vec = (ctypes.c_ubyte * n_pages)()
            if libc.mincore(ctypes.c_void_p(view.__array_interface__["data"][0]), ctypes.c_size_t(size), vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
        finally:
            del view
            mm.close()
    return float(np.mean(np.frombuffer(vec, dtype=np.uint8) & 1))


# ---------------------------------------------------------------- Edge-Efficiency Score (Eq 8)
def ees(acc, gflops, t_start, t_warm, N, compute="log"):
    """EES_N = Acc / (log10(GFLOPs + 1) * T_task(N)),  T_task(N) = T_start + (N - 1) * T_warm.

    acc in %, times in seconds. compute="linear" uses GFLOPs and compute="none" drops the
    compute term (sensitivity checks)."""
    c = {"log": np.log10(gflops + 1), "linear": gflops, "none": 1.0}[compute]
    return acc / (c * (t_start + (N - 1) * t_warm))
