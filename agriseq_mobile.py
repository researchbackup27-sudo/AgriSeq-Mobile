"""AgriSeq-Mobile: reference implementation of the methods in

    AgriSeq-Mobile: cold-start-aware detection of black gram and rubber leaf diseases.

Sections follow the Materials and methods of the paper:

    data split ............ dataset_listing, split
    task model ............ build_model (frozen ImageNet backbone + one Dense layer, Table 5)
    training .............. SCHEDULES, make_optimizer, protocol1_early_stopping, train_run (Table 6)
    statistics ............ metrics_from_cm, resample_weights, boot_metrics, mcnemar_exact, holm,
                            seed_level_t, best_val_epoch
    export ................ export_models (Keras file, TensorFlow Lite float32 and INT8)
    timing ................ time_keras_stages, time_tflite_stages (staged start-up timing)
    Edge-Efficiency Score . ees (Eq 8)

TensorFlow is imported only inside the functions that need it, so the statistics and the score can be
used (and reproduce.py run) without it.
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

# Schedules 1 to 6. Each phase is (optimizer, epochs). A new optimizer is created at every phase,
# so the optimizer state is reset at the switch (and at the restart in the two controls).
SCHEDULES = {
    "adam": [("adam", E_TOTAL)],
    "sgd": [("sgd", E_TOTAL)],
    "adam_sgd": [("adam", E_PHASE), ("sgd", E_PHASE)],
    "sgd_adam": [("sgd", E_PHASE), ("adam", E_PHASE)],
    "adam_adam": [("adam", E_PHASE), ("adam", E_PHASE)],
    "sgd_sgd": [("sgd", E_PHASE), ("sgd", E_PHASE)],
}

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
def make_optimizer(kind):
    import keras
    if kind == "adam":
        return keras.optimizers.Adam(learning_rate=LR["adam"])
    return keras.optimizers.SGD(learning_rate=LR["sgd"], momentum=SGD_MOMENTUM)


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


def train_run(arch, schedule, protocol, seed, X, y, splits, n_classes):
    """Train one configuration and return logits for the validation and test images.

    X: float32 images (N, 224, 224, 3) in [0, 255]; y: int labels; splits: dict of index arrays
    for "train", "val" and "test". protocol: "orig" (Protocol 1, early stopping) or "fixed10"
    (Protocol 2, all 10 epochs; per-epoch logits are returned so that the lowest-validation-loss
    epoch can be selected afterwards with best_val_epoch). For Protocol 1 the returned dict also lists
    the phases in which early stopping fired ("early_stop_events"). Batch order comes from Keras shuffling,
    so seeded runs match the paper's runs statistically, not bit for bit.
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

    tr, va = splits["train"], splits["val"]
    events = []                                              # Protocol 1: where stopping fired
    for ph, (kind, epochs) in enumerate(SCHEDULES[schedule]):
        train_model.compile(optimizer=make_optimizer(kind),     # new optimizer: state reset at the switch
                            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=["acc"])
        es = protocol1_early_stopping()
        cbs = [es] if protocol == "orig" else [KeepEpochLogits()]
        train_model.fit(X[tr], y[tr], batch_size=BATCH, shuffle=True, epochs=epochs,
                        validation_data=(X[va], y[va]), callbacks=cbs, verbose=2)
        if protocol == "orig":
            events += [dict(e, phase=ph) for e in es.events]
    out = {f"{s}_logits": logits(splits[s]) for s in ("val", "test")}
    out.update({f"{s}_true": y[splits[s]] for s in ("val", "test")})
    if protocol == "fixed10":
        out["ep_val_logits"] = np.stack(ep_logits["val"])
        out["ep_test_logits"] = np.stack(ep_logits["test"])
    else:
        out["early_stop_events"] = events
    return out, infer_model


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
    """Paired t-test across seeds: the training run is the unit (df = number of seeds - 1)."""
    from scipy import stats
    return float(stats.ttest_1samp(np.asarray(diffs, float), 0.0).pvalue)


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


# ---------------------------------------------------------------- Edge-Efficiency Score (Eq 8)
def ees(acc, gflops, t_start, t_warm, N, compute="log"):
    """EES_N = Acc / (log10(GFLOPs + 1) * T_task(N)),  T_task(N) = T_start + (N - 1) * T_warm.

    acc in %, times in seconds. compute="linear" uses GFLOPs and compute="none" drops the
    compute term (sensitivity checks)."""
    c = {"log": np.log10(gflops + 1), "linear": gflops, "none": 1.0}[compute]
    return acc / (c * (t_start + (N - 1) * t_warm))
