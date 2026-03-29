"""
STEP 2 — TRAIN MODEL
=====================
Trains a 2-layer LSTM on the collected keypoint data.
Produces:  model.tflite   — for Android app
           model.h5        — for desktop testing / fine-tuning
           scaler.json     — normalization params for Android
           scaler.pkl      — normalization params for Python inference
           labels.json     — class names in training order

PIPELINE
  1. Load (30, 225) .npy files from dataset/
  2. Apply geometric normalization per frame
       • hands  → wrist-relative + scale-normalized
       • pose   → shoulder-midpoint-relative + shoulder-width-normalized
  3. Augment training data 4× (noise / jitter / speed / shift)
  4. Fit StandardScaler on training data only (no leakage)
  5. 5-fold cross-validation → stability report
  6. Train final model → evaluate on held-out test set
  7. Export model.tflite + scaler.json for Android

USAGE
  python step2_train.py
  python step2_train.py --data my_dataset --epochs 120
"""

import argparse
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR      = "dataset"
MODEL_H5      = "model.h5"
MODEL_TFLITE  = "model.tflite"
SCALER_PKL    = "scaler.pkl"
SCALER_JSON   = "scaler.json"      # for Android
LABELS_JSON   = "labels.json"
HISTORY_PNG   = "training_history.png"

EPOCHS        = 150
BATCH_SIZE    = 16
LEARNING_RATE = 0.0003
N_FOLDS       = 5
AUG_FACTOR    = 4          # 1 original + 4 augmented = 5× data
SEQUENCE_LEN  = 30
N_FEATURES    = 225
# ─────────────────────────────────────────────


# ═══════════════════════════════════════════
# GEOMETRIC NORMALIZATION
# (must be IDENTICAL in inference / Android)
# ═══════════════════════════════════════════
def normalize_hand(hand_flat: np.ndarray) -> np.ndarray:
    """
    Wrist-relative + scale-normalized.
    Input/output: (63,)
    """
    pts   = hand_flat.reshape(21, 3).copy()
    wrist = pts[0].copy()
    pts  -= wrist
    scale = np.linalg.norm(pts[9])       # wrist → middle-finger MCP distance
    if scale > 1e-6:
        pts /= scale
    return pts.flatten().astype(np.float32)


def normalize_pose(pose_flat: np.ndarray) -> np.ndarray:
    """
    Shoulder-midpoint-relative + shoulder-width-normalized.
    Input/output: (99,)
    """
    pts             = pose_flat.reshape(33, 3).copy()
    l_shoulder      = pts[11].copy()
    r_shoulder      = pts[12].copy()
    midpoint        = (l_shoulder + r_shoulder) / 2.0
    pts            -= midpoint
    shoulder_width  = np.linalg.norm(l_shoulder - r_shoulder)
    if shoulder_width > 1e-6:
        pts /= shoulder_width
    return pts.flatten().astype(np.float32)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply hand + pose normalization to every frame. (30,225) → (30,225)"""
    out = np.zeros_like(seq, dtype=np.float32)
    for i, frame in enumerate(seq):
        lh   = normalize_hand(frame[:63].copy())
        rh   = normalize_hand(frame[63:126].copy())
        pose = normalize_pose(frame[126:].copy())
        out[i] = np.concatenate([lh, rh, pose])
    return out


# ═══════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════
def augment_sequence(seq: np.ndarray, n_copies: int = 1) -> list[np.ndarray]:
    """
    Generate n_copies augmented variants of a (30, 225) sequence.
      0 — Gaussian noise   : sensor/detection jitter
      1 — Temporal jitter  : random frame drops + interpolation
      2 — Speed perturb    : stretch / compress signing speed
      3 — Spatial shift    : signer at different screen position
    """
    results = []
    for _ in range(n_copies):
        choice = np.random.randint(0, 4)

        if choice == 0:
            noise = np.random.normal(0, 0.015, seq.shape).astype(np.float32)
            results.append(seq + noise)

        elif choice == 1:
            aug    = seq.copy()
            n_drop = np.random.randint(1, 4)
            drop_i = np.sort(np.random.choice(range(1, 29), n_drop, replace=False))
            for di in drop_i:
                aug[di] = (aug[di - 1] + aug[min(di + 1, 29)]) / 2.0
            results.append(aug)

        elif choice == 2:
            factor  = np.random.uniform(0.75, 1.25)
            new_len = max(15, int(SEQUENCE_LEN * factor))
            idx     = np.linspace(0, SEQUENCE_LEN - 1, new_len)
            resamp  = np.array([
                seq[int(i)] * (1 - i % 1) + seq[min(int(i) + 1, SEQUENCE_LEN - 1)] * (i % 1)
                for i in idx
            ], dtype=np.float32)
            if new_len >= SEQUENCE_LEN:
                results.append(resamp[:SEQUENCE_LEN])
            else:
                pad = np.tile(resamp[-1], (SEQUENCE_LEN - new_len, 1))
                results.append(np.vstack([resamp, pad]).astype(np.float32))

        else:
            aug      = seq.copy()
            shift_x  = np.random.uniform(-0.08, 0.08)
            shift_y  = np.random.uniform(-0.08, 0.08)
            aug[:, 0::3] += shift_x
            aug[:, 1::3] += shift_y
            results.append(aug.astype(np.float32))

    return results


# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════
def load_dataset(data_dir: str):
    actions = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if len(actions) < 2:
        raise ValueError(
            f"Need at least 2 classes in '{data_dir}'. "
            f"Found: {actions}"
        )

    label_map  = {label: idx for idx, label in enumerate(actions)}
    sequences  = []
    labels     = []

    print(f"\n📂  Loading data from '{data_dir}/'")
    print(f"    Classes: {actions}\n")

    for action in actions:
        class_dir = os.path.join(data_dir, action)
        files     = sorted(f for f in os.listdir(class_dir) if f.endswith(".npy"))
        print(f"    {action:<20s}: {len(files):3d} samples", end="")

        if not files:
            print("  ⚠️  EMPTY — skipping")
            continue

        for fname in files:
            raw = np.load(os.path.join(class_dir, fname))
            if raw.shape != (SEQUENCE_LEN, N_FEATURES):
                print(f"\n  ⚠️  Skipping {fname}: wrong shape {raw.shape}")
                continue
            norm = normalize_sequence(raw)
            sequences.append(norm)
            labels.append(label_map[action])

        print("  ✓")

    if len(sequences) == 0:
        raise ValueError("No valid sequences found. Check dataset folder.")

    return np.array(sequences, dtype=np.float32), np.array(labels), actions


# ═══════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════
def build_model(n_classes: int) -> Sequential:
    """
    Lightweight 2-layer LSTM.
    Small enough for TFLite on-device inference; strong enough for
    medium-sized datasets (~30–150 samples/class after augmentation).
    """
    model = Sequential([
        LSTM(64, return_sequences=True,
               unroll=True,
             input_shape=(SEQUENCE_LEN, N_FEATURES),
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

           LSTM(32, unroll=True, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ])
    return model


# ═══════════════════════════════════════════
# FOLD PREPARATION  (augment + scale)
# ═══════════════════════════════════════════
def prepare_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    n_classes: int,
    fit_scaler: bool = True,
    scaler: StandardScaler | None = None,
):
    """
    1. Augment training split only
    2. Fit (or reuse) StandardScaler on augmented training data
    3. Scale train + val
    Returns: X_tr_s, y_tr_cat, X_val_s, fitted_scaler
    """
    # Augment
    aug_X, aug_y = list(X_tr), list(y_tr)
    for seq, lbl in zip(X_tr, y_tr):
        for a_seq in augment_sequence(seq, n_copies=AUG_FACTOR):
            aug_X.append(a_seq)
            aug_y.append(lbl)

    X_aug = np.array(aug_X, dtype=np.float32)
    y_aug = np.array(aug_y)

    n, sl, nf = X_aug.shape

    # Scaler
    if fit_scaler:
        sc = StandardScaler()
        sc.fit(X_aug.reshape(-1, nf))
    else:
        sc = scaler

    X_tr_s  = sc.transform(X_aug.reshape(-1, nf)).reshape(n, sl, nf)
    X_val_s = sc.transform(X_val.reshape(-1, nf)).reshape(len(X_val), sl, nf)
    y_tr_c  = to_categorical(y_aug, n_classes).astype(np.float32)

    return X_tr_s, y_tr_c, X_val_s, sc


# ═══════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════
def get_callbacks(save_path: str):
    return [
        EarlyStopping(monitor="val_loss", patience=15,
                      min_delta=0.001, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
        ModelCheckpoint(save_path, monitor="val_categorical_accuracy",
                        save_best_only=True, verbose=0),
    ]


# ═══════════════════════════════════════════
# TFLite EXPORT
# ═══════════════════════════════════════════
def export_tflite(model: Sequential, output_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter._experimental_lower_tensor_list_ops = True

    try:
        tflite_model = converter.convert()
    except Exception as exc:
        msg = str(exc)
        if "TensorList" in msg or "Select TF ops" in msg or "Flex" in msg:
            raise RuntimeError(
                "TFLite export requires Select TF Ops/Flex, which is not supported by this mobile runtime. "
                "Use a builtins-compatible model (current script sets LSTM unroll=True) and retrain, then export again."
            ) from exc
        raise

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"    ✅  TFLite model : {output_path}  ({size_kb:.0f} KB)")


def export_scaler_json(sc: StandardScaler, output_path: str):
    """Export scaler params as JSON so Android can apply the same transform."""
    data = {
        "mean":  sc.mean_.tolist(),
        "scale": sc.scale_.tolist(),
        "n_features": int(sc.mean_.shape[0]),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"    ✅  Scaler JSON  : {output_path}")


# ═══════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════
def save_plots(history, fold_results: list):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Sign Language LSTM — Training History", fontsize=14, fontweight="bold")

    h = history.history
    axes[0, 0].plot(h["categorical_accuracy"],     label="Train", lw=2)
    axes[0, 0].plot(h["val_categorical_accuracy"], label="Val",   lw=2)
    axes[0, 0].set_title("Accuracy"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(h["loss"],     label="Train", lw=2)
    axes[0, 1].plot(h["val_loss"], label="Val",   lw=2)
    axes[0, 1].set_title("Loss"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    gap = [(t - v) * 100 for t, v in
           zip(h["categorical_accuracy"], h["val_categorical_accuracy"])]
    axes[1, 0].plot(gap, lw=2, color="orange")
    axes[1, 0].axhline(5,  color="green", ls="--", alpha=0.7, label="Good  (<5%)")
    axes[1, 0].axhline(10, color="red",   ls="--", alpha=0.7, label="Overfit (>10%)")
    axes[1, 0].fill_between(range(len(gap)), gap, alpha=0.1, color="orange")
    axes[1, 0].set_title("Overfitting Monitor"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    if fold_results:
        fold_nums = [r["fold"] for r in fold_results]
        fold_accs = [r["val_acc"] * 100 for r in fold_results]
        colors    = ["green" if a > 90 else "orange" if a > 75 else "red"
                     for a in fold_accs]
        bars = axes[1, 1].bar(fold_nums, fold_accs, color=colors, alpha=0.7, edgecolor="black")
        axes[1, 1].axhline(np.mean(fold_accs), color="blue", ls="--", lw=2,
                            label=f"Mean {np.mean(fold_accs):.1f}%")
        axes[1, 1].set_ylim(0, 105); axes[1, 1].set_title("K-Fold Results")
        axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3, axis="y")
        for bar, acc in zip(bars, fold_accs):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 1, f"{acc:.1f}%",
                             ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(HISTORY_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅  Training plot: {HISTORY_PNG}")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main(data_dir: str, epochs: int):
    print("\n" + "=" * 65)
    print("🚀  SIGN LANGUAGE MODEL TRAINER")
    print("=" * 65)

    # ── Load ─────────────────────────────────
    X, y_raw, actions = load_dataset(data_dir)
    n_classes = len(actions)

    print(f"\n📊  Dataset summary:")
    print(f"    Total samples : {len(X)}")
    print(f"    Classes       : {n_classes}")
    unique, counts = np.unique(y_raw, return_counts=True)
    for idx, cnt in zip(unique, counts):
        print(f"    {actions[idx]:<20s}: {cnt} samples")

    # Save labels
    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(actions, f, ensure_ascii=False, indent=2)
    print(f"\n    ✅  Labels saved : {LABELS_JSON}")

    # ── Train / test split (on originals) ────
    X_train_orig, X_test_orig, y_train_orig, y_test_raw = train_test_split(
        X, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    print(f"\n    Train (pre-aug) : {len(X_train_orig)}")
    print(f"    Test  (held-out): {len(X_test_orig)}")

    # ── K-Fold ───────────────────────────────
    print(f"\n{'='*65}")
    print(f"📊  {N_FOLDS}-FOLD CROSS VALIDATION")
    print(f"{'='*65}")

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_orig, y_train_orig)):
        print(f"\n─── Fold {fold+1}/{N_FOLDS} ───")

        X_tr_f, y_tr_f, X_val_f, _ = prepare_fold(
            X_train_orig[tr_idx], y_train_orig[tr_idx],
            X_train_orig[val_idx], n_classes,
        )
        y_val_f = to_categorical(y_train_orig[val_idx], n_classes)

        print(f"    Train (aug): {len(X_tr_f)}  |  Val: {len(X_val_f)}")

        fm = build_model(n_classes)
        fm.compile(optimizer=Adam(LEARNING_RATE),
                   loss="categorical_crossentropy",
                   metrics=["categorical_accuracy"])
        fm.fit(X_tr_f, y_tr_f,
               validation_data=(X_val_f, y_val_f),
               epochs=epochs, batch_size=BATCH_SIZE,
               callbacks=get_callbacks(f"fold_{fold+1}_model.h5"),
               verbose=0)

        val_loss, val_acc = fm.evaluate(X_val_f, y_val_f, verbose=0)
        fold_results.append({"fold": fold + 1, "val_acc": val_acc, "val_loss": val_loss})
        print(f"    Fold {fold+1} → acc {val_acc*100:.2f}%  loss {val_loss:.4f}")
        del fm

    accs = [r["val_acc"] for r in fold_results]
    print(f"\n    Mean acc : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    if np.std(accs) * 100 < 5:
        print("    ✅  Low variance — stable model")
    else:
        print("    ⚠️  High variance — collect more diverse data")

    # Clean up fold checkpoint files
    for fold in range(N_FOLDS):
        fp = f"fold_{fold+1}_model.h5"
        if os.path.exists(fp):
            os.remove(fp)

    # ── Final model ──────────────────────────
    print(f"\n{'='*65}")
    print(f"🏆  TRAINING FINAL MODEL")
    print(f"{'='*65}")

    X_tr_raw, X_val_raw, y_tr_raw, y_val_raw = train_test_split(
        X_train_orig, y_train_orig,
        test_size=0.15, random_state=42, stratify=y_train_orig,
    )

    X_tr, y_tr, X_val, final_scaler = prepare_fold(
        X_tr_raw, y_tr_raw, X_val_raw, n_classes, fit_scaler=True,
    )
    y_val = to_categorical(y_val_raw, n_classes)

    _, sl, nf = X_tr.shape
    X_test = final_scaler.transform(
        X_test_orig.reshape(-1, nf)
    ).reshape(len(X_test_orig), sl, nf)
    y_test = to_categorical(y_test_raw, n_classes)

    print(f"\n    Train (aug) : {len(X_tr)}")
    print(f"    Val         : {len(X_val)}")
    print(f"    Test        : {len(X_test)}")

    # Class weights
    cw_arr  = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_tr_raw), y=y_tr_raw
    )
    cw_dict = dict(enumerate(cw_arr))

    final_model = build_model(n_classes)
    final_model.compile(optimizer=Adam(LEARNING_RATE),
                        loss="categorical_crossentropy",
                        metrics=["categorical_accuracy"])
    final_model.summary()

    history = final_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=get_callbacks(MODEL_H5),
        verbose=1,
    )

    # ── Evaluation ───────────────────────────
    print(f"\n{'='*65}")
    print(f"📊  EVALUATION ON HELD-OUT TEST SET")
    print(f"{'='*65}")

    test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
    train_acc = history.history["categorical_accuracy"][-1]
    val_acc   = history.history["val_categorical_accuracy"][-1]

    print(f"\n    Train acc  : {train_acc*100:.2f}%")
    print(f"    Val acc    : {val_acc*100:.2f}%")
    print(f"    Test acc   : {test_acc*100:.2f}%")
    print(f"    Test loss  : {test_loss:.4f}")
    gap = (train_acc - val_acc) * 100
    print(f"    Acc gap    : {gap:.2f}%  ", end="")
    print("✅ Good" if gap < 10 else "⚠️  Possible overfit — collect more data")

    y_pred   = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
    y_true   = np.argmax(y_test, axis=1)
    print(f"\n📋  Per-class report:")
    print(classification_report(y_true, y_pred, target_names=actions, digits=3))

    # ── Save artifacts ───────────────────────
    print(f"\n{'='*65}")
    print(f"💾  SAVING ARTIFACTS")
    print(f"{'='*65}\n")

    final_model.save(MODEL_H5)
    print(f"    ✅  Keras model : {MODEL_H5}")

    joblib.dump(final_scaler, SCALER_PKL)
    print(f"    ✅  Scaler pkl  : {SCALER_PKL}")

    export_scaler_json(final_scaler, SCALER_JSON)
    export_tflite(final_model, MODEL_TFLITE)
    save_plots(history, fold_results)

    print(f"\n{'='*65}")
    print(f"✅  DONE")
    print(f"{'='*65}")
    print(f"\n    Artifacts generated:")
    for f in [MODEL_TFLITE, MODEL_H5, SCALER_PKL, SCALER_JSON, LABELS_JSON, HISTORY_PNG]:
        size = f"{os.path.getsize(f)/1024:.0f} KB" if os.path.exists(f) else "?"
        print(f"      {f:<30s} {size}")

    print(f"\n    ⚠️  Ship to Android: {MODEL_TFLITE} + {SCALER_JSON} + {LABELS_JSON}")
    print(f"\n▶   Next step: python step3_test_desktop.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sign Language Model")
    parser.add_argument("--data",   default=DATA_DIR, help="Dataset folder")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(data_dir=args.data, epochs=args.epochs)
