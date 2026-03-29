"""
STEP 2 — TRAIN LIP READING MODEL
==================================
Trains lip landmark models using LSTM, BiLSTM, or 3D CNN.
Produces architecture-specific model, scaler, labels, and plot artifacts.

PIPELINE
  1. Load (30, 90) .npy files from lip_dataset/
  2. Normalise each frame: centre at mouth centre, scale by mouth width
  3. Augment training data 4×
  4. Fit StandardScaler on training data only
  5. 5-fold CV
  6. Train final model + evaluate on test set
  7. Export TFLite + scaler for mobile

USAGE
  python lip_train.py
  python lip_train.py --data lip_dataset --epochs 120
    python lip_train.py --arch bilstm
    python lip_train.py --arch 3dcnn
"""

import argparse
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv3D,
    Dense,
    Dropout,
    GlobalAveragePooling3D,
    MaxPooling3D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR      = "lip_dataset"
MODEL_H5      = "lip_model.h5"
MODEL_TFLITE  = "lip_model.tflite"
SCALER_PKL    = "lip_scaler.pkl"
SCALER_JSON   = "lip_scaler.json"
LABELS_JSON   = "lip_labels.json"
HISTORY_PNG   = "lip_training_history.png"

EPOCHS        = 150
BATCH_SIZE    = 16
LEARNING_RATE = 0.0003
N_FOLDS       = 5
AUG_FACTOR    = 4          # 1 original + 4 augmented = 5× data
SEQUENCE_LEN  = 30
N_FEATURES    = 90
N_LIP_POINTS  = 30
N_COORDS      = 3
# ─────────────────────────────────────────────

ARCH_ALIASES = {
    "lstm": "lstm",
    "bilstm": "bilstm",
    "bilstem": "bilstm",
    "3dcnn": "3dcnn",
    "3d-cnn": "3dcnn",
    "cnn3d": "3dcnn",
}


# ═══════════════════════════════════════════
# GEOMETRIC NORMALISATION (lip‑specific)
# ═══════════════════════════════════════════
def normalise_lip_frame(frame: np.ndarray) -> np.ndarray:
    """
    frame: (90,) – 30 lip landmarks (x,y,z)
    Returns normalised frame (90,):
        - translate so that mouth centre (mean of all points) is at (0,0,0)
        - scale by mouth width (distance between leftmost and rightmost point)
    """
    pts = frame.reshape(30, 3).copy()          # 30 points
    centre = pts.mean(axis=0)                   # (3,)
    pts -= centre

    # mouth width = distance between leftmost and rightmost point in x
    left  = pts[:, 0].min()
    right = pts[:, 0].max()
    width = right - left
    if width > 1e-6:
        pts /= width

    return pts.flatten().astype(np.float32)


def normalise_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply normalisation to every frame of a (30,90) sequence."""
    out = np.zeros_like(seq, dtype=np.float32)
    for i, frame in enumerate(seq):
        out[i] = normalise_lip_frame(frame)
    return out


# ═══════════════════════════════════════════
# DATA AUGMENTATION (same as sign version)
# ═══════════════════════════════════════════
def augment_sequence(seq: np.ndarray, n_copies: int = 1) -> list[np.ndarray]:
    results = []
    for _ in range(n_copies):
        choice = np.random.randint(0, 4)

        if choice == 0:               # Gaussian noise
            noise = np.random.normal(0, 0.015, seq.shape).astype(np.float32)
            results.append(seq + noise)

        elif choice == 1:             # Temporal jitter (drop & interpolate)
            aug = seq.copy()
            n_drop = np.random.randint(1, 4)
            drop_i = np.sort(np.random.choice(range(1, 29), n_drop, replace=False))
            for di in drop_i:
                aug[di] = (aug[di-1] + aug[min(di+1, 29)]) / 2.0
            results.append(aug)

        elif choice == 2:             # Speed perturbation
            factor = np.random.uniform(0.75, 1.25)
            new_len = max(15, int(SEQUENCE_LEN * factor))
            idx = np.linspace(0, SEQUENCE_LEN-1, new_len)
            resamp = np.array([
                seq[int(i)] * (1 - i%1) + seq[min(int(i)+1, SEQUENCE_LEN-1)] * (i%1)
                for i in idx
            ], dtype=np.float32)
            if new_len >= SEQUENCE_LEN:
                results.append(resamp[:SEQUENCE_LEN])
            else:
                pad = np.tile(resamp[-1], (SEQUENCE_LEN - new_len, 1))
                results.append(np.vstack([resamp, pad]).astype(np.float32))

        else:                         # Spatial shift (x,y only)
            aug = seq.copy()
            shift_x = np.random.uniform(-0.08, 0.08)
            shift_y = np.random.uniform(-0.08, 0.08)
            aug[:, 0::3] += shift_x
            aug[:, 1::3] += shift_y
            results.append(aug.astype(np.float32))

    return results


# ═══════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════
def load_dataset(data_dir: str, arch: str):
    all_actions = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if len(all_actions) < 2:
        raise ValueError(f"Need at least 2 classes in '{data_dir}'.")

    actions = []
    sequences = []
    labels = []

    has_any_crop = False
    if arch == "3dcnn":
        for action in all_actions:
            class_dir = os.path.join(data_dir, action)
            if not os.path.isdir(class_dir):
                continue
            if any(f.endswith("_crop.npy") for f in os.listdir(class_dir)):
                has_any_crop = True
                break

    print(f"\n📂  Loading data from '{data_dir}/'")
    print(f"    Found folders: {all_actions}\n")

    for action in all_actions:
        class_dir = os.path.join(data_dir, action)
        files = sorted(f for f in os.listdir(class_dir) if f.endswith(".npy"))
        print(f"    {action:<20s}: {len(files):3d} samples", end="")
        if not files:
            print("  ⚠️  EMPTY")
            continue

        class_sequences = []
        for fname in files:
            raw = np.load(os.path.join(class_dir, fname))
            if arch in {"lstm", "bilstm"}:
                if fname.endswith("_crop.npy"):
                    continue
                if raw.shape != (SEQUENCE_LEN, N_FEATURES):
                    print(f"\n  ⚠️  Skipping {fname}: wrong shape {raw.shape}")
                    continue
                class_sequences.append(normalise_sequence(raw))
            elif arch == "3dcnn":
                if has_any_crop and not fname.endswith("_crop.npy"):
                    continue
                if raw.ndim == 4 and raw.shape[0] == SEQUENCE_LEN and raw.shape[-1] == 1:
                    crop = raw.astype(np.float32)
                    if crop.max() > 1.0:
                        crop /= 255.0
                    class_sequences.append(crop)
                elif (not has_any_crop) and raw.shape == (SEQUENCE_LEN, N_FEATURES):
                    lm = normalise_sequence(raw).reshape(SEQUENCE_LEN, N_LIP_POINTS, N_COORDS, 1)
                    class_sequences.append(lm.astype(np.float32))
                else:
                    print(f"\n  ⚠️  Skipping {fname}: wrong shape {raw.shape}")
                    continue

        if class_sequences:
            class_idx = len(actions)
            actions.append(action)
            sequences.extend(class_sequences)
            labels.extend([class_idx] * len(class_sequences))
            print("  ✓")
        else:
            print("  ⚠️  NO VALID FILES")

    if len(actions) < 2 or len(sequences) == 0:
        raise ValueError("No valid sequences found.")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels)

    if arch in {"lstm", "bilstm"} and X.ndim != 3:
        raise ValueError(
            f"No landmark tensors found for {arch}. Expected shape (N,{SEQUENCE_LEN},{N_FEATURES}). "
            "Use lip_collect.py and train with *_lm.npy files."
        )
    if arch == "3dcnn" and X.ndim != 5:
        raise ValueError("No crop tensors found for 3dcnn. Expected shape (N,T,H,W,1).")

    print(f"\n    Using classes ({len(actions)}): {actions}")
    return X, y, actions


# ═══════════════════════════════════════════
# MODEL (input shape now (30,90))
# ═══════════════════════════════════════════
def normalise_arch_name(arch: str) -> str:
    key = arch.strip().lower()
    if key not in ARCH_ALIASES:
        valid = ", ".join(sorted(ARCH_ALIASES))
        raise ValueError(f"Unknown architecture '{arch}'. Use one of: {valid}")
    return ARCH_ALIASES[key]


def get_artifact_paths(arch: str) -> dict[str, str]:
    if arch == "lstm":
        prefix = "lip"
    else:
        prefix = f"lip_{arch}"

    return {
        "model_h5": f"{prefix}_model.h5",
        "model_tflite": f"{prefix}_model.tflite",
        "scaler_pkl": f"{prefix}_scaler.pkl",
        "scaler_json": f"{prefix}_scaler.json",
        "labels_json": f"{prefix}_labels.json",
        "history_png": f"{prefix}_training_history.png",
    }


def reshape_for_model(X: np.ndarray, arch: str) -> np.ndarray:
    if arch in {"lstm", "bilstm"}:
        return X
    if arch == "3dcnn":
        if X.ndim == 5:
            return X
        return X.reshape(len(X), SEQUENCE_LEN, N_LIP_POINTS, N_COORDS, 1)
    raise ValueError(f"Unsupported architecture: {arch}")


def build_model(n_classes: int, arch: str, input_shape=None) -> Sequential:
    if arch == "lstm":
        return build_lstm_model(n_classes)
    if arch == "bilstm":
        return build_bilstm_model(n_classes)
    if arch == "3dcnn":
        return build_3dcnn_model(n_classes, input_shape)
    raise ValueError(f"Unsupported architecture: {arch}")


def build_lstm_model(n_classes: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, unroll=True,
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


def build_bilstm_model(n_classes: int) -> Sequential:
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
            input_shape=(SEQUENCE_LEN, N_FEATURES),
        ),
        BatchNormalization(),
        Dropout(0.4),

        Bidirectional(LSTM(32, kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dropout(0.4),

        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ])
    return model


def build_3dcnn_model(n_classes: int, input_shape=None) -> Sequential:
    cnn_input_shape = input_shape or (SEQUENCE_LEN, N_LIP_POINTS, N_COORDS, 1)
    model = Sequential([
        Conv3D(
            32,
            kernel_size=(3, 3, 3),
            activation="relu",
            padding="same",
            input_shape=cnn_input_shape,
            kernel_regularizer=l2(0.0005),
        ),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 1)),
        Dropout(0.25),

        Conv3D(64, kernel_size=(3, 3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 1)),
        Dropout(0.3),

        Conv3D(96, kernel_size=(3, 3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        GlobalAveragePooling3D(),

        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(n_classes, activation="softmax"),
    ])
    return model


# ═══════════════════════════════════════════
# FOLD PREPARATION (augment + scale)
# ═══════════════════════════════════════════
def prepare_fold(X_tr, y_tr, X_val, n_classes, arch: str, fit_scaler=True, scaler=None):
    if arch == "3dcnn":
        X_tr_s = X_tr.astype(np.float32)
        X_val_s = X_val.astype(np.float32)
        y_tr_c = to_categorical(y_tr, n_classes).astype(np.float32)
        return X_tr_s, y_tr_c, X_val_s, None

    # Augment
    aug_X, aug_y = list(X_tr), list(y_tr)
    for seq, lbl in zip(X_tr, y_tr):
        for a_seq in augment_sequence(seq, n_copies=AUG_FACTOR):
            aug_X.append(a_seq)
            aug_y.append(lbl)

    X_aug = np.array(aug_X, dtype=np.float32)
    y_aug = np.array(aug_y)

    n, sl, nf = X_aug.shape

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
# CALLBACKS, TFLITE EXPORT, PLOTS (same as sign version)
# ═══════════════════════════════════════════
def get_callbacks(save_path: str):
    return [
        EarlyStopping(monitor="val_loss", patience=15, min_delta=0.001,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(save_path, monitor="val_categorical_accuracy",
                        save_best_only=True, verbose=0),
    ]


def export_tflite(model, output_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter._experimental_lower_tensor_list_ops = True
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"    ✅  TFLite model : {output_path}  ({size_kb:.0f} KB)")


def export_scaler_json(sc, output_path: str):
    if sc is None:
        data = {
            "type": "none",
            "note": "No scaler used for this architecture"
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"    ✅  Scaler JSON  : {output_path} (no-op)")
        return

    data = {
        "mean": sc.mean_.tolist(),
        "scale": sc.scale_.tolist(),
        "n_features": int(sc.mean_.shape[0]),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    ✅  Scaler JSON  : {output_path}")


def save_plots(history, fold_results, arch: str, output_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Lip Reading {arch.upper()} — Training History", fontsize=14, fontweight="bold")

    h = history.history
    axes[0,0].plot(h["categorical_accuracy"], label="Train", lw=2)
    axes[0,0].plot(h["val_categorical_accuracy"], label="Val", lw=2)
    axes[0,0].set_title("Accuracy"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(h["loss"], label="Train", lw=2)
    axes[0,1].plot(h["val_loss"], label="Val", lw=2)
    axes[0,1].set_title("Loss"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    gap = [(t - v)*100 for t, v in zip(h["categorical_accuracy"], h["val_categorical_accuracy"])]
    axes[1,0].plot(gap, lw=2, color="orange")
    axes[1,0].axhline(5, color="green", ls="--", alpha=0.7, label="Good (<5%)")
    axes[1,0].axhline(10, color="red", ls="--", alpha=0.7, label="Overfit (>10%)")
    axes[1,0].fill_between(range(len(gap)), gap, alpha=0.1, color="orange")
    axes[1,0].set_title("Overfitting Monitor"); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    if fold_results:
        fold_nums = [r["fold"] for r in fold_results]
        fold_accs = [r["val_acc"]*100 for r in fold_results]
        colors = ["green" if a>90 else "orange" if a>75 else "red" for a in fold_accs]
        bars = axes[1,1].bar(fold_nums, fold_accs, color=colors, alpha=0.7, edgecolor="black")
        axes[1,1].axhline(np.mean(fold_accs), color="blue", ls="--", lw=2,
                           label=f"Mean {np.mean(fold_accs):.1f}%")
        axes[1,1].set_ylim(0,105); axes[1,1].set_title("K-Fold Results")
        axes[1,1].legend(); axes[1,1].grid(alpha=0.3, axis="y")
        for bar, acc in zip(bars, fold_accs):
            axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                           f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅  Training plot: {output_path}")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main(data_dir: str, epochs: int, arch: str):
    arch = normalise_arch_name(arch)
    artifacts = get_artifact_paths(arch)

    print("\n" + "=" * 65)
    print("👄  LIP READING MODEL TRAINER")
    print("=" * 65)
    print(f"    Architecture : {arch}")

    X, y_raw, actions = load_dataset(data_dir, arch)
    n_classes = len(actions)

    print(f"\n📊  Dataset summary:")
    print(f"    Total samples : {len(X)}")
    print(f"    Classes       : {n_classes}")
    unique, counts = np.unique(y_raw, return_counts=True)
    for idx, cnt in zip(unique, counts):
        print(f"    {actions[idx]:<20s}: {cnt} samples")

    with open(artifacts["labels_json"], "w", encoding="utf-8") as f:
        json.dump(actions, f, ensure_ascii=False, indent=2)
    print(f"\n    ✅  Labels saved : {artifacts['labels_json']}")

    # Train / test split
    X_train_orig, X_test_orig, y_train_orig, y_test_raw = train_test_split(
        X, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    print(f"\n    Train (pre-aug) : {len(X_train_orig)}")
    print(f"    Test  (held-out): {len(X_test_orig)}")

    # K-Fold CV
    print(f"\n{'='*65}")
    print(f"📊  {N_FOLDS}-FOLD CROSS VALIDATION")
    print(f"{'='*65}")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_orig, y_train_orig)):
        print(f"\n─── Fold {fold+1}/{N_FOLDS} ───")
        X_tr_f, y_tr_f, X_val_f, _ = prepare_fold(
            X_train_orig[tr_idx], y_train_orig[tr_idx],
            X_train_orig[val_idx], n_classes, arch=arch
        )
        X_tr_f = reshape_for_model(X_tr_f, arch)
        X_val_f = reshape_for_model(X_val_f, arch)
        y_val_f = to_categorical(y_train_orig[val_idx], n_classes)
        print(f"    Train (aug): {len(X_tr_f)}  |  Val: {len(X_val_f)}")

        fm = build_model(n_classes, arch, input_shape=X_tr_f.shape[1:])
        fm.compile(optimizer=Adam(LEARNING_RATE),
                   loss="categorical_crossentropy",
                   metrics=["categorical_accuracy"])
        fm.fit(X_tr_f, y_tr_f,
               validation_data=(X_val_f, y_val_f),
               epochs=epochs, batch_size=BATCH_SIZE,
               callbacks=get_callbacks(f"fold_{fold+1}_{arch}_model.h5"),
               verbose=0)
        val_loss, val_acc = fm.evaluate(X_val_f, y_val_f, verbose=0)
        fold_results.append({"fold": fold+1, "val_acc": val_acc, "val_loss": val_loss})
        print(f"    Fold {fold+1} → acc {val_acc*100:.2f}%  loss {val_loss:.4f}")
        del fm

    accs = [r["val_acc"] for r in fold_results]
    print(f"\n    Mean acc : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    if np.std(accs)*100 < 5:
        print("    ✅  Low variance — stable model")
    else:
        print("    ⚠️  High variance — collect more diverse data")

    # Clean up fold files
    for fold in range(N_FOLDS):
        fp = f"fold_{fold+1}_{arch}_model.h5"
        if os.path.exists(fp):
            os.remove(fp)

    # Final model
    print(f"\n{'='*65}")
    print(f"🏆  TRAINING FINAL MODEL")
    print(f"{'='*65}")

    X_tr_raw, X_val_raw, y_tr_raw, y_val_raw = train_test_split(
        X_train_orig, y_train_orig,
        test_size=0.15, random_state=42, stratify=y_train_orig,
    )

    X_tr_seq, y_tr, X_val_seq, final_scaler = prepare_fold(
        X_tr_raw, y_tr_raw, X_val_raw, n_classes, arch=arch, fit_scaler=True
    )
    X_tr = reshape_for_model(X_tr_seq, arch)
    X_val = reshape_for_model(X_val_seq, arch)
    y_val = to_categorical(y_val_raw, n_classes)

    if final_scaler is None:
        X_test_seq = X_test_orig.astype(np.float32)
    else:
        _, sl, nf = X_tr_seq.shape
        X_test_seq = final_scaler.transform(
            X_test_orig.reshape(-1, nf)
        ).reshape(len(X_test_orig), sl, nf)
    X_test = reshape_for_model(X_test_seq, arch)
    y_test = to_categorical(y_test_raw, n_classes)

    print(f"\n    Train (aug) : {len(X_tr)}")
    print(f"    Val         : {len(X_val)}")
    print(f"    Test        : {len(X_test)}")

    present_classes = np.unique(y_tr_raw).astype(int)
    cw_arr = class_weight.compute_class_weight(
        class_weight="balanced", classes=present_classes, y=y_tr_raw
    )
    # Keras expects class ids as keys; keep neutral weight for classes absent in this split.
    cw_dict = {int(i): 1.0 for i in range(n_classes)}
    cw_dict.update({int(cls): float(w) for cls, w in zip(present_classes, cw_arr)})
    if len(present_classes) < n_classes:
        missing = sorted(set(range(n_classes)) - set(present_classes.tolist()))
        print(f"    ⚠️  Missing classes in final train split (weight=1.0): {missing}")

    final_model = build_model(n_classes, arch, input_shape=X_tr.shape[1:])
    final_model.compile(optimizer=Adam(LEARNING_RATE),
                        loss="categorical_crossentropy",
                        metrics=["categorical_accuracy"])
    final_model.summary()

    history = final_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=get_callbacks(artifacts["model_h5"]),
        verbose=1,
    )

    # Evaluation
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

    y_pred = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(f"\n📋  Per-class report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        target_names=actions,
        digits=3,
        zero_division=0,
    ))

    # Save artifacts
    print(f"\n{'='*65}")
    print(f"💾  SAVING ARTIFACTS")
    print(f"{'='*65}\n")

    final_model.save(artifacts["model_h5"])
    print(f"    ✅  Keras model : {artifacts['model_h5']}")

    if final_scaler is not None:
        joblib.dump(final_scaler, artifacts["scaler_pkl"])
        print(f"    ✅  Scaler pkl  : {artifacts['scaler_pkl']}")
    else:
        with open(artifacts["scaler_pkl"], "w", encoding="utf-8") as f:
            f.write("No scaler used for this architecture.\n")
        print(f"    ✅  Scaler pkl  : {artifacts['scaler_pkl']} (no-op)")

    export_scaler_json(final_scaler, artifacts["scaler_json"])
    try:
        export_tflite(final_model, artifacts["model_tflite"])
    except Exception as exc:
        print(f"    ⚠️  TFLite export failed for {arch}: {exc}")
    save_plots(history, fold_results, arch, artifacts["history_png"])

    print(f"\n{'='*65}")
    print(f"✅  DONE")
    print(f"{'='*65}")
    print(f"\n    Artifacts generated:")
    for f in [
        artifacts["model_tflite"],
        artifacts["model_h5"],
        artifacts["scaler_pkl"],
        artifacts["scaler_json"],
        artifacts["labels_json"],
        artifacts["history_png"],
    ]:
        size = f"{os.path.getsize(f)/1024:.0f} KB" if os.path.exists(f) else "?"
        print(f"      {f:<30s} {size}")

    print(f"\n▶   Next step: python lip_test.py --arch {arch}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lip Reading Model")
    parser.add_argument("--data", default=DATA_DIR, help="Dataset folder")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--arch", default="lstm", help="Model architecture: lstm, bilstm, bilstem, 3dcnn")
    args = parser.parse_args()
    main(data_dir=args.data, epochs=args.epochs, arch=args.arch)