import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf

from normalization import normalize_sequence

SEQUENCE_LEN = 30
N_FEATURES = 225


def load_labels(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        labels = json.load(f)
    return [str(x) for x in labels]


def load_scaler_json(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mean = np.asarray(data["mean"], dtype=np.float32)
    scale = np.asarray(data["scale"], dtype=np.float32)
    if mean.shape != (N_FEATURES,) or scale.shape != (N_FEATURES,):
        raise ValueError(f"Invalid scaler shape. mean={mean.shape}, scale={scale.shape}, expected {(N_FEATURES,)}")
    return mean, scale


def preprocess(seq: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if seq.shape != (SEQUENCE_LEN, N_FEATURES):
        raise ValueError(f"Expected sample shape {(SEQUENCE_LEN, N_FEATURES)}, got {seq.shape}")
    seq = normalize_sequence(seq.astype(np.float32))
    flat = seq.reshape(-1, N_FEATURES)
    flat = (flat - mean) / scale
    return flat.reshape(SEQUENCE_LEN, N_FEATURES)


def main():
    parser = argparse.ArgumentParser(description="Debug model predictions on dataset samples")
    parser.add_argument("--model", default="model.h5")
    parser.add_argument("--scaler", default="scaler.json")
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--max-per-class", type=int, default=30, help="max files to evaluate per dataset folder")
    parser.add_argument("--synthetic-if-empty", action="store_true",
                        help="run one synthetic zero-input prediction when dataset has no samples")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Missing model: {args.model}")
    if not os.path.exists(args.scaler):
        raise FileNotFoundError(f"Missing scaler: {args.scaler}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Missing labels: {args.labels}")

    labels = load_labels(args.labels)
    mean, scale = load_scaler_json(args.scaler)

    model = tf.keras.models.load_model(args.model)
    n_out = int(model.output_shape[-1])

    print("=" * 72)
    print("MODEL DEBUG REPORT")
    print("=" * 72)
    print(f"Model output classes: {n_out}")
    print(f"Labels count        : {len(labels)}")
    print(f"Labels              : {labels}")
    if len(labels) != n_out:
        print("\n⚠️  LABEL MISMATCH: model class count and labels.json do not match.")
        print("    Fix labels.json to exact training class order from your friend.")

    folders = sorted([d for d in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, d))])
    if not folders:
        raise ValueError(f"No class folders found in {args.data}")

    folder_pred_counter: dict[str, Counter] = defaultdict(Counter)
    folder_conf: dict[str, list[float]] = defaultdict(list)
    global_counter = Counter()

    for folder in folders:
        class_dir = os.path.join(args.data, folder)
        files = sorted([f for f in os.listdir(class_dir) if f.endswith(".npy")])[: args.max_per_class]
        for fname in files:
            path = os.path.join(class_dir, fname)
            try:
                arr = np.load(path)
                inp = preprocess(arr, mean, scale)
            except Exception:
                continue

            probs = model.predict(np.expand_dims(inp, axis=0), verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

            folder_pred_counter[folder][pred_idx] += 1
            folder_conf[folder].append(conf)
            global_counter[pred_idx] += 1

    if sum(global_counter.values()) == 0:
        print("\n⚠️  No valid .npy samples found in dataset folders.")
        if args.synthetic_if_empty:
            print("\nRunning synthetic prediction on zero input...")
            dummy = np.zeros((1, SEQUENCE_LEN, N_FEATURES), dtype=np.float32)
            probs = model.predict(dummy, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            print(f"Top class: {label} (index={idx}, conf={float(probs[idx])*100:.2f}%)")
            print("Raw probs:", probs.tolist())
        print("\nNext:")
        print("- Collect samples first: python step1_collect.py")
        print("- Then run this debugger again.")
        return

    print("\nPer-folder predicted class distribution:")
    print("-" * 72)
    for folder in folders:
        total = sum(folder_pred_counter[folder].values())
        if total == 0:
            print(f"{folder:<16} -> no valid samples")
            continue

        top_idx, top_count = folder_pred_counter[folder].most_common(1)[0]
        top_label = labels[top_idx] if top_idx < len(labels) else f"class_{top_idx}"
        avg_conf = np.mean(folder_conf[folder]) if folder_conf[folder] else 0.0

        dist_text = ", ".join(
            [
                f"{(labels[i] if i < len(labels) else f'class_{i}')}:{c}"
                for i, c in folder_pred_counter[folder].most_common()
            ]
        )
        print(
            f"{folder:<16} -> top={top_label:<12} ({top_count}/{total})  avg_conf={avg_conf*100:5.1f}%"
        )
        print(f"{'':<16}    {dist_text}")

    print("\nGlobal predicted distribution:")
    print("-" * 72)
    total_global = sum(global_counter.values())
    for idx, count in global_counter.most_common():
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        pct = (count / total_global * 100.0) if total_global else 0.0
        print(f"{label:<16}  {count:4d}  ({pct:5.1f}%)")

    print("\nTip:")
    print("- If one dataset folder mostly maps to one output index, use that mapping")
    print("  to rewrite labels.json in the exact model class order.")


if __name__ == "__main__":
    main()
