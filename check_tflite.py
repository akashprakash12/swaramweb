import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_labels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Invalid labels file: {path}")
    return [str(x) for x in labels]


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        scaler = json.load(f)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    if mean.shape != scale.shape:
        raise ValueError("Scaler mean/scale shape mismatch")
    return mean, scale


def prepare_input(
    sample: Path | None,
    seq_len: int,
    n_features: int,
    mean: np.ndarray | None,
    scale: np.ndarray | None,
) -> np.ndarray:
    if sample is None:
        data = np.zeros((seq_len, n_features), dtype=np.float32)
    else:
        arr = np.load(sample)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape != (seq_len, n_features):
            raise ValueError(
                f"Sample shape must be ({seq_len}, {n_features}) or (1,{seq_len},{n_features}), got {arr.shape}"
            )
        data = arr.astype(np.float32)

    if mean is not None and scale is not None:
        flat = data.reshape(-1, n_features)
        flat = (flat - mean) / scale
        data = flat.reshape(seq_len, n_features)

    return np.expand_dims(data, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Quick TFLite inference checker")
    parser.add_argument("--tflite", default="model.tflite", help="Path to model.tflite")
    parser.add_argument("--labels", default="labels.json", help="Path to labels.json")
    parser.add_argument("--scaler", default="scaler.json", help="Path to scaler.json")
    parser.add_argument("--sample", default="", help="Optional .npy sample with shape (30,225)")
    parser.add_argument("--topk", type=int, default=3, help="Top-K predictions to print")
    args = parser.parse_args()

    tflite_path = Path(args.tflite)
    labels_path = Path(args.labels)
    scaler_path = Path(args.scaler)
    sample_path = Path(args.sample) if args.sample else None

    if not tflite_path.exists():
        raise FileNotFoundError(f"Missing tflite model: {tflite_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    labels = load_labels(labels_path)
    mean = scale = None
    if scaler_path.exists():
        mean, scale = load_scaler(scaler_path)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    try:
        interpreter.allocate_tensors()
    except RuntimeError as exc:
        msg = str(exc)
        print("❌ TFLite allocate_tensors failed")
        print(f"   Reason: {msg}")
        if "Select TensorFlow op" in msg or "Flex" in msg:
            print("\n💡 This model requires Select TF Ops (Flex delegate).")
            print("   Android fix:")
            print("   1) Use a development build (not Expo Go)")
            print("   2) Add dependency: org.tensorflow:tensorflow-lite-select-tf-ops")
            print("   3) Use a TFLite RN package/native module that supports Select TF Ops")
        raise

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    _, seq_len, n_features = input_details["shape"]
    model_input = prepare_input(sample_path, int(seq_len), int(n_features), mean, scale)

    interpreter.set_tensor(input_details["index"], model_input)
    try:
        interpreter.invoke()
    except RuntimeError as exc:
        print("❌ TFLite invoke failed")
        print(f"   Reason: {exc}")
        raise
    output = interpreter.get_tensor(output_details["index"])[0].astype(np.float32)

    if output.ndim != 1:
        raise RuntimeError(f"Unexpected output shape: {output.shape}")

    if len(labels) != output.shape[0]:
        print(f"⚠️  Labels count ({len(labels)}) != output classes ({output.shape[0]})")

    order = np.argsort(output)[::-1]
    topk = max(1, min(args.topk, len(order)))

    print("✅ TFLite inference ran successfully")
    print(f"   Input shape : {tuple(input_details['shape'])}")
    print(f"   Output shape: {tuple(output_details['shape'])}")
    print(f"   Classes     : {output.shape[0]}")
    print(f"   Sample used : {sample_path if sample_path else 'zeros (smoke test)'}")
    print("\nTop predictions:")

    for i in range(topk):
        idx = int(order[i])
        name = labels[idx] if idx < len(labels) else f"class_{idx}"
        print(f"  {i+1}. {name:<20} {output[idx]*100:6.2f}%")


if __name__ == "__main__":
    main()
