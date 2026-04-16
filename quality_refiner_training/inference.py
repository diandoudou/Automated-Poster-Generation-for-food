import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

ATTRIBUTE_NAMES = [
    "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
    "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
]


def build_multi_output_model():
    """
    Defines the Functional API architecture for NimaMobileNetV2MultiOutput,
    built on a MobileNetV2 backbone. The model produces multiple outputs with
    sigmoid activation, suitable for multi-label or normalized score predictions.
    """
    inputs = Input(shape=(224, 224, 3))
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling='avg',
        weights=None,
        name="MobileNetV2_Base"
    )
    x = base(inputs)
    x = Dropout(0.25, name="dropout_1")(x)
    shared = Dense(10, activation="relu", name="shared_dense_relu")(x)

    outputs = []
    outputs.append(Dense(1, activation="sigmoid", name="aesthetic_score")(shared))

    for name in ATTRIBUTE_NAMES:
        outputs.append(Dense(1, activation="sigmoid", name=f"{name}_output")(shared))
    return Model(inputs, outputs)


def load_and_preprocess(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        return None


def run_inference_and_grouping(model_path, val_dir):
    print("\n==============================")
    print(" 1. Loading model for inference")
    print("==============================")

    model = build_multi_output_model()
    model.build((None, 224, 224, 3))

    try:
        model.load_weights(model_path)
    except Exception as e:
        print(
            f"Error: Failed to load weights. Please ensure the model structure matches the weight file {model_path}. Details: {e}")
        return

    print(f"Model weights {os.path.basename(model_path)} loaded successfully.\n")

    print("==============================")
    print(" 2. Running Inference")
    print("==============================")

    rows = []
    val_path = Path(val_dir)
    img_files = [f.name for f in val_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

    if not img_files:
        print(f"Error: No image files found in validation directory {val_dir}.")
        return

    for img_name in tqdm(img_files, desc="Inference"):
        img = load_and_preprocess(os.path.join(val_dir, img_name))
        if img is None:
            continue
        preds = model.predict(np.expand_dims(img, 0), verbose=0)
        aesthetic_score = float(preds[0].squeeze())
        row = {
            "image_name": img_name,
            "aesthetic_score": aesthetic_score
        }

        for i, name in enumerate(ATTRIBUTE_NAMES):
            row[f'{name}_score'] = float(preds[i + 1].squeeze())
        rows.append(row)

    df_scores = pd.DataFrame(rows)
    N_SAMPLES = len(df_scores)
    print(f"\nInference completed. Total {N_SAMPLES} images.")

    print("\n==============================")
    print(" 3. Grouping Data (1:1 Median Split) and Saving CSV")
    print("==============================")

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    median_score = df_scores['aesthetic_score'].median()
    print(f"Computed median threshold (Median Score): {median_score:.4f}")

    # High-quality group (G_High): scores strictly greater than median
    df_high = df_scores[df_scores["aesthetic_score"] > median_score].copy()
    output_high_path = os.path.join(output_dir, 'G_high_scores.csv')
    df_high.to_csv(output_high_path, index=False)

    # Low-quality group (G_Low): scores strictly less than median
    df_low = df_scores[df_scores["aesthetic_score"] < median_score].copy()
    output_low_path = os.path.join(output_dir, 'G_low_scores.csv')
    df_low.to_csv(output_low_path, index=False)

    # Excluded median group (Score == median)
    df_median = df_scores[df_scores["aesthetic_score"] == median_score].copy()

    # Print grouping results
    print(
        f"\nGrouping strategy: Using median {median_score:.4f} as threshold, split into > Median and < Median groups.")
    print(f"High-quality group (Score > {median_score:.4f}): {len(df_high)} images, saved to: {output_high_path}")
    print(f"Low-quality group (Score < {median_score:.4f}): {len(df_low)} images, saved to: {output_low_path}")
    print(f"Excluded median group (Score = {median_score:.4f}): {len(df_median)} images")

    # Check totals
    total_grouped = len(df_high) + len(df_low) + len(df_median)
    print(
        f"Total images: {N_SAMPLES}, Grouped total: {total_grouped}. {'Match' if N_SAMPLES == total_grouped else 'Mismatch'}")
    print("\nProcessing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference and group data into high/low aesthetic groups.')
    parser.add_argument("--model", required=True,
                        help="Path to the best model weights file (e.g., output/checkpoints/best_model.weights.h5)")
    parser.add_argument("--val_dir", required=True, help="Path to the validation images folder (e.g., validation)")
    # Removed threshold_high and threshold_low argument definitions

    args = parser.parse_args()

    if not os.path.exists(args.model) or not os.path.isdir(args.val_dir):
        print("Error: Model file or image directory not found. Please check the paths.")
    else:
        # Updated function call, only passing model and val_dir
        run_inference_and_grouping(args.model, args.val_dir)
