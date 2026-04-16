import os
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# 10 food aesthetic attribute names (for model structure)
ATTRIBUTE_NAMES = [
    "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
    "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
]

# --- Diagnostic Thresholds and Direct Operational Advice ---
# Uses P5/P95 values derived from high-scoring group (G_High).
DIAGNOSTIC_ATTRIBUTES = {
    "clarity_score": {  # Clarity: High-Score Tendency (Higher is better)
        "P5": 0.107, "P95": 0.656,
        "Optimal_Target": 0.344,
        "Diagnosis_Low": "Severe Clarity Deficiency (Image too soft or blurred)",
        "Diagnosis_High": "Excessive Clarity (Sharpening Artifacts)",
        # High-score tendency (mu_High > mu_Low)
        "Advice_Low": "Please **INCREASE CLARITY/SHARPNESS**. The image may be out of focus or too soft. **Action Advice:** In editing software, moderately increase **Clarity/Texture** or use the **Sharpening** tool. Check focus and reshoot if necessary.",
        "Advice_High": "Please **DECREASE CLARITY/SHARPNESS**. The current level of sharpness is likely too high, causing unnatural artifacts. **Action Advice:** In editing software, moderately reduce the **Clarity/Texture** parameter, or decrease the amount of **Sharpening** applied."
    },
    "saturation_score": {  # Saturation: High-Score Tendency (Higher is better)
        "P5": 0.547, "P95": 0.705,
        "Optimal_Target": 0.626,
        "Diagnosis_Low": "Saturation Deficiency (Dull Colors)",
        "Diagnosis_High": "Excessive Saturation (Distorted Colors)",
        # High-score tendency (mu_High > mu_Low)
        "Advice_Low": "Please **INCREASE COLOR SATURATION**. The colors appear dull and unappetizing. **Action Advice:** In editing software, moderately increase the **Saturation** or **Vibrance** parameter.",
        "Advice_High": "Please **DECREASE COLOR SATURATION**. The colors are too vivid and potentially distorted. **Action Advice:** In editing software, moderately reduce the **Saturation** parameter. If you only want to adjust less vibrant colors, try adjusting **Vibrance**."
    },
    "segment_colorfulness_score": {  # Segment Colourfulness: Low-Score Tendency (Lower is better)
        "P5": 0.846, "P95": 0.989,
        "Optimal_Target": 0.940,
        "Diagnosis_Low": "Low Segment Colourfulness (Usually good, but too extreme)",
        "Diagnosis_High": "Excessive Segment Colourfulness (Tendency of low-score images)",
        # Low-score tendency (mu_High < mu_Low)
        "Advice_Low": "Score is excessively low. Please **MODERATELY INCREASE SEGMENT COLOURFULNESS** to reach the target range. **Action Advice:** Increase overall **Saturation** or **Vibrance**. Also, check for large grey or brown areas in the image and enhance their color moderately.",
        "Advice_High": "Score is too high, approaching the G_Low mean. Please **DECREASE SEGMENT COLOURFULNESS** to move towards the optimal lower mean ($\mu_{High}=0.940$). **Action Advice:** Moderately reduce the overall **Saturation** or **Vibrance** parameters in your editing software."
    }
}

# Helper Function: Model Structure (Must match training script)
def build_multi_output_model():
    """Defines the Functional API structure for NimaMobileNetV2 Multi-Output."""
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

    # All outputs use Sigmoid (0-1)
    outputs.append(Dense(1, activation="sigmoid", name="aesthetic_score")(shared))

    for name in ATTRIBUTE_NAMES:
        outputs.append(Dense(1, activation="sigmoid", name=f"{name}_output")(shared))
    return Model(inputs, outputs)

# Helper Function: Image Preprocessing
def load_and_preprocess(img_path):
    """Loads, resizes, and normalizes the image to [0, 1]."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

# Main Function: Scoring and Diagnosis
def run_suggestion(image_path, model_path, aesthetic_threshold=0.70):
    print("\n==============================")
    print(" 1. MODEL LOADING")
    print("==============================")

    model = build_multi_output_model()
    model.build((None, 224, 224, 3))

    try:
        model.load_weights(model_path)
    except Exception as e:
        print(
            f"ERROR: Weight loading failed. Ensure model structure matches the weights file {model_path}. Error: {e}")
        return

    print(f"Model weights {os.path.basename(model_path)} loaded successfully.")

    print("\n==============================")
    print(" 2. IMAGE INFERENCE & SCORING")
    print("==============================")

    img = load_and_preprocess(image_path)
    if img is None:
        return

    # Run inference to get 1 (aesthetic score) + 10 (attribute scores) outputs
    preds = model.predict(np.expand_dims(img, 0), verbose=0)

    aesthetic_score_raw = float(preds[0].squeeze())
    scaled_score = aesthetic_score_raw * 100

    # Build a dictionary of attribute scores
    attribute_scores = {}
    for i, name in enumerate(ATTRIBUTE_NAMES):
        attribute_scores[f'{name}_score'] = float(preds[i + 1].squeeze())

    # --- Print Scores ---
    print(f"Aesthetic Score (0-100): {scaled_score:.2f}")
    print("\n--- Key Attribute Scores (0-1) ---")

    key_attributes_to_print = ["clarity_score", "saturation_score", "segment_colorfulness_score"]

    for attr_key in key_attributes_to_print:
        print(f"- {attr_key.replace('_score', '').title()}: {attribute_scores.get(attr_key, 0.0):.4f}")

    print("\n==============================")
    print(" 3. DIAGNOSIS REPORT")
    print("==============================")

    if scaled_score >= aesthetic_threshold * 100:
        print(
            f"QUALIFIED! Aesthetic Score ({scaled_score:.2f}) exceeds the threshold ({aesthetic_threshold * 100:.2f}). No adjustment needed.")
    else:
        print(
            f"NOT QUALIFIED. Aesthetic Score is below the threshold ({aesthetic_threshold * 100:.2f}). The following adjustments are REQUIRED:")

        has_suggestion = False

        # Iterate through key attributes for diagnosis
        for attr_key, diag_data in DIAGNOSTIC_ATTRIBUTES.items():
            attr_score = attribute_scores.get(attr_key, 0.0)

            # --- Diagnosis Logic ---
            # Deficiency Diagnosis (Below P5)
            if attr_score < diag_data['P5']:
                print(f"\n[DEFECT: {diag_data['Diagnosis_Low']}]")
                print(
                    f"- Current {attr_key}: {attr_score:.4f} (Target Range [{diag_data['P5']:.4f}, {diag_data['P95']:.4f}])")
                print(f"- OPTIMIZATION ADVICE: {diag_data['Advice_Low']}")
                has_suggestion = True

            # Excess Diagnosis (Above P95)
            elif attr_score > diag_data['P95']:
                print(f"\n[DEFECT: {diag_data['Diagnosis_High']}]")
                print(
                    f"- Current {attr_key}: {attr_score:.4f} (Target Range [{diag_data['P5']:.4f}, {diag_data['P95']:.4f}])")
                print(f"- OPTIMIZATION ADVICE: {diag_data['Advice_High']}")
                has_suggestion = True

            # Within acceptable range
            else:
                pass

        if not has_suggestion:
            print(
                "\nAll three most critical attributes are within the acceptable range, but the overall score is low.")
            print("To achieve a higher score, we suggest a general refinement:")

            # General advice with actionable guidance
            for attr_key, diag_data in DIAGNOSTIC_ATTRIBUTES.items():
                print(f"\n- {attr_key.replace('_score', '').title()}:")
                print(f"  > Target the optimal score of {diag_data['Optimal_Target']:.4f}.")

                # Provide action based on attribute tendency
                if attr_key in ["clarity_score", "saturation_score"]:  # High-Score Tendency
                    print(
                        f"  > Action Advice: Try slightly increasing the image's {attr_key.replace('_score', '').upper()} parameter until the score improves.")
                elif attr_key == "segment_colorfulness_score":  # Low-Score Tendency
                    print(
                        f"  > Action Advice: Try slightly decreasing the image's {attr_key.replace('_score', '').upper()} parameter until the score improves.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Food Aesthetic Score & Diagnosis Tool.')
    parser.add_argument("--image", required=True, help="Path to the input image file (e.g., poster.jpg)")
    parser.add_argument("--model", type=str, default='output/checkpoints/best_model.weights.h5',
                        help="Path to the model weights file (.weights.h5)")
    parser.add_argument("--threshold", type=float, default=70,
                        help="Aesthetic score threshold for qualification (0-100). Default is 70.")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image file not found: {args.image}")
    elif not os.path.exists(args.model):
        print(f"Error: Model weights file not found: {args.model}. Please complete training first.")
    else:
        # Convert threshold from 0-100 to 0-1 for internal use
        internal_threshold = args.threshold / 100.0
        run_suggestion(args.image, args.model, internal_threshold)