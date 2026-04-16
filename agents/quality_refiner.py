from __future__ import annotations

import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from langgraph.graph import END
from tools.testing_decorator import measure_time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# 10 food aesthetic attribute names (for model structure)
ATTRIBUTE_NAMES = [
    "brightness", "saturation", "contrast", "clarity", "warm", "colourfulness",
    "size_difference", "color_difference", "texture_difference", "segment_colorfulness"
]

# Diagnostic Thresholds and Direct Operational Advice
DIAGNOSTIC_ATTRIBUTES = {
    "clarity_score": {  # Clarity: Most significant factor distinguishing high/low scores
        "P5": 0.0655, "P95": 0.5833,
        "Optimal_Target": 0.2790,
        "Diagnosis_Low": "Severe Clarity Deficiency",
        "Diagnosis_High": "Excessive Clarity (Sharpening Artifacts)",
        # Direct operational advice
        "Advice_Low": "Please **INCREASE CLARITY/SHARPNESS**. The image may be out of focus or too soft. Re-focus or moderately enhance sharpness in post-processing.",
        "Advice_High": "Please **DECREASE CLARITY/SHARPNESS**. The current level of sharpness is likely too high, causing unnatural artifacts. Reduce detail enhancement or sharpening."
    },
    "saturation_score": {  # Saturation: Affects visual appeal
        "P5": 0.5329, "P95": 0.6955,
        "Optimal_Target": 0.6161,
        "Diagnosis_Low": "Saturation Deficiency",
        "Diagnosis_High": "Excessive Saturation",
        # Direct operational advice
        "Advice_Low": "Please **INCREASE COLOR SATURATION**. The colors appear dull and unappetizing. Moderately increase saturation in post-processing.",
        "Advice_High": "Please **DECREASE COLOR SATURATION**. The colors are too vivid and potentially distorted. Slightly reduce saturation in post-processing."
    },
    "size_difference_score": {  # Size Difference: Affects composition and complexity
        "P5": 0.0007, "P95": 0.0259,
        "Optimal_Target": 0.0083,
        "Diagnosis_Low": "Overly Simple Composition",
        "Diagnosis_High": "Overly Complex/Distracting Elements",
        # Direct operational advice
        "Advice_Low": "Please **MODERATELY INCREASE COMPOSITIONAL COMPLEXITY**. The image may be too monotonous. Consider adding supporting elements or adjusting foreground/background contrast.",
        "Advice_High": "Please **SIMPLIFY THE COMPOSITION**. The elements are too complex or have excessively large size differences. Reduce background clutter to keep focus on the food."
    }
}

# Quality Refiner Agent (LangGraph Compatible)
class QualityRefinerAgent:
    """
    Quality Refiner Agent for evaluating poster aesthetic quality.
    Compatible with LangGraph workflow - receives and returns state dictionary.
    """
    def __init__(
            self,
            model_path: str = "../models/best_model.weights.h5",
            aesthetic_threshold: float = 0.70,
            max_iterations: int = 3,
            output_dir: str = "../outputs/quality_reports",
    ):
        """
        Initialize the Quality Refiner Agent.
        Args:
            model_path: Path to trained model weights (.weights.h5)
            aesthetic_threshold: Score threshold (0-1) for qualification
            max_iterations: Maximum refinement iterations
            output_dir: Directory to save quality reports
        """
        self.model_path = Path(model_path)
        self.aesthetic_threshold = aesthetic_threshold
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None  # Lazy loading

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")

    # LangGraph Interface (Main Entry Point)
    @measure_time("QualityRefiner")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph-compatible run method.
        Receives state from LayoutPlanner, evaluates quality, returns updated state.
        Args:
            state: Current workflow state containing layout_output
        Returns:
            Updated state with quality evaluation and next_node decision
        """
        print("\n" + "=" * 60)
        print("🔍 Quality Refiner Agent - Starting")
        print("=" * 60)

        # Extract poster image path from state
        layout_output = state.get("layout_output", {})
        poster_path = layout_output.get("layout_path")

        if not poster_path or not Path(poster_path).exists():
            raise ValueError(f"Poster image not found in state: {poster_path}")

        # Load model (lazy)
        if self.model is None:
            self._load_model()

        # Evaluate quality
        evaluation = self._evaluate_poster(poster_path)

        # Save report
        report_path = self._save_report(evaluation, state)
        suggestions = self._load_suggestions(report_path)

        # Determine next action
        iteration_count = state.get("iteration_count", 0)
        next_node = self._decide_next_node(
            evaluation["qualified"],
            iteration_count,
            self.max_iterations
        )

        # Print summary
        self._print_summary(evaluation, next_node, iteration_count)

        evaluation_update = {
            "quality_evaluation": evaluation,
            "quality_report_path": str(report_path),
            "next_node": next_node,
            "iteration_count": iteration_count + 1
        }

        if next_node == "VisualDesigner":
            evaluation_update["quality_suggestions"] = suggestions
            print(f"Next node is VisualDesigner, suggestions: {evaluation_update.get('quality_suggestions')}")
        else:
            evaluation_update["quality_suggestions"] = []

        return evaluation_update

    def _load_suggestions(self, json_path: Path) -> List[str]:
        p = Path(json_path)
        if not p.exists():
            return []
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # safe extraction
        suggestions = data.get("diagnosis", {}).get("suggestions", [])
        return suggestions

    def _evaluate_poster(self, image_path: str | Path) -> Dict[str, Any]:
        """
        Evaluate poster aesthetic quality.
        Args:
            image_path: Path to poster image
        Returns:
            Dictionary with scores, diagnosis, and suggestions
        """
        print(f"📊 Evaluating: {image_path}")

        # Preprocess image
        img = self._load_and_preprocess(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Run inference
        preds = self.model.predict(np.expand_dims(img, 0), verbose=0)

        # Extract scores
        aesthetic_score_raw = float(preds[0].squeeze())
        scaled_score = aesthetic_score_raw * 100

        attribute_scores = {}
        for i, name in enumerate(ATTRIBUTE_NAMES):
            attribute_scores[f'{name}_score'] = float(preds[i + 1].squeeze())

        # Diagnose issues
        qualified = scaled_score >= (self.aesthetic_threshold * 100)
        diagnosis = self._diagnose_issues(attribute_scores, qualified)

        return {
            "image_path": str(image_path),
            "aesthetic_score": aesthetic_score_raw,
            "scaled_score": scaled_score,
            "qualified": qualified,
            "attribute_scores": attribute_scores,
            "diagnosis": diagnosis,
            "threshold": self.aesthetic_threshold * 100,
        }

    def _diagnose_issues(
            self,
            attribute_scores: Dict[str, float],
            qualified: bool
    ) -> Dict[str, Any]:
        """
        Diagnose quality issues and provide suggestions.
        Args:
            attribute_scores: Dictionary of attribute scores
            qualified: Whether the poster passed the threshold
        Returns:
            Diagnosis dictionary with issues and suggestions
        """
        issues = []
        suggestions = []

        if qualified:
            return {
                "has_issues": False,
                "issues": [],
                "suggestions": ["✅ No adjustments needed - quality is acceptable"]
            }

        # Check each diagnostic attribute
        for attr_key, diag_data in DIAGNOSTIC_ATTRIBUTES.items():
            attr_score = attribute_scores.get(attr_key, 0.0)

            # Below P5 (deficiency)
            if attr_score < diag_data['P5']:
                issues.append({
                    "attribute": attr_key,
                    "type": "deficiency",
                    "severity": "high",
                    "current_value": attr_score,
                    "target_range": [diag_data['P5'], diag_data['P95']],
                    "diagnosis": diag_data['Diagnosis_Low'],
                })
                suggestions.append(diag_data['Advice_Low'])

            # Above P95 (excess)
            elif attr_score > diag_data['P95']:
                issues.append({
                    "attribute": attr_key,
                    "type": "excess",
                    "severity": "high",
                    "current_value": attr_score,
                    "target_range": [diag_data['P5'], diag_data['P95']],
                    "diagnosis": diag_data['Diagnosis_High'],
                })
                suggestions.append(diag_data['Advice_High'])

        # If no specific issues but still failed
        if not issues:
            suggestions.append(
                "All critical attributes are within range, but overall score is low. "
                "Consider improving composition, lighting, or other attributes."
            )

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "suggestions": suggestions,
        }

    # Decision Logic
    def _decide_next_node(
            self,
            qualified: bool,
            iteration_count: int,
            max_iterations: int
    ) -> str:
        """
        Decide whether to END or loop back to LayoutPlanner.
        Args:
            qualified: Whether the poster passed quality check
            iteration_count: Current iteration number
            max_iterations: Maximum allowed iterations
        Returns:
            Next node name ("END" or "LayoutPlanner")
        """
        if qualified:
            return END

        if iteration_count >= max_iterations:
            print(f"⚠️  Max iterations ({max_iterations}) reached. Stopping refinement.")
            return END

        return "VisualDesigner"

    def _load_model(self):
        """Load the trained aesthetic evaluation model."""
        print(f"📦 Loading quality model: {self.model_path}")

        self.model = self._build_multi_output_model()
        self.model.build((None, 224, 224, 3))

        try:
            self.model.load_weights(str(self.model_path))
            print("✅ Model loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {self.model_path}. "
                f"Ensure the model structure matches the weights file. Error: {e}"
            )

    @staticmethod
    def _build_multi_output_model() -> Model:
        """Build the NimaMobileNetV2 Multi-Output model architecture."""
        inputs = Input(shape=(224, 224, 3))

        base = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            pooling='avg',
            weights=None,
            # name="MobileNetV2_Base"
        )

        x = base(inputs)
        x = Dropout(0.25, name="dropout_1")(x)
        shared = Dense(10, activation="relu", name="shared_dense_relu")(x)

        outputs = []

        # Aesthetic score output
        outputs.append(Dense(1, activation="sigmoid", name="aesthetic_score")(shared))

        # Attribute outputs
        for name in ATTRIBUTE_NAMES:
            outputs.append(Dense(1, activation="sigmoid", name=f"{name}_output")(shared))

        return Model(inputs, outputs)

    # Image Processing
    @staticmethod
    def _load_and_preprocess(img_path: str | Path) -> np.ndarray | None:
        """Load and preprocess image for model input."""
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            return None

    def _save_report(
            self,
            evaluation: Dict[str, Any],
            state: Dict[str, Any]
    ) -> Path:
        """Save quality evaluation report to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iteration = state.get("iteration_count", 0)

        report_dir = self.output_dir / f"iteration_{iteration}"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / f"quality_report_{timestamp}.json"

        # Build structured report
        report = {
            "metadata": {
                "report_type": "Quality Evaluation Report",
                "timestamp": timestamp,
                "timestamp_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "iteration": iteration,
                "user": "N/A"
            },
            "input": {
                "image_path": evaluation['image_path'],
                "threshold": round(evaluation['threshold'], 2)
            },
            "results": {
                "aesthetic_score": round(evaluation['scaled_score'], 2),
                "aesthetic_score_raw": round(evaluation['aesthetic_score'], 4),
                "qualified": evaluation['qualified'],
                "status": "QUALIFIED" if evaluation['qualified'] else "NOT QUALIFIED"
            },
            "diagnosis": {
                "has_issues": evaluation['diagnosis']['has_issues'],
                "issues": [
                    {
                        "diagnosis": issue['diagnosis'],
                        "attribute": issue['attribute'],
                        "type": issue['type'],
                        "severity": issue['severity'],
                        "current_value": round(issue['current_value'], 4),
                        "target_range": {
                            "min": round(issue['target_range'][0], 4),
                            "max": round(issue['target_range'][1], 4)
                        }
                    }
                    for issue in evaluation['diagnosis']['issues']
                ] if evaluation['diagnosis']['has_issues'] else [],
                "suggestions": evaluation['diagnosis']['suggestions']
            },
            "attribute_scores": {
                attr: round(score, 4)
                for attr, score in evaluation['attribute_scores'].items()
            }
        }

        # Save as JSON with pretty formatting
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 Report saved: {report_path}")
        return report_path

    def _print_summary(
            self,
            evaluation: Dict[str, Any],
            next_node: str,
            iteration: int
    ):
        """Print evaluation summary to console."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"Score: {evaluation['scaled_score']:.2f}/100")
        print(f"Status: {'✅ QUALIFIED' if evaluation['qualified'] else '❌ NOT QUALIFIED'}")
        print(f"Iteration: {iteration + 1}/{self.max_iterations}")

        if evaluation['diagnosis']['has_issues']:
            print(f"\nIssues Found: {len(evaluation['diagnosis']['issues'])}")
            for issue in evaluation['diagnosis']['issues']:
                print(f"  - {issue['diagnosis']}")

        print(f"\nNext Action: {next_node}")
        print("=" * 60)

# Standalone Test
def test_quality_refiner():
    """Test the agent standalone"""
    # Mock state from LayoutPlanner
    mock_state = {
        "layout_output": {
            "layout_path": "../outputs/posters/poster_20251119_234116.png"
        },
        "iteration_count": 0,
        "max_iterations": 3,
    }

    # Create agent
    agent = QualityRefinerAgent(
        model_path="../models/best_model.weights.h5",
        aesthetic_threshold=0.70,
        max_iterations=3,
    )

    # Run
    result_state = agent.run(mock_state)

    # Check output
    print("\n" + "=" * 60)
    print("Test Result:")
    print("=" * 60)
    print(f"Score: {result_state['quality_evaluation']['scaled_score']:.2f}")
    print(f"Qualified: {result_state['quality_evaluation']['qualified']}")
    print(f"Next Node: {result_state['next_node']}")
    print(f"Report: {result_state['quality_report_path']}")

if __name__ == "__main__":
    test_quality_refiner()
