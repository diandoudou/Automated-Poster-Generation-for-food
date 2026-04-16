from pathlib import Path
from agents.layout_planner import LayoutPlannerAgent


def test_interface():
    print("=" * 70)
    print("Layout Planner Interface and Pipeline Validation Test")
    print("=" * 70)

    # Input interface validation: simulate Concept Expander output (brief)
    print("\n[1/5] Testing input interface: Concept Expander brief")
    print("-" * 70)

    brief = {
        "theme": "Seafood Noodle",
        "style": "authentic Asian cuisine",
        "tone": "warm",
        "emotion": "satisfying umami",
        "tagline": "Fresh from the Ocean",  # Although no longer displayed, keep it in the interface
        "title": "Seafood Noodle",
        "description": "Rich seafood broth with fresh shrimp, scallops, and squid, paired with springy noodles. A taste of the ocean in every bowl.",
        "selling_points": [
            "Fresh Daily Seafood Selection",
            "Rich Umami Broth",
            "Handmade Springy Noodles",
        ],
        "cta": "Order Now",
        "badge": "Chef's Special",
    }

    print(f"Brief format: Dict[str, Any]")
    print(f"Brief fields: {list(brief.keys())}")
    print(f"Theme: {brief['theme']}")
    print(f"Style: {brief['style']}")
    print(f"Selling points: {len(brief['selling_points'])} items")

    # Input interface validation: simulate Visual Designer output
    print("\n[2/5] Testing input interface: Visual Designer visual_output")
    print("-" * 70)

    visual_output = {
        "image_path": "test_images/sample.png"  # Required field
    }

    image_path = Path(visual_output["image_path"])
    print(f"Visual output format: Dict[str, Any]")
    print(f"Required field: image_path")
    print(f"Image path: {visual_output['image_path']}")
    print(f"Image exists: {image_path.exists()}")

    if not image_path.exists():
        print("[WARNING] Image file does not exist, but interface test continues...")

    # Layout Planner processing
    print("\n[3/5] Executing Layout Planner")
    print("-" * 70)

    try:
        agent = LayoutPlannerAgent()
        print("[OK] LayoutPlannerAgent initialized successfully")

        result = agent.run(brief, visual_output)
        print("[OK] LayoutPlannerAgent.run() executed successfully")

    except Exception as e:
        print(f"[ERROR] Execution failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Output interface validation
    print("\n[4/5] Validating output interface format")
    print("-" * 70)

    # Check required output fields
    required_output_fields = ["poster_image_path"]
    optional_output_fields = ["template", "palette", "typography", "guidelines", "storyline"]

    print(f"Output format: Dict[str, Any]")
    print(f"Returned fields: {list(result.keys())}")

    # Validate required fields
    missing_fields = []
    for field in required_output_fields:
        if field not in result:
            missing_fields.append(field)

    if missing_fields:
        print(f"[ERROR] Missing required fields: {missing_fields}")
        return False
    else:
        print(f"[OK] All required fields present: {required_output_fields}")

    # Validate optional fields
    present_optional = [f for f in optional_output_fields if f in result]
    print(f"[OK] Optional fields present: {present_optional}")

    # Validate output file
    print("\n[5/5] Validating output file")
    print("-" * 70)

    poster_path = Path(result["poster_image_path"])
    print(f"Poster path: {result['poster_image_path']}")
    print(f"File exists: {poster_path.exists()}")
    print(f"File size: {poster_path.stat().st_size / 1024:.2f} KB" if poster_path.exists() else "N/A")

    if not poster_path.exists():
        print("[ERROR] Output file does not exist")
        return False
    else:
        print("[OK] Output file exists and is valid")

    # Summary
    print("\n" + "=" * 70)
    print("Interface and Pipeline Validation Results")
    print("=" * 70)
    print("[OK] Input interface format correct")
    print("  - brief (Concept Expander output): Dict[str, Any]")
    print("  - visual_output (Visual Designer output): Dict[str, Any] with 'image_path'")
    print("[OK] Output interface format correct")
    print("  - Return: Dict[str, Any] with 'poster_image_path' (required)")
    print(f"[OK] Poster generated successfully: {result['poster_image_path']}")
    print("[OK] Pipeline complete: Concept Expander -> Visual Designer -> Layout Planner -> Poster")
    print("=" * 70)

    # Print detailed output info
    print("\nOutput details:")
    print(f"  - Poster path: {result['poster_image_path']}")
    print(f"  - Template used: {result.get('template', 'N/A')}")
    print(f"  - Color palette: {result.get('palette', {})}")
    print(f"  - Storyline: {result.get('storyline', 'N/A')}")
    return True

if __name__ == "__main__":
    success = test_interface()
    if success:
        print("\n[SUCCESS] All interface tests passed!")
    else:
        print("\n[FAILED] Interface test failed, please check error messages")
