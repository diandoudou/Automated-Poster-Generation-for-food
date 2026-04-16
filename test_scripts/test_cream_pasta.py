from agents.layout_planner import LayoutPlannerAgent

if __name__ == "__main__":
    # Simulated Visual Designer output - using prepared transparent background image
    visual_output = {
        "image_path": "test_images/sample.png"
    }

    state = {
        "expanded_concept": """{
            "theme": "Seafood Noodle",
            "style": "authentic Asian cuisine",
            "tone": "warm",
            "emotion": "satisfying umami",
            "description": "Rich seafood broth with fresh shrimp, scallops, and squid",
            "tagline": "Fresh from the Ocean"
        }""",
        "visual_output": visual_output
    }

    agent = LayoutPlannerAgent()
    out = agent.run(state)

    print("\n" + "=" * 60)
    print("[Success] Poster generated!")
    print("=" * 60)
    print(f"Poster path: {out['poster_image_path']}")
    print(f"Template used: {out['template']}")
    print(f"Color palette: {out['palette']}")
    print(f"Storyline: {out.get('storyline', 'N/A')}")
    print("=" * 60)

