from agents.layout_planner import LayoutPlannerAgent

if __name__ == "__main__":
    brief = {
        "theme": "Velvet Cream Cake",
        "style": "retro luxury patisserie",
        "tone": "warm",
        "emotion": "cozy nostalgic",
        "tagline": "Limited Time Offer",
        "description": "Silky cream blended with Madagascar vanilla, layered on toasted almond crumble.",
        "selling_points": [
            "32% low sugar whipped cream",
            "Fresh berries hand-picked every morning",
            "Complimentary artisan tea pairing",
        ],
        "cta": "立即预约品尝",
    }

    visual_output = {
        "image_path": "test_images/sample.png"
    }

    agent = LayoutPlannerAgent()
    out = agent.run(brief, visual_output)

    print("生成的海报路径：", out["poster_image_path"], out["template"], out["palette"])
