from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from tools.testing_decorator import measure_time

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance

class VisualDesignerAgent:
    """
    Visual Designer Agent for generating poster main visuals using Stable Diffusion.
    Compatible with LangGraph workflow - receives and returns state dictionary.
    """
    def __init__(
            self,
            model_id: str = "runwayml/stable-diffusion-v1-5",
            output_dir: str = "../test_images",
            image_width: int = 512,
            image_height: int = 512,
    ):
        """
        Initialize the Visual Designer Agent.
        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save generated images
            image_width: Width of generated image
            image_height: Height of generated image
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_width = image_width
        self.image_height = image_height

        self.pipe = None  # Lazy loading
        self.device = self._get_device()

    # LangGraph Interface (Main Entry Point)
    @measure_time("VisualDesigner")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph-compatible run method.
        Receives state from ConceptExpander, generates visual, returns updated state.
        Args:
            state: Current workflow state containing expanded_concept
        Returns:
            Updated state with visual_output added
        """
        print("\n" + "=" * 60)
        print("🎨 Visual Designer Agent - Starting")
        print("=" * 60)

        # Extract data from state
        expanded_concept = state.get("expanded_concept")
        suggestions = state.get("quality_suggestions", [])
        iteration_count = state.get("iteration_count", 0)

        print(f"Iteration: {iteration_count}")
        print(f"Expanded concept type: {type(expanded_concept).__name__}")

        # Validate expanded_concept
        if expanded_concept is None:
            raise ValueError("Expanded concept not found in state")

        # Handle if it's a JSON string
        if isinstance(expanded_concept, str):
            print("⚠️  Expanded concept is a string, parsing JSON...")

            # Strip markdown code blocks if present
            expanded_concept = expanded_concept.strip()

            # Remove ```json and ``` markers
            if expanded_concept.startswith("```"):
                print("   Removing markdown code block markers...")
                # Remove opening ```json or ```
                lines = expanded_concept.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]  # Remove first line
                # Remove closing ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]  # Remove last line
                expanded_concept = '\n'.join(lines).strip()

            try:
                expanded_concept = json.loads(expanded_concept) # json format str -> json
                print("   ✅ Parsed successfully")
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parsing failed: {e}")
                print(f"   Content preview: {expanded_concept[:300]}...")
                raise ValueError(
                    f"Failed to parse expanded_concept: {e}\n"
                    f"Content: {expanded_concept[:200]}..."
                )

        # Validate it's not empty
        if not expanded_concept:
            raise ValueError("Expanded concept is empty")

        # Log suggestions
        if not suggestions:
            print("⚠️  No quality suggestions - first iteration")
        else:
            print(f"📋 Processing with {len(suggestions)} quality suggestions:")
            for i, s in enumerate(suggestions, 1):
                print(f"   {i}. {s}")

        # Prepare concept for visual generation
        if isinstance(expanded_concept, dict):
            concept_with_suggestions = expanded_concept.copy()
            concept_with_suggestions["quality_suggestions"] = suggestions
        else:
            concept_with_suggestions = {
                "content": str(expanded_concept),
                "quality_suggestions": suggestions
            }

        # Generate visual
        print("\n🎨 Generating visual with Stable Diffusion...")
        try:
            visual_result = self._generate_visual(concept_with_suggestions)
        except Exception as e:
            print(f"❌ Visual generation failed: {e}")
            raise

        # Validate output
        if not visual_result or "image_path" not in visual_result:
            raise ValueError("Visual generation failed: no image_path in result")

        image_path = visual_result["image_path"]

        # Verify file exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Generated visual not found: {image_path}")

        print(f"\n✅ Visual generated successfully!")
        print(f"   Image path: {image_path}")
        print(f"   File size: {Path(image_path).stat().st_size / 1024:.2f} KB")
        print(f"   Prompt: {visual_result.get('prompt', 'N/A')[:100]}...")
        print(f"   Seed: {visual_result.get('seed', 'N/A')}")
        print("=" * 60)

        # Return ONLY the fields to update
        return {
            "visual_output": visual_result
        }

    # Core Visual Generation Logic
    def _generate_visual(self, brief: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate main visual image from design brief.
        Args:
            brief: Design brief with theme, style, description, etc.
        Returns:
            Dictionary with image_path, prompt, seed, metadata
        """
        # Load pipeline (lazy)
        if self.pipe is None:
            self.pipe = self._load_pipeline()

        # Construct prompt from brief
        prompt = self._build_prompt(brief)
        print(f"📝 Prompt: {prompt}")

        # Generate image
        print("🖼️  Generating image with Stable Diffusion...")
        img, seed = self._generate_image(prompt)

        # Apply color grading
        img = self._apply_color_grade(img)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visual_{timestamp}_seed{seed}.png"
        output_path = self.output_dir / filename
        img.save(output_path, quality=95)

        print(f"✅ Visual saved: {output_path}")

        return {
            "image_path": str(output_path),
            "prompt": prompt,
            "seed": seed,
            "width": self.image_width,
            "height": self.image_height,
            "timestamp": timestamp,
        }

    def _build_prompt(self, brief: Dict[str, Any]) -> str:
        """
        Construct Stable Diffusion prompt from design brief.
        Args:
            brief: Design brief dictionary
        Returns:
            Optimized prompt string
        """
        theme = brief.get("theme", "")
        style = brief.get("style", "")
        tone = brief.get("tone", "")
        emotion = brief.get("emotion", "")
        tagline = brief.get("tagline", "")
        description = brief.get("description", "")
        suggestion = brief.get("quality_suggestions", "")

        # Build prompt with quality boosters
        prompt_parts = []

        if theme:
            prompt_parts.append(f"{theme} theme")

        if style:
            prompt_parts.append(f"{style} style")

        if tone or emotion:
            mood = f"{tone} {emotion}".strip()
            prompt_parts.append(f"{mood} atmosphere")

        if description:
            prompt_parts.append(description)

        if suggestion:
            prompt_parts.append(f"suggestion: {suggestion}")

        # Quality boosters
        prompt_parts.extend([
            "professional food photography",
            "studio lighting",
            "high detail",
            "8k resolution",
            "appetizing",
        ])

        prompt = ", ".join(filter(None, prompt_parts))
        return prompt

    def _generate_image(
            self,
            prompt: str,
            seed: int | None = None
    ) -> tuple[Image.Image, int]:
        """
        Generate image using Stable Diffusion.
        Args:
            prompt: Text prompt
            seed: Random seed (optional)
        Returns:
            Tuple of (PIL Image, seed used)
        """
        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)

        print(f"🎲 Using seed: {seed}")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        result = self.pipe(
            prompt,
            width=self.image_width,
            height=self.image_height,
            guidance_scale=7.5,
            num_inference_steps=28,
            generator=generator,
        )

        return result.images[0], seed

    # Image Processing
    @staticmethod
    def _apply_color_grade(img: Image.Image) -> Image.Image:
        """Apply color grading to enhance image quality"""
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(1.02)
        img = ImageEnhance.Sharpness(img).enhance(1.02)
        return img

    # Pipeline Setup
    def _load_pipeline(self) -> StableDiffusionPipeline:
        """Load Stable Diffusion pipeline with optimizations"""
        print(f"📦 Loading model: {self.model_id}")

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,  # MPS requires float32
            use_safetensors=True,
            safety_checker=None,
        ).to(self.device)

        # Memory optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()

        print("✅ Pipeline loaded successfully")
        return pipe

    @staticmethod
    def _get_device() -> str:
        """Detect best available device (MPS/CUDA/CPU)"""
        if torch.backends.mps.is_available():
            print("🔥 Using Apple Silicon MPS GPU")
            return "mps"
        elif torch.cuda.is_available():
            print("🔥 Using NVIDIA CUDA GPU")
            return "cuda"
        else:
            print("⚠️  Using CPU (slower)")
            return "cpu"


def test_visual_designer():
    """Test the agent standalone"""
    # Mock state from ConceptExpander
    mock_state = {
        "expanded_concept": """{
            "theme": "Seafood Noodle",
            "style": "authentic Asian cuisine",
            "tone": "warm",
            "emotion": "satisfying umami",
            "description": "Rich seafood broth with fresh shrimp, scallops, and squid",
            "tagline": "Fresh from the Ocean"
        }""",
        "iteration_count": 0,
        "max_iterations": 3,
    }

    # Create agent
    agent = VisualDesignerAgent(
        output_dir="test_images",
        image_width=512,
        image_height=512,
    )
    # Run
    agent.run(mock_state)

if __name__ == "__main__":
    # Run standalone test
    test_visual_designer()