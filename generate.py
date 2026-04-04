"""
Multi-Model Image Generation

Compare image generation across 3 AI models using the same prompt.
Uses the Hugging Face free Inference API to generate images from
Stable Diffusion XL, Flux Schnell, and SD 3 Medium side by side.
"""

import io
import os
import sys
import time

import requests
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

IMAGE_MODELS = {
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Flux Schnell": "black-forest-labs/FLUX.1-schnell",
    "SD 3 Medium": "stabilityai/stable-diffusion-3-medium-diffusers",
}


def generate_image(model_id: str, prompt: str, retries: int = 3) -> Image.Image | str:
    """Generate an image from a text prompt using a Hugging Face model."""
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set. Add it to .env file.")
        sys.exit(1)

    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    for attempt in range(retries):
        response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=120)

        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))

        if response.status_code == 503:
            wait = response.json().get("estimated_time", 30)
            print(f"  Model loading, waiting {wait:.0f}s (attempt {attempt + 1}/{retries})...")
            time.sleep(wait)
            continue

        return f"Error {response.status_code}: {response.text[:100]}"

    return "Error: model did not load after retries"


def display_results(prompt: str, results: dict[str, Image.Image | str], output_path: str):
    """Display generated images side by side and save to file."""
    valid = {k: v for k, v in results.items() if isinstance(v, Image.Image)}

    if not valid:
        print("No images were generated successfully.")
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(8 * len(valid), 8))
    if len(valid) == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, valid.items()):
        ax.imshow(img)
        ax.set_title(name, fontsize=16)
        ax.axis("off")

    display_prompt = prompt[:120] + "..." if len(prompt) > 120 else prompt
    fig.suptitle(f'Prompt: {display_prompt}', fontsize=18, color="#1a73e8", fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate images with 3 AI models and compare them side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py "a flat illustration of a merchant reviewing analytics on a tablet in a retail store"
  python generate.py "an isometric coffee shop with warm lighting, minimal style"
  python generate.py "a vector illustration of a team meeting" --output team.png
        """,
    )
    parser.add_argument("prompt", help="Text description of the image to generate")
    parser.add_argument("--output", "-o", default="comparison.png", help="Output file path (default: comparison.png)")
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Model Image Generation")
    print("=" * 60)
    print(f"\nPrompt: {args.prompt}\n")

    results = {}
    for name, model_id in IMAGE_MODELS.items():
        print(f"[{len(results) + 1}/{len(IMAGE_MODELS)}] Generating with {name}...")
        result = generate_image(model_id, args.prompt)
        results[name] = result
        if isinstance(result, Image.Image):
            print(f"  ✓ Success")
        else:
            print(f"  ✗ {result}")

    print("\nDisplaying comparison...")
    display_results(args.prompt, results, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
