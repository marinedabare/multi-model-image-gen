"""
AI Image Style Transfer Pipeline

Two-model pipeline that analyses an image's visual style using a vision model (Gemini),
then generates new images matching that style using multiple image generation models
(Hugging Face). Compares outputs side by side.

Pipeline:
    Reference image → Gemini (vision) → style description → HF models → generated images
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

VISION_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{VISION_MODEL}:generateContent"

IMAGE_MODELS = {
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Flux Schnell": "black-forest-labs/FLUX.1-schnell",
    "SD 3 Medium": "stabilityai/stable-diffusion-3-medium-diffusers",
}

STYLE_ANALYSIS_PROMPT = """Analyse this illustration and describe its visual style for an AI image generator.
Be specific about:
- Art style (flat, isometric, line art, vector, etc.)
- Colour palette (specific tones, warm/cool, pastel/vibrant)
- Composition (centered, scene-based, icon-style)
- Level of detail (minimal, detailed, complex)
- Shapes (geometric, organic, rounded)
- Background style (solid, gradient, scene)
- Overall mood/feel

Output ONLY the style description, no other text. Keep it under 100 words."""


def encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64 string, mime type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_types.get(suffix, "image/png")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def analyse_style(image_path: str) -> str:
    """Use Gemini vision to describe an image's visual style."""
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    print(f"Analysing style of {image_path}...")
    img_b64, mime = encode_image(image_path)

    payload = {
        "contents": [{
            "parts": [
                {"text": STYLE_ANALYSIS_PROMPT},
                {"inline_data": {"mime_type": mime, "data": img_b64}},
            ]
        }]
    }

    response = requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        print(f"Gemini error {response.status_code}: {response.text}")
        sys.exit(1)

    description = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return description.strip()


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


def build_prompt(style_description: str, subject: str) -> str:
    """Combine the style description with a subject to create the generation prompt."""
    return f"{subject}. Style: {style_description}. No text, no watermark."


def display_results(
    ref_path: str | None,
    style_description: str,
    subject: str,
    results: dict[str, Image.Image | str],
    output_path: str,
):
    """Display reference image, style description, and generated images."""
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in results.items() if isinstance(v, Image.Image)}
    has_ref = ref_path is not None
    cols = max(len(valid), 1)
    rows = 2 if has_ref else 1

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows + 2))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    elif cols == 1:
        axes = [[axes[0]], [axes[1]]]

    if has_ref:
        ref_img = Image.open(ref_path)
        mid = cols // 2
        for i in range(cols):
            axes[0][i].axis("off")
        axes[0][mid].imshow(ref_img)
        axes[0][mid].set_title("Reference Image", fontsize=14, color="blue", fontweight="bold")

        row = axes[1]
    else:
        row = axes if isinstance(axes[0], plt.Axes) else axes[0]

    for i, (name, img) in enumerate(valid.items()):
        row[i].imshow(img)
        row[i].set_title(name, fontsize=14, color="green", fontweight="bold")
        row[i].axis("off")
    for i in range(len(valid), cols):
        row[i].axis("off")

    prompt_text = build_prompt(style_description, subject)
    title = f'Subject: "{subject}"\nStyle: "{style_description[:120]}..."' if len(style_description) > 120 else f'Subject: "{subject}"\nStyle: "{style_description}"'
    fig.suptitle(title, fontsize=10, y=0.99, ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Image Style Transfer: analyse a reference image's style and generate new images matching it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyse a reference image and generate new images in its style
  python generate.py --reference my_illustration.png --subject "a merchant checking inventory"

  # Generate without a reference image (uses a default style description)
  python generate.py --subject "a coffee shop owner using a POS system" --style "flat vector, pastel colours, minimal"

  # Save output to a specific file
  python generate.py --reference logo.png --subject "a team meeting" --output result.png
        """,
    )
    parser.add_argument("--reference", "-r", help="Path to a reference image to analyse its style")
    parser.add_argument("--subject", "-s", required=True, help="What to generate (e.g. 'a merchant reviewing analytics')")
    parser.add_argument("--style", help="Manual style description (skips Gemini analysis)")
    parser.add_argument("--output", "-o", default="comparison.png", help="Output file path (default: comparison.png)")
    args = parser.parse_args()

    if not args.reference and not args.style:
        print("Error: provide either --reference (image to analyse) or --style (manual description)")
        sys.exit(1)

    print("=" * 60)
    print("AI Image Style Transfer Pipeline")
    print("=" * 60)

    # Step 1: Get style description
    if args.reference:
        if not Path(args.reference).exists():
            print(f"Error: file not found: {args.reference}")
            sys.exit(1)
        style = analyse_style(args.reference)
        print(f"\nDetected style:\n  {style}\n")
    else:
        style = args.style
        print(f"\nUsing manual style:\n  {style}\n")

    # Step 2: Build the generation prompt
    prompt = build_prompt(style, args.subject)
    print(f"Generation prompt:\n  {prompt}\n")

    # Step 3: Generate with all models
    results = {}
    for name, model_id in IMAGE_MODELS.items():
        print(f"[{len(results) + 1}/{len(IMAGE_MODELS)}] Generating with {name}...")
        result = generate_image(model_id, prompt)
        results[name] = result
        if isinstance(result, Image.Image):
            print(f"  ✓ Success")
        else:
            print(f"  ✗ {result}")

    # Step 4: Display comparison
    print("\nDisplaying comparison...")
    display_results(args.reference, style, args.subject, results, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
