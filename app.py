import io
import os
import time

import gradio as gr
import requests
from PIL import Image

HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = {
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Flux Schnell": "black-forest-labs/FLUX.1-schnell",
    "SD 3 Medium": "stabilityai/stable-diffusion-3-medium-diffusers",
}


def generate_image(model_id: str, prompt: str, retries: int = 3):
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=120)
        except requests.RequestException:
            continue

        content_type = response.headers.get("content-type", "unknown")

        if response.status_code == 200:
            if "image" not in content_type:
                raise gr.Error(f"{model_id}: expected image, got {content_type} — {response.text[:200]}")
            try:
                return Image.open(io.BytesIO(response.content))
            except Exception as e:
                raise gr.Error(f"{model_id}: failed to decode image ({content_type}, {len(response.content)} bytes): {e}")

        if response.status_code == 402:
            return "limit"

        if response.status_code == 410:
            raise gr.Error(f"{model_id}: model has been deprecated and is no longer available")

        if response.status_code == 503:
            try:
                wait = response.json().get("estimated_time", 30)
            except Exception:
                wait = 30
            time.sleep(min(wait, 60))
            continue

        raise gr.Error(f"{model_id}: error {response.status_code} — {response.text[:300]}")

    return None


def generate_all(prompt):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt")

    if not HF_TOKEN:
        raise gr.Error("HF_TOKEN secret not set in Space settings")

    results = []
    for name, model_id in MODELS.items():
        result = generate_image(model_id, prompt)
        if result == "limit":
            gr.Warning("Free API limit reached. Please try again later.")
            return [None, None, None]
        results.append(result)

    return results


demo = gr.Interface(
    fn=generate_all,
    inputs=gr.Textbox(
        label="Prompt",
        placeholder="a flat illustration of a smiling merchant reviewing analytics on a tablet in a retail store",
        lines=2,
    ),
    outputs=[
        gr.Image(label=name)
        for name in MODELS
    ],
    title="Multi-Model Image Generation",
    description="Compare image generation across 3 AI models using the same prompt. All models are free via the Hugging Face Inference API.",
    examples=[
        ["a flat illustration of a smiling merchant reviewing analytics on a tablet in a retail store, minimal style, no text, no watermark"],
        ["an isometric coffee shop with warm lighting, minimal style, pastel colours"],
        ["a vector illustration of a team meeting in a modern office, flat design"],
    ],
    flagging_mode="never",
)

demo.launch()
