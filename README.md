# Multi-Model Image Generation

Compare image generation across 3 AI models using the same prompt. See which model produces the best result for your use case.

## How it works

```
Text prompt
    │
    ▼
┌──────────────────────────────────────────────────┐
│              Image Generation Models             │
│                                                  │
│  ┌──────────────┐ ┌────────────┐ ┌───────────┐   │
│  │ Stable       │ │ Flux       │ │ SD 3      │   │
│  │ Diffusion XL │ │ Schnell    │ │ Medium    │   │
│  └──────┬───────┘ └─────┬──────┘ └─────┬─────┘   │
│         │               │              │         │
└─────────┼───────────────┼──────────────┼─────────┘
          │               │              │
          ▼               ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Image 1  │   │ Image 2  │   │ Image 3  │
    └──────────┘   └──────────┘   └──────────┘
```

One prompt, three models, side-by-side comparison. All using the free Hugging Face Inference API.

## Get a free Hugging Face token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Sign up (free)
3. Create a new token (read access is enough)

## Quick start

### Option A: Run locally

```bash
git clone https://github.com/marinedabare/multi-model-image-gen.git
cd multi-model-image-gen

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your Hugging Face token
```

```bash
python generate.py "a flat illustration of a merchant reviewing analytics on a tablet"
python generate.py "an isometric coffee shop with warm lighting" --output coffee.png
```

| Argument | Required | Description |
|----------|----------|-------------|
| `prompt` | Yes | Text description of the image to generate |
| `--output`, `-o` | No | Output file path (default: `comparison.png`) |

### Option B: Run in Google Colab (no install needed)

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook
2. In **Cell 1**, install dependencies:

```
!pip install Pillow requests matplotlib
```

3. In **Cell 2**, paste this code and run it:

```python
import requests, io, time
from PIL import Image
import matplotlib.pyplot as plt

HF_TOKEN = "paste-your-token-here"  # get one free at https://huggingface.co/settings/tokens
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

PROMPT = "a flat illustration of a merchant reviewing analytics on a tablet, minimal style, pastel colours, no text"

models = {
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Flux Schnell": "black-forest-labs/FLUX.1-schnell",
    "SD 3 Medium": "stabilityai/stable-diffusion-3-medium-diffusers",
}

def generate(model_id, prompt):
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    for _ in range(3):
        r = requests.post(url, headers=HEADERS, json={"inputs": prompt}, timeout=120)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content))
        if r.status_code == 503:
            time.sleep(r.json().get("estimated_time", 30))
            continue
        return None
    return None

images = {}
for name, mid in models.items():
    print(f"Generating with {name}...")
    images[name] = generate(mid, PROMPT)

valid = {k: v for k, v in images.items() if v is not None}
fig, axes = plt.subplots(1, len(valid), figsize=(8 * len(valid), 8))
if len(valid) == 1: axes = [axes]
for ax, (name, img) in zip(axes, valid.items()):
    ax.imshow(img)
    ax.set_title(name, fontsize=16)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

4. Change `PROMPT` to try different styles and subjects

## Models

| Model | Provider | Strengths |
|-------|----------|-----------|
| [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Stability AI | Good general-purpose, consistent quality |
| [FLUX.1 Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Black Forest Labs | Fast, strong prompt adherence |
| [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | Stability AI | Newer architecture, better text understanding |

All models are accessed via the [Hugging Face Inference API](https://huggingface.co/docs/inference-providers) (free tier).

## Tech stack

- **Python 3.11+**
- **Hugging Face Inference API** — text-to-image generation (free)
- **Pillow** — image processing
- **matplotlib** — side-by-side comparison display

## License

MIT
