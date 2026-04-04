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

## How to run

### Option A: Live demo on Hugging Face

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/marinedabare/multi-model-image-gen)

No setup, no token needed — just type a prompt and click Submit.

### Option B: Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marinedabare/multi-model-image-gen/blob/main/notebook.ipynb?flush_cache=true)

1. Click the badge above to open the notebook in Colab
2. Paste your Hugging Face token in the `HF_TOKEN` variable
3. Run all cells — images appear side by side after ~1-2 minutes
4. Change `PROMPT` to try different styles and subjects

### Option C: Run locally

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
python generate.py "a flat illustration of a merchant reviewing analytics on a tablet in a retail store"
python generate.py "an isometric coffee shop with warm lighting" --output coffee.png
```

| Argument | Required | Description |
|----------|----------|-------------|
| `prompt` | Yes | Text description of the image to generate |
| `--output`, `-o` | No | Output file path (default: `comparison.png`) |

The output is a single image containing all 3 model results side by side, with the prompt displayed above.

## Models

| Model | Provider | Strengths |
|-------|----------|-----------|
| [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Stability AI | Good general-purpose, consistent quality |
| [FLUX.1 Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Black Forest Labs | Fast, strong prompt adherence |
| [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | Stability AI | Newer architecture, better text understanding |

All models are accessed via the [Hugging Face Inference API](https://huggingface.co/docs/inference-providers) (free tier).

> **Note:** The free Hugging Face API has a limited number of requests per day (~1,000 requests/day, so ~300 generations since each run calls 3 models). If you hit the limit, wait a few hours and try again.

## Tech stack

- **Python 3.11+**
- **Hugging Face Inference API** — text-to-image generation (free)
- **Pillow** — image processing
- **matplotlib** — side-by-side comparison display

## License

MIT
