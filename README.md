# AI Image Style Transfer Pipeline

A two-model AI pipeline that analyses a reference image's visual style and generates new images matching that style using multiple image generation models.

## How it works

```
Reference image
    │
    ▼
┌──────────────────────────┐
│  Gemini Vision (Google)  │  ← Analyses the image
│  "Describe this style"   │
└────────────┬─────────────┘
             │
             ▼
    Style description:
    "Flat vector illustration,
     pastel purple and teal,
     minimal geometric shapes..."
             │
             ▼
┌──────────────────────────────────────────────────┐
│              Image Generation Models              │
│                                                  │
│  ┌──────────────┐ ┌────────────┐ ┌───────────┐  │
│  │ Stable       │ │ Flux       │ │ SD 3      │  │
│  │ Diffusion XL │ │ Schnell    │ │ Medium    │  │
│  └──────┬───────┘ └─────┬──────┘ └─────┬─────┘  │
│         │               │              │         │
└─────────┼───────────────┼──────────────┼─────────┘
          │               │              │
          ▼               ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Image 1  │   │ Image 2  │   │ Image 3  │
    └──────────┘   └──────────┘   └──────────┘
```

The pipeline chains two AI capabilities:
1. **Vision analysis** (Gemini) — reads the reference image and outputs a detailed style description
2. **Image generation** (Hugging Face) — uses that description as a prompt to generate new images

This lets you reproduce a visual style without manually writing complex prompts.

## Quick start

### 1. Clone the repo

```bash
git clone https://github.com/marinedabare/ai-image-style-transfer.git
cd ai-image-style-transfer
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Get API keys (both are free)

| Service | Get key at | Cost |
|---------|-----------|------|
| Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free (1,500 requests/day) |
| Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free (rate-limited) |

### 4. Set up environment

```bash
cp .env.example .env
# Edit .env and paste your API keys
```

### 5. Run

```bash
# Analyse a reference image and generate new images in its style
python generate.py --reference my_illustration.png --subject "a merchant checking inventory"

# Or provide a style description manually (no Gemini call needed)
python generate.py --subject "a coffee shop" --style "flat vector, pastel colours, minimal shapes"
```

## Usage

```
python generate.py --reference <image> --subject <what to generate> [--output <file>]
python generate.py --style <description> --subject <what to generate> [--output <file>]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--reference`, `-r` | One of `--reference` or `--style` | Path to reference image (Gemini analyses its style) |
| `--style` | One of `--reference` or `--style` | Manual style description (skips Gemini) |
| `--subject`, `-s` | Yes | What the image should depict |
| `--output`, `-o` | No | Output file path (default: `comparison.png`) |

## Examples

### With a reference image

```bash
python generate.py \
  --reference examples/reference.png \
  --subject "a retail store owner serving a customer"
```

Output: the reference image on top, three generated images below — each from a different model, all matching the reference style.

### Without a reference image

```bash
python generate.py \
  --subject "a business owner looking at analytics" \
  --style "isometric illustration, soft blue and purple tones, clean lines, minimal detail, white background"
```

## Models used

| Model | Provider | Strengths |
|-------|----------|-----------|
| [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Stability AI | Good general-purpose, consistent quality |
| [FLUX.1 Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Black Forest Labs | Fast, strong prompt adherence |
| [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) | Stability AI | Newer architecture, better text understanding |

All models are accessed via the [Hugging Face Inference API](https://huggingface.co/docs/inference-providers) (free tier).

## Tech stack

- **Python 3.11+**
- **Google Gemini API** — vision model for style analysis
- **Hugging Face Inference API** — text-to-image generation
- **Pillow** — image processing
- **matplotlib** — side-by-side comparison display

## Why this architecture?

Free image generation APIs don't accept reference images for style matching. This pipeline works around that limitation by using a vision model to extract the style as text, then using that text as the generation prompt. The result: style transfer using only free APIs.

## License

MIT
