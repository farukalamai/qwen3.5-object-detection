# Zero-Shot Object Detection with Qwen3.5

Open-vocabulary object detection using [Qwen3.5](https://github.com/QwenLM/Qwen3.5), Alibaba's natively multimodal model. Unlike traditional detectors (YOLO, RT-DETR) that need fixed class lists and training data, this uses natural language prompts to detect anything — named individuals, spatial relationships, actions, jersey numbers, colors, logos, and more.

## What It Does

The notebook (`qwen35_open_vocab_detection.ipynb`) loads a quantized Qwen3.5 model and runs text-prompted detection on images:

- **Traffic scene** — detect cars, yellow taxis, multi-class vehicles
- **Solvay Conference photo** — detect named scientists (Einstein, Curie), spatial queries ("person between Einstein and Curie"), accessories (hats, canes)
- **Basketball game** — detect by jersey color, jersey number (#7), team name, action ("player about to shoot"), bench players, sponsor logos

Also includes thinking mode comparison, batch processing with VRAM tracking, and a section to try your own images.

## Requirements

- NVIDIA GPU with 6+ GB VRAM (tested on RTX 5070 Ti 16GB)
- Python 3.10+
- `transformers` (nightly), `bitsandbytes`, `supervision`, `qwen-vl-utils`

## Usage

```bash
pip install 'git+https://github.com/huggingface/transformers.git@main'
pip install supervision accelerate bitsandbytes qwen-vl-utils torchvision pillow
```