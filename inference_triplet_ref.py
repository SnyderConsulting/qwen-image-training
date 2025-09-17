import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from diffusers import QwenImageEditPipeline


def load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def letterbox_to_height(img: Image.Image, height: int, pad_color=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    scale = height / h
    new_w = max(1, int(round(w * scale)))
    resized = img.resize((new_w, height), Image.LANCZOS)
    if new_w != height:
        pad_left = (height - new_w) // 2
        pad_right = height - new_w - pad_left
        resized = ImageOps.expand(resized, border=(pad_left, 0, pad_right, 0), fill=pad_color)
    return resized


def make_composite(left_img: Image.Image, right_img: Image.Image, height: int = 1024):
    left = letterbox_to_height(left_img, height)
    right = letterbox_to_height(right_img, height)
    canvas = Image.new("RGB", (left.width + right.width, height), (0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas, left.width, height


def run(
    model: str = "Qwen/Qwen-Image-Edit",
    lora: str = None,
    source_path: str = None,
    ref_path: str = None,
    out_path: str = "edited.png",
    height: int = 1024,
    steps: int = 50,
):
    if source_path is None or ref_path is None:
        raise ValueError("source_path and ref_path must be provided")

    source = load_rgb(source_path)
    reference = load_rgb(ref_path)
    composite, left_width, height = make_composite(source, reference, height)

    pipe = QwenImageEditPipeline.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")
    if lora:
        pipe.load_lora_weights(lora)

    result = pipe(
        image=composite,
        prompt=" ",
        negative_prompt=" ",
        true_cfg_scale=1.0,
        num_inference_steps=steps,
        generator=torch.manual_seed(0),
    ).images[0]

    edited_left = result.crop((0, 0, left_width, height))
    output_path = Path(out_path)
    edited_left.save(output_path)
    print("Saved:", output_path.resolve())


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: python inference_triplet_ref.py <lora_path> <source.png> <reference.png> [out.png] [height] [steps]"
        )

    lora_path = sys.argv[1]
    source_img = sys.argv[2]
    reference_img = sys.argv[3]
    output = sys.argv[4] if len(sys.argv) > 4 else "edited.png"
    height = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 40

    run(
        lora=lora_path,
        source_path=source_img,
        ref_path=reference_img,
        out_path=output,
        height=height,
        steps=steps,
    )
