import os, torch
from PIL import Image
from diffusers import QwenImageEditPipeline

pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
pipe.to(torch.bfloat16).to("cuda")

# input
image = Image.open("Bagel/test/input/img1.jpg").convert("RGB")
prompt = "Give the woman 38DD breasts"

# controls that matter
out = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=50,  # 30–60 usually good
    true_cfg_scale=4.0,      # recommended starting point
    negative_prompt=" ",     # keep empty unless you see artifacts
    generator=torch.manual_seed(0)
).images[0]

out.save("output_image_edit.png")
print("Saved:", os.path.abspath("output_image_edit.png"))

