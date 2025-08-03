import torch
import time
from pipeline_stable_diffusion_3_S import StableDiffusion3SPipeline

seed = 43
generator = torch.manual_seed(seed)

pipe = StableDiffusion3SPipeline.from_pretrained(
    "kpsss34/Stable-Diffusion-3.5-Small-Preview1",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
).to("cuda")

filename = f"./img_{seed}_{int(time.time())}.png"

image = pipe(
    prompt="",
    negative_prompt="",
    num_inference_steps=40,
    guidance_scale=5.0,
    width=768,
    height=768,
    generator=generator,
).images[0]

image.save(filename)
print(f"Saved as {filename} (seed: {seed})")
