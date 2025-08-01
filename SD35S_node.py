# --- START OF FILE SD35S_node.py ---

import os
import gc
import json
import hashlib
import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as model_management
import comfy.utils
from safetensors.torch import safe_open, load_file
from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextConfig
import base64

# This creates a permanent cache directory inside your custom node folder.
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(NODE_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _reconstruct_component(metadata, state_dict, name, model_class, device, dtype):
    config_str = metadata.get(f"{name}_config")
    if not config_str: raise ValueError(f"Config for '{name}' not found in metadata.")
    config_dict = json.loads(config_str)
    
    if model_class is CLIPTextModelWithProjection:
        config = CLIPTextConfig.from_dict(config_dict)
        model = model_class(config)
    else:
        model = model_class(**config_dict)
    
    component_sd = {k.replace(f"{name}.", ""): v for k, v in state_dict.items() if k.startswith(f"{name}.")}
    model.load_state_dict(component_sd)
    
    return model.to(device, dtype=dtype).eval()

def _reconstruct_tokenizer(metadata, name, tokenizer_class, model_name):
    model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    tokenizer_cache_dir = os.path.join(CACHE_DIR, model_hash, name)

    if not os.path.exists(os.path.join(tokenizer_cache_dir, "tokenizer_config.json")):
        os.makedirs(tokenizer_cache_dir, exist_ok=True)
        files_written = 0
        for key, value in metadata.items():
            if key.startswith(f"{name}_"):
                if key.endswith("_b64"):
                    filename = key.replace(f"{name}_", "").replace("_b64", "")
                    path = os.path.join(tokenizer_cache_dir, filename)
                    content = base64.b64decode(value)
                    with open(path, 'wb') as f: f.write(content)
                else:
                    if f"{key}_b64" in metadata: continue
                    filename = key.replace(f"{name}_", "")
                    path = os.path.join(tokenizer_cache_dir, filename)
                    with open(path, 'w', encoding='utf-8') as f: f.write(value)
                files_written += 1
        
        if files_written == 0:
            raise RuntimeError(f"FATAL: No tokenizer files for '{name}' were found in the repacked model's metadata.")

    return tokenizer_class.from_pretrained(tokenizer_cache_dir)

class SD35SLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list("checkpoints"),)}}

    RETURN_TYPES = ("SD35_S_COMPONENTS",)
    RETURN_NAMES = ("components_s",)
    FUNCTION = "unpack_model"
    CATEGORY = "SD35S"
    TITLE = "SD3.5 S Loader"

    def unpack_model(self, model_name):
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        device = model_management.get_torch_device()
        dtype = torch.bfloat16 if model_management.should_use_bf16() else torch.float16
        
        with safe_open(model_path, framework="pt", device="cpu") as f: metadata = f.metadata()
        state_dict = load_file(model_path, device="cpu")
        
        components = {}
        components['vae'] = _reconstruct_component(metadata, state_dict, "vae", AutoencoderKL, device, dtype)
        components['transformer'] = _reconstruct_component(metadata, state_dict, "transformer", SD3Transformer2DModel, device, dtype)
        components['text_encoders'] = [_reconstruct_component(metadata, state_dict, name, mc, device, dtype) for name, mc in [("text_encoder", CLIPTextModelWithProjection), ("text_encoder_2", CLIPTextModelWithProjection)]]
        components['tokenizers'] = [_reconstruct_tokenizer(metadata, name, tc, model_name) for name, tc in [("tokenizer", CLIPTokenizer), ("tokenizer_2", CLIPTokenizer)]]
        
        components['scheduler'] = FlowMatchEulerDiscreteScheduler.from_config(json.loads(metadata.get("scheduler_config", "{}")))
        components['device'], components['dtype'] = device, dtype
        
        return (components,)

# --- แก้ไข: เปลี่ยนชื่อคลาสและปรับการทำงานเป็น Img2Img ---
class SD35S_Img2Img:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { 
            "components_s": ("SD35_S_COMPONENTS",),
            "image": ("IMAGE",), 
            "prompt": ("STRING", {"default": "a beautiful portrait, masterpiece", "multiline": True}), 
            "negative_prompt": ("STRING", {"default": "blurry, ugly, deformed", "multiline": True}),
            "steps": ("INT", {"default": 28, "min": 1, "max": 100}), 
            "cfg": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.5}),
            "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        } }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "img2img"
    CATEGORY = "SD35S"
    TITLE = "SD3.5 S Img2Img"

    def img2img(self, components_s, image, prompt, negative_prompt, steps, cfg, denoise, seed):
        vae = components_s['vae']
        transformer = components_s['transformer']
        text_encoders = components_s['text_encoders']
        tokenizers = components_s['tokenizers']
        scheduler = components_s['scheduler']
        device = components_s['device']
        dtype = components_s['dtype']
        
        # 1. Preprocess Image and Encode Prompt
        with torch.no_grad():
            # Image to Latent
            input_image_tensor = image.permute(0, 3, 1, 2).to(device, dtype=dtype)
            latents = vae.encode(input_image_tensor).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Prompt Encoding
            def encode(p):
                text_inputs_1 = tokenizers[0]([p], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                outputs_1 = text_encoders[0](text_inputs_1.input_ids.to(device), output_hidden_states=True)
                text_inputs_2 = tokenizers[1]([p], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
                outputs_2 = text_encoders[1](text_inputs_2.input_ids.to(device), output_hidden_states=True)
                pooled = torch.cat([outputs_1.pooler_output, outputs_2.pooler_output], dim=-1)
                embeds = torch.cat([outputs_1.hidden_states[-2], outputs_2.hidden_states[-2]], dim=-1)
                target_dim = transformer.config.joint_attention_dim
                if embeds.shape[-1] < target_dim:
                    embeds = torch.nn.functional.pad(embeds, (0, target_dim - embeds.shape[-1]))
                return embeds, pooled

            cond_embeds, cond_pooled = encode(prompt)
            uncond_embeds, uncond_pooled = encode(negative_prompt)

            prompt_embeds = torch.cat([uncond_embeds, cond_embeds])
            pooled_prompt_embeds = torch.cat([uncond_pooled, uncond_pooled])

        # 2. Denoising Loop
        scheduler.set_timesteps(steps, device=device)
        start_step = max(steps - int(steps * denoise), 0)
        timesteps = scheduler.timesteps[start_step:]
        
        noise = torch.randn_like(latents, generator=torch.manual_seed(seed))
        latents = scheduler.add_noise(latents, noise, timesteps[0].unsqueeze(0))

        pbar = comfy.utils.ProgressBar(len(timesteps))
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            timestep_tensor = t.expand(latent_model_input.shape[0])
            
            noise_pred = transformer(
                hidden_states=latent_model_input, 
                timestep=timestep_tensor, 
                encoder_hidden_states=prompt_embeds, 
                pooled_projections=pooled_prompt_embeds
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)
            
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            pbar.update(1)

        # 3. Decode and Postprocess
        latents = latents / vae.config.scaling_factor
        output_image = vae.decode(latents).sample
        output_tensor = (output_image.clamp(-1, 1) / 2 + 0.5).cpu().permute(0, 2, 3, 1).float()
            
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "SD35SLoader": SD35SLoader,
    "SD35S_Img2Img": SD35S_Img2Img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SD35SLoader": "SD3.5 S Loader",
    "SD35S_Img2Img": "SD3.5 S Img2Img"
}