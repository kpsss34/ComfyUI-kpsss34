import torch
import random
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as model_management
from diffusers import StableDiffusionXLInstructPix2PixPipeline, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextConfig
from safetensors.torch import load_file
import os
import json
import tempfile
import shutil

class i2iFlash:
    def __init__(self):
        self.pipe = None; self.current_model_path = None; self.temp_dir = None

    def __del__(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try: shutil.rmtree(self.temp_dir)
            except Exception as e: print(f"Error removing temp dir: {e}")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "model_file": (folder_paths.get_filename_list("checkpoints"),), "image": ("IMAGE",),"instruction": ("STRING", { "multiline": True, "default": "prompt edit..." }),"guidance_scale": ("FLOAT", { "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1 }),"image_guidance_scale": ("FLOAT", { "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1 }),"num_inference_steps": ("INT", { "default": 30, "min": 10, "max": 100, "step": 1 }),"seed": ("INT", { "default": 43, "min": 0, "max": 2**32 }),"width": ("INT", { "default": 512, "min": 64, "max": 2048, "step": 8 }),"height": ("INT", { "default": 512, "min": 64, "max": 2048, "step": 8 }),}}

    RETURN_TYPES = ("IMAGE",); FUNCTION = "edit_image"; CATEGORY = "image/editing"

    def load_model(self, model_path):
        if not os.path.isfile(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
        
        from safetensors import safe_open
        with safe_open(model_path, framework="pt", device="cpu") as f: metadata = f.metadata()
        
        if not metadata or metadata.get("__format__") != "ComfyUI-I2I_06BFlash-TrueSingleFile-v2":
             raise ValueError("Model not packed correctly or is an old version. Please re-pack the model with the latest script.")

        if self.pipe is not None and self.current_model_path == model_path: return
        if self.pipe is not None: del self.pipe; torch.cuda.empty_cache();
        if self.temp_dir and os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir); self.temp_dir = None

        print(f"Loading TRUE OFFLINE single-file model: {model_path}")
        device = model_management.get_torch_device()
        state_dict = load_file(model_path, device="cpu")
        dtype = getattr(torch, metadata["__torch_dtype__"].split('.')[-1])
        
        unet_config = json.loads(metadata["__unet_config__"])
        vae_config = json.loads(metadata["__vae_config__"])
        scheduler_config = json.loads(metadata["__scheduler_config__"])
        text_encoder_config_dict = json.loads(metadata["__text_encoder_config__"])
        text_encoder_2_config_dict = json.loads(metadata["__text_encoder_2_config__"])
        
        print("Recreating tokenizers from embedded data...")
        self.temp_dir = tempfile.mkdtemp()
        
        tok1_dir = os.path.join(self.temp_dir, 'tokenizer_1'); os.makedirs(tok1_dir)
        with open(os.path.join(tok1_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f: f.write(metadata["__tokenizer_1_config__"])
        with open(os.path.join(tok1_dir, 'vocab.json'), 'w', encoding='utf-8') as f: f.write(metadata["__tokenizer_1_vocab__"])
        with open(os.path.join(tok1_dir, 'merges.txt'), 'w', encoding='utf-8') as f: f.write(metadata["__tokenizer_1_merges__"])
        tokenizer_1 = CLIPTokenizer.from_pretrained(tok1_dir)
        
        tok2_dir = os.path.join(self.temp_dir, 'tokenizer_2'); os.makedirs(tok2_dir)
        with open(os.path.join(tok2_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f: f.write(metadata["__tokenizer_2_config__"])
        with open(os.path.join(tok2_dir, 'vocab.json'), 'w', encoding='utf-8') as f: f.write(metadata["__tokenizer_2_vocab__"])
        with open(os.path.join(tok2_dir, 'merges.txt'), 'w', encoding='utf-8') as f: f.write(metadata["__tokenizer_2_merges__"])
        tokenizer_2 = CLIPTokenizer.from_pretrained(tok2_dir)

        unet = UNet2DConditionModel.from_config(unet_config)
        vae = AutoencoderKL.from_config(vae_config)
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        text_encoder_config = CLIPTextConfig.from_dict(text_encoder_config_dict)
        text_encoder_2_config = CLIPTextConfig.from_dict(text_encoder_2_config_dict)
        text_encoder = CLIPTextModel(text_encoder_config)
        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_2_config)

        unet.load_state_dict({k.replace('unet.', ''): v for k, v in state_dict.items() if k.startswith('unet.')})
        vae.load_state_dict({k.replace('vae.', ''): v for k, v in state_dict.items() if k.startswith('vae.')})
        text_encoder.load_state_dict({k.replace('text_encoder.', ''): v for k, v in state_dict.items() if k.startswith('text_encoder.') and not k.startswith('text_encoder_2.')})
        text_encoder_2.load_state_dict({k.replace('text_encoder_2.', ''): v for k, v in state_dict.items() if k.startswith('text_encoder_2.')})
            
        self.pipe = StableDiffusionXLInstructPix2PixPipeline(vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, tokenizer=tokenizer_1, tokenizer_2=tokenizer_2, unet=unet, scheduler=scheduler).to(device, dtype=dtype)
        self.current_model_path = model_path
        print(f"Model fully loaded from single file onto {device}")

    def edit_image(self, model_file, image, instruction, guidance_scale, image_guidance_scale, num_inference_steps, seed, width, height):
        model_path = folder_paths.get_full_path("checkpoints", model_file)
        self.load_model(model_path)
        
        pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        if (width, height) != pil_image.size: pil_image = pil_image.resize((width, height), Image.LANCZOS)
        
        current_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
        generator = torch.manual_seed(current_seed)
        
        print(f"Editing image with seed: {current_seed}...")
        with torch.no_grad():

            result_image = self.pipe(
                prompt=instruction, 
                image=pil_image, 
                width=width,
                height=height,
                guidance_scale=guidance_scale, 
                image_guidance_scale=image_guidance_scale, 
                num_inference_steps=num_inference_steps, 
                generator=generator
            ).images[0]
        
        return (torch.from_numpy(np.array(result_image).astype(np.float32) / 255.0).unsqueeze(0),)

NODE_CLASS_MAPPINGS = {"i2iFlash": i2iFlash}
NODE_DISPLAY_NAME_MAPPINGS = {"i2iFlash": "i2iFlash"}
