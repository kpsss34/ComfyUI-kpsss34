import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as model_management
import comfy.utils
from safetensors.torch import safe_open
from diffusers import AutoencoderKL, SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler

class SD35PlusLoader:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "model_name": (folder_paths.get_filename_list("checkpoints"),) } }

    RETURN_TYPES = ("SD35PLUS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SD35Plus"
    TITLE = "SD3.5+ Loader"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for key in f.keys():
                model_state_dict[key] = f.get_tensor(key)

        transformer_dict = {k.replace("model.diffusion_model.", ""): v for k, v in model_state_dict.items() if k.startswith("model.diffusion_model.")}
        vae_dict = {k.replace("first_stage_model.", ""): v for k, v in model_state_dict.items() if k.startswith("first_stage_model.")}

        model_data = {
            "transformer_dict": transformer_dict,
            "vae_dict": vae_dict,
            "metadata": metadata,
            "device": model_management.get_torch_device(),
            "dtype": torch.bfloat16 if model_management.should_use_bf16() else torch.float16
        }
        return (model_data,)

class SD35PlusImageEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("SD35PLUS_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "sd35plus add portrait details", "multiline": True}),
                "steps": ("INT", {"default": 30, "min": 10, "max": 150, "step": 1}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "enhance_image"
    CATEGORY = "SD35Plus"
    TITLE = "SD3.5+ Image Enhancer"

    def __init__(self):
        self.transformer = None
        self.vae = None
        self.scheduler = None
        self.current_model_data = None

    def create_transformer_from_state_dict(self, state_dict, device, dtype):
        config = {
            "sample_size": 128, "patch_size": 2, "in_channels": 32, "num_layers": 24,
            "attention_head_dim": 64, "num_attention_heads": 18, "joint_attention_dim": 4096,
            "caption_projection_dim": 1152, "pooled_projection_dim": 2048, "out_channels": 16
        }
        transformer = SD3Transformer2DModel(**config)
        transformer.load_state_dict(state_dict, strict=False)
        return transformer.to(device, dtype=dtype)

    def create_vae_from_state_dict(self, state_dict, device, dtype):
        config = {
            "in_channels": 3, "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D"]*4, "up_block_types": ["UpDecoderBlock2D"]*4,
            "block_out_channels": [128, 256, 512, 512], "layers_per_block": 2, "act_fn": "silu",
            "latent_channels": 16, "norm_num_groups": 32, "sample_size": 1024,
            "scaling_factor": 1.5305, "shift_factor": 0.0609,
        }
        vae = AutoencoderKL(**config)
        vae.load_state_dict(state_dict, strict=False)
        return vae.to(device, dtype=dtype)

    def load_components_if_needed(self, model_data):
        if self.current_model_data is model_data:
            return
            
        device, dtype = model_data["device"], model_data["dtype"]
        
        if not model_data["transformer_dict"]: raise ValueError("Transformer state_dict not found in model file.")
        self.transformer = self.create_transformer_from_state_dict(model_data["transformer_dict"], device, dtype).eval()
        
        if not model_data["vae_dict"]: raise ValueError("VAE state_dict not found in model file.")
        self.vae = self.create_vae_from_state_dict(model_data["vae_dict"], device, dtype).eval()
        
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.current_model_data = model_data

    def encode_dummy_prompt(self, device, dtype):
        prompt_embeds = torch.zeros((1, 77, 4096), device=device, dtype=dtype)
        pooled_embeds = torch.zeros((1, 2048), device=device, dtype=dtype)
        return prompt_embeds, pooled_embeds

    def enhance_image(self, model_data, image, prompt, steps, strength):
        self.load_components_if_needed(model_data)
        device, dtype = model_data["device"], model_data["dtype"]
        
        image_pil = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        
        w, h = image_pil.size
        new_w, new_h = (w - w % 16, h - h % 16)
        if (w,h) != (new_w, new_h):
            image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)

        from torchvision import transforms
        image_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])(image_pil).unsqueeze(0).to(device, dtype=self.vae.dtype)

        with torch.no_grad():
            prompt_embeds, pooled_embeds = self.encode_dummy_prompt(device, dtype)
            
            source_latents = self.vae.encode(image_tensor).latent_dist.sample()
            source_latents = source_latents * self.vae.config.scaling_factor
            
            self.scheduler.set_timesteps(steps, device=device)
            timesteps = self.scheduler.timesteps
            
            start_step = max(0, int(steps * (1.0 - strength)))
            timesteps = timesteps[start_step:]
            
            if not timesteps.numel():
                latents = source_latents
            else:
                start_timestep = timesteps[0]
                noise = torch.randn_like(source_latents)
                latents = self.scheduler.add_noise(source_latents, noise, start_timestep.unsqueeze(0))
            
                pbar = comfy.utils.ProgressBar(len(timesteps))
                for t in timesteps:
                    latent_model_input = torch.cat([latents, source_latents], dim=1)
                    
                    timestep_tensor = t.expand(latent_model_input.shape[0])
                    
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input.to(dtype),
                        timestep=timestep_tensor,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds
                    ).sample
                    
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    pbar.update(1)

            latents = latents / self.vae.config.scaling_factor
            enhanced_image = self.vae.decode(latents.to(self.vae.dtype)).sample
            
            enhanced_image = (enhanced_image / 2 + 0.5).clamp(0, 1)
            enhanced_image = enhanced_image.cpu().permute(0, 2, 3, 1).float()
            
        return (enhanced_image,)

NODE_CLASS_MAPPINGS = {
    "SD35PlusLoader": SD35PlusLoader,
    "SD35PlusImageEnhancer": SD35PlusImageEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD35PlusLoader": "SD3.5+ Loader",
    "SD35PlusImageEnhancer": "SD3.5+ Image Enhancer",
}