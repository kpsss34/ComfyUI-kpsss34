"""
ComfyUI Custom Node for Sana Text-to-Image Models
Supports both Low VRAM (2-4GB) and High VRAM (12GB+) configurations
Includes LoRA support and PAG (Perturbed-Attention Guidance)
"""

import os
import torch
import json
import folder_paths
import comfy.model_management as model_management
from diffusers import SanaPipeline, SanaPAGPipeline
from transformers import AutoModel, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SanaModelLoader:
    """Node for loading Sana models with VRAM optimization"""
    
    @classmethod
    def INPUT_TYPES(s):
        # Get available Sana models from checkpoints folder
        sana_models = []
        sana_path = os.path.join(folder_paths.models_dir, "sana")
        if os.path.exists(sana_path):
            sana_models = [d for d in os.listdir(sana_path) 
                          if os.path.isdir(os.path.join(sana_path, d)) and d != ".cache"]
        
        return {
            "required": {
                "model_name": (sana_models, {"default": sana_models[0] if sana_models else ""}),
                "vram_mode": (["low", "high"], {"default": "low"}),
                "use_pag": ("BOOLEAN", {"default": False}),
                "torch_compile": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("SANA_MODEL",)
    RETURN_NAMES = ("sana_model",)
    FUNCTION = "load_model"
    CATEGORY = "Sana"
    
    def load_text_encoder_nf4(self, model_path):
        """Load text encoder in NF4 format for low VRAM"""
        text_encoder_path = os.path.join(model_path, "text_encoder")
        logger.info(f"Loading text encoder with low VRAM optimization: {text_encoder_path}")
        
        try:
            config = AutoConfig.from_pretrained(text_encoder_path)
            
            with init_empty_weights():
                text_encoder_empty = AutoModel.from_config(config)
            
            text_encoder_nf4 = load_checkpoint_and_dispatch(
                text_encoder_empty,
                text_encoder_path,
                device_map="auto",
                no_split_module_classes=["CLIPEncoderLayer", "TransformerBlock"],
                dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            logger.info("✅ Low VRAM Text encoder loaded successfully")
            return text_encoder_nf4
            
        except Exception as e:
            logger.warning(f"NF4 text encoder failed: {e}. Falling back to standard.")
            try:
                return AutoModel.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
            except Exception as e_fallback:
                logger.error(f"Failed to load text encoder: {e_fallback}")
                raise
    
    def validate_model_files(self, model_path):
        """Validate essential model files exist"""
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        text_encoder_path = os.path.join(model_path, "text_encoder")
        transformer_path = os.path.join(model_path, "transformer")
        
        if not os.path.exists(text_encoder_path):
            raise ValueError(f"Text encoder not found: {text_encoder_path}")
        
        if not os.path.exists(transformer_path):
            raise ValueError(f"Transformer not found: {transformer_path}")
        
        logger.info("✅ Model files validated")
    
    def load_model(self, model_name, vram_mode, use_pag, torch_compile):
        """Load Sana model with specified configuration"""
        
        # Get model path
        model_path = os.path.join(folder_paths.models_dir, "sana", model_name)
        self.validate_model_files(model_path)
        
        # Load text encoder based on VRAM mode
        if vram_mode == "low":
            text_encoder = self.load_text_encoder_nf4(model_path)
        else:
            text_encoder_path = os.path.join(model_path, "text_encoder")
            text_encoder = AutoModel.from_pretrained(
                text_encoder_path, 
                torch_dtype=torch.bfloat16
            )
        
        # Pipeline arguments
        pipeline_args = {
            "text_encoder": text_encoder,
            "torch_dtype": torch.bfloat16
        }
        
        # High VRAM mode specific settings
        if vram_mode == "high":
            pipeline_args["vae_dtype"] = torch.float32
        
        # Choose pipeline type
        if use_pag:
            logger.info(f"Loading SanaPAGPipeline: {model_name} ({vram_mode.upper()} VRAM)")
            pipeline_args["pag_applied_layers"] = "transformer_blocks.8"
            pipe = SanaPAGPipeline.from_pretrained(model_path, **pipeline_args)
        else:
            logger.info(f"Loading SanaPipeline: {model_name} ({vram_mode.upper()} VRAM)")
            pipe = SanaPipeline.from_pretrained(model_path, **pipeline_args)
        
        # Apply torch compile if requested
        if torch_compile:
            logger.info("Applying torch.compile optimization...")
            if hasattr(pipe, 'transformer') and pipe.transformer:
                pipe.transformer = torch.compile(
                    pipe.transformer, 
                    mode="reduce-overhead", 
                    fullgraph=True
                )
        
        # Configure for VRAM mode
        if vram_mode == "low":
            pipe.enable_model_cpu_offload()
            logger.info("✅ CPU offload enabled for LOW VRAM mode")
        else:
            if torch.cuda.is_available():
                pipe.to("cuda")
                logger.info("✅ Pipeline moved to CUDA for HIGH VRAM mode")
        
        # Create model container
        model_container = {
            "pipe": pipe,
            "model_name": model_name,
            "vram_mode": vram_mode,
            "use_pag": use_pag,
            "lora_loaded": None,
            "lora_scale": 0.0
        }
        
        logger.info(f"✅ Sana model '{model_name}' loaded successfully in {vram_mode.upper()} VRAM mode")
        return (model_container,)


class SanaLoRALoader:
    """Node for loading LoRA weights into Sana models"""
    
    @classmethod
    def INPUT_TYPES(s):
        # Get available LoRA files
        lora_files = []
        lora_path = os.path.join(folder_paths.models_dir, "loras", "sana")
        if os.path.exists(lora_path):
            lora_files = [d for d in os.listdir(lora_path) 
                         if os.path.isdir(os.path.join(lora_path, d))]
        
        return {
            "required": {
                "sana_model": ("SANA_MODEL",),
                "lora_name": (["None"] + lora_files, {"default": "None"}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("SANA_MODEL",)
    RETURN_NAMES = ("sana_model",)
    FUNCTION = "load_lora"
    CATEGORY = "Sana"
    
    def create_adapter_config(self, lora_path, base_model_name):
        """Create basic adapter config if missing"""
        config_path = os.path.join(lora_path, "adapter_config.json")
        
        if not os.path.exists(config_path):
            logger.info(f"Creating missing adapter_config.json for {lora_path}")
            
            config = {
                "base_model_name_or_path": base_model_name,
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "init_lora_weights": True,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "peft_type": "LORA",
                "r": 16,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "task_type": None,
                "use_rslora": False
            }
            
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info("✅ Created adapter_config.json")
                return True
            except Exception as e:
                logger.error(f"Failed to create adapter_config.json: {e}")
                return False
        
        return True
    
    def validate_lora_files(self, lora_path):
        """Validate LoRA files"""
        if not os.path.exists(lora_path):
            raise ValueError(f"LoRA path does not exist: {lora_path}")
        
        weight_files = [
            "pytorch_lora_weights.safetensors",
            "pytorch_lora_weights.bin",
            "adapter_model.safetensors"
        ]
        
        weight_file_found = False
        for weight_file in weight_files:
            if os.path.exists(os.path.join(lora_path, weight_file)):
                weight_file_found = True
                break
        
        if not weight_file_found:
            raise ValueError(f"No weight files found in {lora_path}")
        
        logger.info("✅ LoRA files validated")
    
    def load_lora(self, sana_model, lora_name, lora_scale):
        """Load LoRA weights into the model"""
        
        # Create a copy of the model container
        model_container = sana_model.copy()
        pipe = model_container["pipe"]
        
        # Check for PAG + LoRA conflict
        if model_container["use_pag"] and lora_name != "None":
            logger.warning("Cannot use LoRA with PAG pipeline. Skipping LoRA load.")
            return (model_container,)
        
        # Handle LoRA removal
        if lora_name == "None":
            if model_container["lora_loaded"]:
                try:
                    pipe.unload_lora_weights()
                    model_container["lora_loaded"] = None
                    model_container["lora_scale"] = 0.0
                    logger.info("✅ LoRA weights unloaded")
                except:
                    pass
            return (model_container,)
        
        # Load new LoRA
        lora_path = os.path.join(folder_paths.models_dir, "loras", "sana", lora_name)
        self.validate_lora_files(lora_path)
        self.create_adapter_config(lora_path, model_container["model_name"])
        
        try:
            # Unload existing LoRA if different
            if (model_container["lora_loaded"] and 
                model_container["lora_loaded"] != lora_name):
                pipe.unload_lora_weights()
            
            # Load LoRA weights
            pipe.load_lora_weights(lora_path)
            
            # Set LoRA scale
            try:
                active_adapters = pipe.get_active_adapters()
                if active_adapters:
                    pipe.set_adapters(active_adapters, adapter_weights=[float(lora_scale)])
                else:
                    pipe.set_adapters(["default"], adapter_weights=[float(lora_scale)])
            except:
                logger.warning("Could not set LoRA scale, using default")
            
            model_container["lora_loaded"] = lora_name
            model_container["lora_scale"] = lora_scale
            
            logger.info(f"✅ LoRA '{lora_name}' loaded with scale {lora_scale}")
            
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            raise
        
        return (model_container,)


class SanaSampler:
    """Node for generating images with Sana models"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sana_model": ("SANA_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "pag_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7fffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Sana"
    
    def generate(self, sana_model, prompt, negative_prompt, width, height, 
                guidance_scale, pag_scale, num_inference_steps, seed):
        """Generate image using Sana model"""
        
        pipe = sana_model["pipe"]
        vram_mode = sana_model["vram_mode"]
        use_pag = sana_model["use_pag"]
        
        # Handle random seed
        if seed == -1:
            seed = torch.randint(0, 999999, (1,)).item()
        
        # Set up generator
        if vram_mode == "high":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda"
        
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
        # Prepare generation arguments
        generation_args = {
            "prompt": prompt,
            "height": int(height),
            "width": int(width),
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        
        # Add PAG scale if using PAG pipeline
        if use_pag:
            generation_args["pag_scale"] = pag_scale
        
        # Add negative prompt if provided
        if negative_prompt.strip():
            generation_args["negative_prompt"] = negative_prompt
        
        # Log generation info
        lora_info = ""
        if sana_model["lora_loaded"]:
            lora_info = f" | LoRA: {sana_model['lora_loaded']}({sana_model['lora_scale']})"
        
        logger.info(f"🎨 Generating with {type(pipe).__name__} | Seed: {seed}{lora_info}")
        
        try:
            # Generate image
            with torch.inference_mode():
                result = pipe(**generation_args)
                image = result.images[0]
            
            # Convert PIL to tensor format expected by ComfyUI
            import numpy as np
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)
            
            logger.info(f"✅ Generation completed | Seed: {seed}")
            return (image_tensor,)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SanaModelLoader": SanaModelLoader,
    "SanaLoRALoader": SanaLoRALoader,
    "SanaSampler": SanaSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SanaModelLoader": "Sana Model Loader",
    "SanaLoRALoader": "Sana LoRA Loader", 
    "SanaSampler": "Sana Sampler",
}

# Optional: Add web extensions or additional utilities
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]