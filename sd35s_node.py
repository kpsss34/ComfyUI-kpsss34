import os
import gc
import json
import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management as model_management
import comfy.utils
import tempfile
import inspect
import atexit
import weakref
from diffusers import DiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union
try:
    from safetensors import safe_open
    from safetensors.torch import load_file
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
    from diffusers.models.autoencoders import AutoencoderKL
    from diffusers.models.transformers import SD3Transformer2DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils import scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
    from transformers import (
        CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast,
        CLIPTextConfig, T5Config, PreTrainedModel, PretrainedConfig
    )
    print("SD3.5s Node: All required libraries loaded successfully.")
except ImportError as e:
    print(f"SD3.5 Node ERROR: A required library is missing: {e}")

class MemoryManager:
    """Enhanced memory management for SD3.5s models"""
    _instance = None
    _cached_objects = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Register cleanup on exit
            atexit.register(cls._instance.cleanup_all)
        return cls._instance
    
    def register_object(self, obj):
        """Register object for cleanup tracking"""
        self._cached_objects.append(weakref.ref(obj))
    
    def cleanup_all(self):
        """Comprehensive cleanup of all cached objects and memory"""
        print("SD3.5s: Starting comprehensive memory cleanup...")
        
        # Clear all weak references
        for ref in self._cached_objects:
            obj = ref()
            if obj is not None:
                try:
                    if hasattr(obj, 'to'):
                        obj.to('cpu')
                    del obj
                except:
                    pass
        self._cached_objects.clear()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"CUDA cache cleared. Available VRAM: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated():.2f} MB")
        
        print("SD3.5s: Memory cleanup completed")
    
    def cleanup_torch_memory(self):
        """Specifically clean up PyTorch memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            gc.collect()

# Global memory manager instance
memory_manager = MemoryManager()

class StableDiffusion3SPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, SD3IPAdapterMixin):
    _optional_components = ["text_encoder_3", "tokenizer_3"]

    def __init__(self, transformer, scheduler, vae, text_encoder, tokenizer, text_encoder_2, tokenizer_2, text_encoder_3, tokenizer_3):
        super().__init__()
        self.register_modules(vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, text_encoder_3=text_encoder_3, tokenizer=tokenizer, tokenizer_2=tokenizer_2, tokenizer_3=tokenizer_3, transformer=transformer, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 77
        self.tokenizer_3_max_length = 256
        self.default_sample_size = self.transformer.config.sample_size
        self.patch_size = self.transformer.config.patch_size
        
        # Register with memory manager
        memory_manager.register_object(self)

    def _get_clip_prompt_embeds(self, prompt, num_images_per_prompt, device, clip_model_index):
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]
        tokenizer, text_encoder = clip_tokenizers[clip_model_index], clip_text_encoders[clip_model_index]
        text_inputs = tokenizer(prompt, padding="max_length", max_length=self.tokenizer_max_length, truncation=True, return_tensors="pt")
        outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        return outputs.hidden_states[-2].repeat_interleave(num_images_per_prompt, dim=0), outputs[0].repeat_interleave(num_images_per_prompt, dim=0)

    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, lora_scale):
        if lora_scale is not None:
            for model in [self.transformer, self.text_encoder, self.text_encoder_2]:
                if model and USE_PEFT_BACKEND: scale_lora_layers(model, lora_scale)
        
        prompts = [prompt] if isinstance(prompt, str) else prompt
        
        pos_embeds_1, pos_pooled_1 = self._get_clip_prompt_embeds(prompts, num_images_per_prompt, device, 0)
        pos_embeds_2, pos_pooled_2 = self._get_clip_prompt_embeds(prompts, num_images_per_prompt, device, 1)
        t5_inputs = self.tokenizer_3(prompts, padding="max_length", max_length=self.tokenizer_3_max_length, truncation=True, return_tensors="pt")
        t5_embeds = self.text_encoder_3(t5_inputs.input_ids.to(device))[0].repeat_interleave(num_images_per_prompt, dim=0)
        
        pooled_prompt_embeds = torch.cat([pos_pooled_1, pos_pooled_2], dim=-1)
        clip_embeds = torch.cat([pos_embeds_1, pos_embeds_2], dim=-1)
        
        target_dim = self.transformer.config.joint_attention_dim
        if clip_embeds.shape[-1] < target_dim: clip_embeds = torch.nn.functional.pad(clip_embeds, (0, target_dim - clip_embeds.shape[-1]))
        if t5_embeds.shape[-1] < target_dim: t5_embeds = torch.nn.functional.pad(t5_embeds, (0, target_dim - t5_embeds.shape[-1]))
        prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=1)

        if do_classifier_free_guidance:
            negative_prompts = [negative_prompt or ""] * len(prompts)
            neg_embeds_1, neg_pooled_1 = self._get_clip_prompt_embeds(negative_prompts, num_images_per_prompt, device, 0)
            neg_embeds_2, neg_pooled_2 = self._get_clip_prompt_embeds(negative_prompts, num_images_per_prompt, device, 1)
            t5_inputs_neg = self.tokenizer_3(negative_prompts, padding="max_length", max_length=self.tokenizer_3_max_length, truncation=True, return_tensors="pt")
            t5_embeds_neg = self.text_encoder_3(t5_inputs_neg.input_ids.to(device))[0].repeat_interleave(num_images_per_prompt, dim=0)

            negative_pooled_prompt_embeds = torch.cat([neg_pooled_1, neg_pooled_2], dim=-1)
            neg_clip_embeds = torch.cat([neg_embeds_1, neg_embeds_2], dim=-1)
            if neg_clip_embeds.shape[-1] < target_dim: neg_clip_embeds = torch.nn.functional.pad(neg_clip_embeds, (0, target_dim - neg_clip_embeds.shape[-1]))
            if t5_embeds_neg.shape[-1] < target_dim: t5_embeds_neg = torch.nn.functional.pad(t5_embeds_neg, (0, target_dim - t5_embeds_neg.shape[-1]))
            negative_prompt_embeds = torch.cat([neg_clip_embeds, t5_embeds_neg], dim=1)
        else:
            negative_prompt_embeds, negative_pooled_prompt_embeds = None, None
            
        return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def __call__(self, prompt, height=1024, width=1024, num_inference_steps=28, guidance_scale=7.0, negative_prompt=None, generator=None, callback_on_step_end=None):
        device = self.device
        prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt, device, 1, True, negative_prompt, lora_scale=None)
        
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        
        shape = (1, self.transformer.config.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds
            )[0]
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            if callback_on_step_end is not None: callback_on_step_end(self, i, t, {})
        
        latents /= self.vae.config.scaling_factor if hasattr(self.vae.config, 'scaling_factor') else 0.13025
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = self.image_processor.postprocess(image, output_type="pil")
            
        return StableDiffusion3PipelineOutput(images=image)

class SD35RepackedLoaderSampler:
    _pipeline_cache = {}
    def __init__(self):
        self._pipeline_cache = {}

    @classmethod
    def INPUT_TYPES(s):
        checkpoints_path = folder_paths.get_folder_paths("checkpoints")[0]
        files = [f for f in os.listdir(checkpoints_path) if f.endswith(".safetensors")]
        return { "required": {
                "model_name": (files,),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "bad hands, bad finger, worst quality, low quality, jpeg artifacts, cartoon, painting, doll, ugly, disfigured, deformed, mutated, extra limbs, extra fingers, missing fingers, long neck, bad anatomy, bad proportions, unrealistic face, cloned face, blurred, watermark, text"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "width": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "clear_cache": ("BOOLEAN", {"default": False}),  # เพิ่มตัวเลือกการล้าง cache
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "loaders/sd35s"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs and clear cache if requested"""
        if kwargs.get("clear_cache", False):
            cls.clear_all_caches()
        return True

    @classmethod 
    def clear_all_caches(cls):
        """Clear all pipeline caches and perform memory cleanup"""
        print("SD3.5s: Clearing all caches and cleaning memory...")
        
        # Clear pipeline cache
        for pipe in cls._pipeline_cache.values():
            try:
                if hasattr(pipe, 'to'):
                    pipe.to('cpu')
                del pipe
            except:
                pass
        cls._pipeline_cache.clear()
        
        # Use global memory manager for cleanup
        memory_manager.cleanup_all()
        
        print("SD3.5s: All caches cleared and memory cleaned")

    def _load_pipeline(self, model_path, dtype):
        if model_path in self._pipeline_cache:
            return self._pipeline_cache[model_path]

        print(f"Unpacking and loading SD3.5 model from: {os.path.basename(model_path)}")
        with safe_open(model_path, framework="pt", device="cpu") as f: metadata = f.metadata()
        if metadata is None: raise ValueError(f"Model '{model_path}' missing metadata.")
        state_dict = load_file(model_path, device="cpu")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            def load_component(cls, name):
                config_dict = json.loads(metadata.get(f"{name}_config"))
                if issubclass(cls, PreTrainedModel):
                    config = (CLIPTextConfig if cls is CLIPTextModelWithProjection else T5Config).from_dict(config_dict)
                    model = cls(config)
                else:
                    model = cls.from_config(config_dict)
                
                component_sd = {k.replace(f"{name}.", ""): v for k, v in state_dict.items() if k.startswith(f"{name}.")}
                if name == "text_encoder_3" and "encoder.embed_tokens.weight" not in component_sd and "shared.weight" in component_sd:
                    component_sd["encoder.embed_tokens.weight"] = component_sd["shared.weight"]
                model.load_state_dict(component_sd)
                
                # Register each component with memory manager
                memory_manager.register_object(model)
                return model

            def load_tokenizer(cls, name):
                tokenizer_dir = os.path.join(temp_dir, name)
                os.makedirs(tokenizer_dir, exist_ok=True)
                for key, value in metadata.items():
                    if key.startswith(f"{name}_") and not key.endswith("_b64"):
                        with open(os.path.join(tokenizer_dir, key.replace(f"{name}_", "")), 'w', encoding='utf-8') as f: f.write(value)
                return cls.from_pretrained(tokenizer_dir)
            
            try:
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(json.loads(metadata["scheduler_config"]))
            except:
                scheduler = FlowMatchEulerDiscreteScheduler()

            pipe = StableDiffusion3SPipeline(
                vae=load_component(AutoencoderKL, "vae").to(dtype=dtype),
                transformer=load_component(SD3Transformer2DModel, "transformer").to(dtype=dtype),
                text_encoder=load_component(CLIPTextModelWithProjection, "text_encoder").to(dtype=dtype),
                text_encoder_2=load_component(CLIPTextModelWithProjection, "text_encoder_2").to(dtype=dtype),
                text_encoder_3=load_component(T5EncoderModel, "text_encoder_3").to(dtype=dtype),
                tokenizer=load_tokenizer(CLIPTokenizer, "tokenizer"),
                tokenizer_2=load_tokenizer(CLIPTokenizer, "tokenizer_2"),
                tokenizer_3=load_tokenizer(T5TokenizerFast, "tokenizer_3"),
                scheduler=scheduler,
            )
            
            # Clean up state_dict and force garbage collection
            del state_dict
            gc.collect()
            memory_manager.cleanup_torch_memory()
            
            self._pipeline_cache[model_path] = pipe
            return pipe

    def generate(self, model_name, positive_prompt, negative_prompt, seed, steps, cfg, width, height, clear_cache=False):
        # Clear cache if requested
        if clear_cache:
            self.clear_all_caches()
            
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        device = model_management.get_torch_device()
        dtype = torch.bfloat16 if model_management.should_use_bf16() else torch.float16

        pbar = comfy.utils.ProgressBar(steps)
        def callback(pipe, i, t, kwargs): 
            pbar.update(1)
            # Cleanup intermediate tensors during generation
            if i % 5 == 0:  # Every 5 steps
                memory_manager.cleanup_torch_memory()
            return kwargs

        pipe = self._load_pipeline(model_path, dtype)
        pipe.to(device)
        generator = torch.manual_seed(seed)
        
        try:
            image_pil = pipe(positive_prompt, height, width, steps, cfg, negative_prompt, generator=generator, callback_on_step_end=callback).images[0]
            
            image_np = np.array(image_pil).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

        finally:
            # Always cleanup after generation
            pipe.to("cpu")
            del generator
            gc.collect()
            memory_manager.cleanup_torch_memory()

        return (image_tensor,)

# Hook into ComfyUI's cleanup system
def on_comfyui_reset():
    """Called when ComfyUI resets - clean up all memory"""
    print("ComfyUI Reset detected - cleaning SD3.5s memory...")
    SD35RepackedLoaderSampler.clear_all_caches()
    memory_manager.cleanup_all()

# Try to register with ComfyUI's cleanup system if available
try:
    import comfy.model_management
    # Register cleanup callback if ComfyUI supports it
    if hasattr(comfy.model_management, 'cleanup_models'):
        original_cleanup = comfy.model_management.cleanup_models
        def enhanced_cleanup(*args, **kwargs):
            result = original_cleanup(*args, **kwargs)
            on_comfyui_reset()
            return result
        comfy.model_management.cleanup_models = enhanced_cleanup
        print("SD3.5s: Registered with ComfyUI cleanup system")
except:
    print("SD3.5s: Could not register with ComfyUI cleanup system")

NODE_CLASS_MAPPINGS = { "SD35sLoaderSampler": SD35RepackedLoaderSampler }

NODE_DISPLAY_NAME_MAPPINGS = { "SD35sLoaderSampler": "Load and Sample SD3.5s" }
