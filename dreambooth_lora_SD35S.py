import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import math
import itertools
import argparse
import sys

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.models import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from accelerate import Accelerator
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import wandb
from typing import Optional, List, Dict, Any
import logging

try:
    from pipeline_stable_diffusion_3_S import StableDiffusion3SPipeline
    CUSTOM_PIPELINE_AVAILABLE = True
    print("Successfully imported StableDiffusion3SPipeline from local folder")
except ImportError as e:
    print(f"Warning: Could not import custom pipeline: {e}")
    print("Validation will be skipped or use default StableDiffusion3Pipeline")
    CUSTOM_PIPELINE_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="kpsss34/Stable-Diffusion-3.5-Small-Preview1", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--instance_data_dir", type=str, default="./datasets", help="A folder containing the training data of instance images and caption.txt.")
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--instance_prompt", type=str, default="a photo of ... ", help="The prompt with identifier specifying the instance")
    parser.add_argument("--rank", type=int, default=64, help="The rank of the LoRA model, 16 32 64 128")
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    parser.add_argument("--text_encoder_lr", type=float, default=5e-6, help="The learning rate for the text encoder.")
    parser.add_argument("--max_train_steps", type=int, default=500, help="Total number of training steps to perform.")
    parser.add_argument("--resolution", type=int, default=1024, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Number of steps for the warmup in the lr scheduler. If not set, will be 10% of max_train_steps.")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"], help="The weighting scheme to use for loss computation.")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="The mean to use when using the `logit_normal` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="The std to use when using the `logit_normal` weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="The scale to use when using the `mode` weighting scheme.")
    parser.add_argument("--precondition_outputs", action="store_true", default=True, help="Whether to precondition the outputs with the noisy input.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory.")
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "comet_ml", "all"], help="The integration to report the results and logs to.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop the input images.")
    parser.add_argument("--validation_prompt", type=str, default="a photo of ...", help="A prompt for validation.")
    parser.add_argument("--validation_epochs", type=int, default=1, help="Run validation every X epochs.")
    parser.add_argument("--num_validation_images", type=int, default=1, help="Number of images to generate during validation.")
    parser.add_argument("--validation_steps", type=int, default=50, help="Run validation every X steps. Overrides validation_epochs.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether to use 8-bit AdamW optimizer from bitsandbytes.")
    
    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_private_repo", action="store_true", help="Whether the model repository is private.")
    
    # Checkpoint management
    parser.add_argument("--checkpoints_total_limit", type=int, default=2, help="Max number of checkpoints to store. Older checkpoints are deleted.")
    parser.add_argument("--checkpointing_steps", type=int, default=None, help="Save a checkpoint of the training state every X updates. Overrides validation_steps for checkpointing.")

    args = parser.parse_args()
    if args.warmup_steps is None:
        args.warmup_steps = int(args.max_train_steps * 0.10)
    return args

class DreamBoothLoRA3Dataset(Dataset):
    def __init__(self, data_dir: str, instance_prompt: str, size: int = 1024, center_crop: bool = False):
        self.data_dir = Path(data_dir)
        self.instance_prompt = instance_prompt
        self.size = size
        self.center_crop = center_crop
        self.image_paths = []
        self.caption_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for img_path in self.data_dir.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists():
                    self.image_paths.append(img_path)
                    self.caption_paths.append(txt_path)
        self.num_instance_images = len(self.image_paths)
        self._length = self.num_instance_images
        print(f"Found {self.num_instance_images} image-caption pairs.")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        image_path = self.image_paths[index % self.num_instance_images]
        image = Image.open(image_path).convert('RGB')
        pixel_values = self._preprocess_image(image)
        caption_path = self.caption_paths[index % self.num_instance_images]
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        final_prompt = f"{self.instance_prompt.strip()}, {caption}"
        return {"pixel_values": pixel_values, "prompt": final_prompt}

    def _preprocess_image(self, image):
        w, h = image.size
        if self.center_crop:
            if w < h: w_new, h_new = self.size, int(h * self.size / w)
            else: h_new, w_new = self.size, int(w * self.size / h)
            image = image.resize((w_new, h_new), Image.LANCZOS)
            left, top = (w_new - self.size) // 2, (h_new - self.size) // 2
            image = image.crop((left, top, left + self.size, top + self.size))
        else:
            if w > h: new_w, new_h = self.size, int(h * self.size / w)
            else: new_h, new_w = self.size, int(w * self.size / h)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            new_image = Image.new("RGB", (self.size, self.size), (128, 128, 128))
            paste_x, paste_y = (self.size - new_w) // 2, (self.size - new_h) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        return torch.from_numpy(image).permute(2, 0, 1)

def load_models(args):
    logging.info(f"Loading base model for LoRA training from: {args.pretrained_model_name_or_path}")
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    tokenizer_three = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3")
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    text_encoder_three = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_3")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    transformer = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    return locals()

def add_lora_adapters(models, args):
    transformer_lora_config = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"])
    models["transformer"].add_adapter(transformer_lora_config)
    logging.info("Added LoRA adapter to Transformer.")
    if args.train_text_encoder:
        text_lora_config = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        models["text_encoder_one"].add_adapter(text_lora_config)
        models["text_encoder_two"].add_adapter(text_lora_config)
        logging.info("Added LoRA adapters to Text Encoders 1 & 2.")

def encode_prompt_sd3(models, prompts, device):
    prompt_embeds_list, pooled_embeds_list = [], []
    for caption in prompts:
        tok_out_1 = models["tokenizer_one"](caption, padding="max_length", max_length=models["tokenizer_one"].model_max_length, truncation=True, return_tensors="pt")
        text_encoder_output_1 = models["text_encoder_one"](tok_out_1.input_ids.to(device), output_hidden_states=True)
        clip_l_embeds, clip_l_pooled = text_encoder_output_1.hidden_states[-2], text_encoder_output_1[0]
        tok_out_2 = models["tokenizer_two"](caption, padding="max_length", max_length=models["tokenizer_two"].model_max_length, truncation=True, return_tensors="pt")
        text_encoder_output_2 = models["text_encoder_two"](tok_out_2.input_ids.to(device), output_hidden_states=True)
        clip_g_embeds, clip_g_pooled = text_encoder_output_2.hidden_states[-2], text_encoder_output_2[0]
        tok_out_3 = models["tokenizer_three"](caption, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        t5_embeds = models["text_encoder_three"](tok_out_3.input_ids.to(device)).last_hidden_state
        clip_combined = torch.cat([clip_l_embeds, clip_g_embeds], dim=-1)
        pooled_combined = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)
        target_dim = models["transformer"].config.joint_attention_dim
        if clip_combined.shape[-1] < target_dim: clip_combined = F.pad(clip_combined, (0, target_dim - clip_combined.shape[-1]))
        if t5_embeds.shape[-1] < target_dim: t5_embeds = F.pad(t5_embeds, (0, target_dim - t5_embeds.shape[-1]))
        prompt_embeds = torch.cat([clip_combined, t5_embeds], dim=1)
        prompt_embeds_list.append(prompt_embeds)
        pooled_embeds_list.append(pooled_combined)
    return torch.cat(prompt_embeds_list, dim=0), torch.cat(pooled_embeds_list, dim=0)

def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim: sigma = sigma.unsqueeze(-1)
    return sigma

def run_validation(args, accelerator, checkpoint_dir, global_step, epoch=None):
    if not accelerator.is_main_process:
        return

    logging.info(f"Running validation at step {global_step} using checkpoint: {checkpoint_dir}")
    
    weight_dtype = torch.bfloat16 if args.mixed_precision == 'bf16' else (torch.float16 if args.mixed_precision == 'fp16' else torch.float32)
    
    torch.cuda.empty_cache()
    
    if CUSTOM_PIPELINE_AVAILABLE:
        pipeline_class = StableDiffusion3SPipeline
    else:
        pipeline_class = StableDiffusion3Pipeline

    pipeline = None
    try:
        pipeline = pipeline_class.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=weight_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            pipeline.load_lora_weights(checkpoint_dir)
            logging.info(f"Loaded LoRA weights from {checkpoint_dir}")
        else:
            logging.warning(f"No LoRA weights found in {checkpoint_dir}, using base model")
        
        pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_model_cpu_offload'):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(accelerator.device)
        
        pipeline.scheduler.set_timesteps(28, device=accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        validation_dir = os.path.join(args.output_dir, "validation_images")
        os.makedirs(validation_dir, exist_ok=True)
        
        base_seed = args.seed if args.seed else 42
        generator = torch.Generator(device=accelerator.device).manual_seed(base_seed + global_step)
        
        images = []
        for i in range(args.num_validation_images):
            torch.cuda.empty_cache()
            
            with torch.inference_mode():
                try:
                    image = pipeline(
                        args.validation_prompt,
                        num_inference_steps=30,
                        generator=generator,
                        height=args.resolution,
                        width=args.resolution,
                        guidance_scale=5.0,
                        lora_scale=1.0
                    ).images[0]
                    
                    img_array = np.array(image)
                    if img_array.mean() < 10:
                        logging.warning(f"Generated image {i} appears to be too dark, retrying with different settings")
                        image = pipeline(
                            args.validation_prompt,
                            num_inference_steps=50,
                            generator=generator,
                            height=args.resolution,
                            width=args.resolution,
                            guidance_scale=10.0,
                            lora_scale=1.0
                        ).images[0]
                    
                    images.append(image)
                    logging.info(f"Successfully generated validation image {i+1}/{args.num_validation_images}")
                    
                except Exception as e:
                    logging.error(f"Error generating validation image {i}: {e}")
                    placeholder = Image.new('RGB', (args.resolution, args.resolution), (128, 128, 128))
                    images.append(placeholder)
        
        step_str = f"step_{global_step:06d}" if epoch is None else f"epoch_{epoch:03d}_step_{global_step:06d}"
        for i, image in enumerate(images):
            image_path = os.path.join(validation_dir, f"{step_str}_image_{i:02d}.png")
            image.save(image_path)
        
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                try:
                    tracker.log({
                        "validation": [wandb.Image(img, caption=f"{i}: {args.validation_prompt}") for i, img in enumerate(images)],
                        "validation_step": global_step
                    }, step=global_step)
                except Exception as e:
                    logging.error(f"Error logging to wandb: {e}")

        logging.info(f"Saved validation images to {validation_dir}")

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline is not None:
            del pipeline
        torch.cuda.empty_cache()

def save_lora_weights(save_directory, accelerator, models, args):
    if accelerator.is_main_process:
        os.makedirs(save_directory, exist_ok=True)
        transformer = accelerator.unwrap_model(models["transformer"])
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        text_encoder_lora_layers = None
        text_encoder_2_lora_layers = None
        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(models["text_encoder_one"])
            text_encoder_two = accelerator.unwrap_model(models["text_encoder_two"])
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one)
            text_encoder_2_lora_layers = get_peft_model_state_dict(text_encoder_two)
        
        if transformer_lora_layers:
            StableDiffusion3Pipeline.save_lora_weights(
                save_directory=save_directory, 
                transformer_lora_layers=transformer_lora_layers, 
                text_encoder_lora_layers=text_encoder_lora_layers, 
                text_encoder_2_lora_layers=text_encoder_2_lora_layers
            )
            logging.info(f"LoRA weights saved to {save_directory}")
            
            if args.checkpoints_total_limit is not None and save_directory != args.output_dir:
                manage_checkpoints(args.output_dir, args.checkpoints_total_limit)
        else:
            logging.warning("No LoRA weights to save!")

def manage_checkpoints(output_dir, total_limit):
    """Manage checkpoint directories to keep only the most recent ones"""
    if total_limit is None:
        return
    
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
            try:
                step_num = int(item.split("-")[-1])
                checkpoint_dirs.append((step_num, os.path.join(output_dir, item)))
            except ValueError:
                continue
    
    if len(checkpoint_dirs) > total_limit:
        checkpoint_dirs.sort(key=lambda x: x[0])
        dirs_to_remove = checkpoint_dirs[:-total_limit]
        
        for _, dir_path in dirs_to_remove:
            try:
                import shutil
                shutil.rmtree(dir_path)
                logging.info(f"Removed old checkpoint: {dir_path}")
            except Exception as e:
                logging.warning(f"Failed to remove checkpoint {dir_path}: {e}")

def push_to_hub_if_needed(args, save_directory, accelerator):
    """Push model to Hugging Face Hub if requested"""
    if not args.push_to_hub:
        return
        
    if not accelerator.is_main_process:
        logging.info("Skipping Hub push - not main process")
        return

    try:
        logging.info("Starting Hub upload process...")
        
        # Check if we have required dependencies
        try:
            from huggingface_hub import HfApi, create_repo, HfFolder
            logging.info("huggingface_hub imported successfully")
        except ImportError as e:
            raise ImportError(f"huggingface_hub not available: {e}. Install with: pip install huggingface_hub")
        
        # Get and validate token
        token = args.hub_token
        if token is None:
            try:
                token = HfFolder.get_token()
                if token is None:
                    raise ValueError("No Hub token found. Please login using `huggingface-cli login` or provide --hub_token")
                logging.info("Hub token found from local storage")
            except Exception as e:
                raise ValueError(f"Error getting Hub token: {e}")
        else:
            logging.info("Hub token provided via argument")
        
        # Get and validate hub model id
        hub_model_id = args.hub_model_id
        if hub_model_id is None:
            hub_model_id = f"dreambooth-lora-{args.instance_prompt.replace(' ', '-').replace(',', '').lower()}"
            logging.info(f"No hub_model_id provided, using: {hub_model_id}")
        else:
            logging.info(f"Using hub_model_id: {hub_model_id}")
        
        api = HfApi()
        
        # Test API connection
        try:
            user_info = api.whoami(token=token)
            logging.info(f"Connected to Hub as: {user_info['name']}")
        except Exception as e:
            raise ValueError(f"Failed to connect to Hub: {e}")
        
        # Check if save directory exists and has files
        if not os.path.exists(save_directory):
            raise ValueError(f"Save directory does not exist: {save_directory}")
        
        files_in_dir = os.listdir(save_directory)
        if not files_in_dir:
            raise ValueError(f"Save directory is empty: {save_directory}")
        
        logging.info(f"Found {len(files_in_dir)} files in save directory: {files_in_dir}")
        
        # Create model card content
        model_card_content = f"""---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
instance_prompt: {args.instance_prompt}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
- dreambooth
inference: true
---

# LoRA DreamBooth model

These are LoRA adaption weights for {args.pretrained_model_name_or_path}. The weights were trained on the concept of `{args.instance_prompt}` using [DreamBooth](https://dreambooth.github.io/).

## Training details
- **Base model**: {args.pretrained_model_name_or_path}
- **Instance prompt**: {args.instance_prompt}
- **Validation prompt**: {args.validation_prompt}
- **LoRA rank**: {args.rank}
- **Learning rate**: {args.learning_rate}
- **Training steps**: {args.max_train_steps}
- **Resolution**: {args.resolution}
- **Train text encoder**: {args.train_text_encoder}

## Usage

```python
import torch
from pipeline_stable_diffusion_3_S import StableDiffusion3SPipeline

model_id = "{args.pretrained_model_name_or_path}"
pipe = StableDiffusion3SPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.load_lora_weights("{hub_model_id}")
pipe.to("cuda")

# Now you can use the pipeline with the trained LoRA
image = pipe("{args.instance_prompt}", num_inference_steps=30, guidance_scale=5.0).images[0]
image.save("result.png")
```
"""
        
        # Save model card
        model_card_path = os.path.join(save_directory, "README.md")
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)
        logging.info("Model card created")
        
        # Create repository if it doesn't exist
        try:
            logging.info(f"Creating/checking repository: {hub_model_id}")
            repo_url = create_repo(
                repo_id=hub_model_id,
                token=token,
                private=args.hub_private_repo,
                exist_ok=True
            )
            logging.info(f"Repository ready: {repo_url}")
        except Exception as e:
            logging.error(f"Failed to create repository: {e}")
            raise
        
        # Upload files
        try:
            logging.info(f"Starting file upload to {hub_model_id}...")
            api.upload_folder(
                folder_path=save_directory,
                repo_id=hub_model_id,
                token=token,
                commit_message=f"Upload LoRA weights trained with DreamBooth",
                ignore_patterns=["*.git*", "validation_images/", "*.log"]
            )
            logging.info(f"Successfully pushed model to: https://huggingface.co/{hub_model_id}")
            
            # Save hub info to local file
            hub_info = {
                "hub_model_id": hub_model_id,
                "hub_url": f"https://huggingface.co/{hub_model_id}",
                "pushed_at": str(torch.distributed.get_rank() if torch.distributed.is_initialized() else "single_process"),
                "base_model": args.pretrained_model_name_or_path,
                "instance_prompt": args.instance_prompt
            }
            with open(os.path.join(save_directory, "hub_info.json"), "w") as f:
                json.dump(hub_info, f, indent=2)
            logging.info("Hub info saved to hub_info.json")
                
        except Exception as e:
            logging.error(f"Failed to upload files: {e}")
            raise
            
    except Exception as e:
        logging.error(f"Error during Hub upload: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        # Don't raise here to avoid stopping the training completely
        logging.error("Hub upload failed, but training completed successfully")

def weighted_loss(model_pred, target, weighting):
    return torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()

def setup_training(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        mixed_precision=args.mixed_precision, 
        log_with=args.report_to, 
        project_dir=args.logging_dir
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    models = load_models(args)
    models["vae"].requires_grad_(False)
    models["text_encoder_one"].requires_grad_(False)
    models["text_encoder_two"].requires_grad_(False)
    models["text_encoder_three"].requires_grad_(False)
    models["transformer"].requires_grad_(False)
    
    add_lora_adapters(models, args)
    
    dataset = DreamBoothLoRA3Dataset(
        data_dir=args.instance_data_dir, 
        instance_prompt=args.instance_prompt, 
        size=args.resolution, 
        center_crop=args.center_crop
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    
    params_to_optimize = []
    transformer_lora_params = list(filter(lambda p: p.requires_grad, models["transformer"].parameters()))
    params_to_optimize.append({"params": transformer_lora_params, "lr": args.learning_rate})
    logging.info(f"Found {len(transformer_lora_params)} trainable LoRA parameters in Transformer.")
    
    if args.train_text_encoder:
        te1_lora_params = list(filter(lambda p: p.requires_grad, models["text_encoder_one"].parameters()))
        te2_lora_params = list(filter(lambda p: p.requires_grad, models["text_encoder_two"].parameters()))
        params_to_optimize.append({"params": te1_lora_params, "lr": args.text_encoder_lr or args.learning_rate})
        params_to_optimize.append({"params": te2_lora_params, "lr": args.text_encoder_lr or args.learning_rate})
        logging.info(f"Found {len(te1_lora_params) + len(te2_lora_params)} trainable LoRA parameters in Text Encoders.")
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes.optim as bnb_optim
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`")
        optimizer_class = bnb_optim.AdamW8bit
        logging.info("Using 8-bit AdamW optimizer.")
    else:
        optimizer_class = torch.optim.AdamW
        logging.info("Using standard AdamW optimizer.")
    
    optimizer = optimizer_class(
        params_to_optimize, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.adam_weight_decay, 
        eps=args.adam_epsilon
    )
    
    lr_scheduler = get_scheduler(
        "constant_with_warmup", 
        optimizer=optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=args.max_train_steps
    )
    
    models_to_prepare = [models["transformer"]]
    if args.train_text_encoder:
        models_to_prepare.extend([models["text_encoder_one"], models["text_encoder_two"]])
    
    prepared = accelerator.prepare(*models_to_prepare, optimizer, dataloader, lr_scheduler)
    unpacked_models = prepared[:len(models_to_prepare)]
    optimizer, dataloader, lr_scheduler = prepared[len(models_to_prepare):]
    
    idx = 0
    models["transformer"] = unpacked_models[idx]; idx += 1
    if args.train_text_encoder:
        models["text_encoder_one"] = unpacked_models[idx]; idx += 1
        models["text_encoder_two"] = unpacked_models[idx]; idx += 1
    
    weight_dtype = torch.bfloat16 if args.mixed_precision == 'bf16' else (torch.float16 if args.mixed_precision == 'fp16' else torch.float32)
    
    models["vae"].to(accelerator.device, dtype=weight_dtype).eval()
    models["vae"].enable_slicing()
    models["text_encoder_three"].to(accelerator.device, dtype=weight_dtype).eval()
    
    if not args.train_text_encoder:
        models["text_encoder_one"].to(accelerator.device, dtype=weight_dtype).eval()
        models["text_encoder_two"].to(accelerator.device, dtype=weight_dtype).eval()
    else:
        logging.info("Setting Text Encoders to float32 for training stability")
        
        models["text_encoder_one"] = models["text_encoder_one"].float()
        models["text_encoder_two"] = models["text_encoder_two"].float()
    
    if args.gradient_checkpointing:
        models["transformer"].enable_gradient_checkpointing()
        if args.train_text_encoder:
            models["text_encoder_one"].gradient_checkpointing_enable()
            models["text_encoder_two"].gradient_checkpointing_enable()
    
    return accelerator, models, dataloader, optimizer, lr_scheduler

def train(args):
    accelerator, models, dataloader, optimizer, lr_scheduler = setup_training(args)
    device = accelerator.device
    
    weight_dtype = torch.bfloat16 if args.mixed_precision == 'bf16' else (
        torch.float16 if args.mixed_precision == 'fp16' else torch.float32
    )
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    model_to_accumulate = models["transformer"]
    noise_scheduler_copy = models["noise_scheduler"]
    
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    should_validate_by_steps = args.validation_steps is not None
    should_validate_by_epochs = args.validation_epochs is not None and not should_validate_by_steps
    should_checkpoint_by_steps = args.checkpointing_steps is not None

    for epoch in range(num_train_epochs):
        models["transformer"].train()
        if args.train_text_encoder:
            models["text_encoder_one"].train()
            models["text_encoder_two"].train()
            
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model_to_accumulate):
                # VAE encoding
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(device=device, dtype=models["vae"].dtype)
                    latents = models["vae"].encode(pixel_values).latent_dist.sample()
                    latents = latents * models["vae"].config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                # Noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, 
                    batch_size=bsz, 
                    logit_mean=args.logit_mean, 
                    logit_std=args.logit_std, 
                    mode_scale=args.mode_scale
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)
                sigmas = get_sigmas(noise_scheduler_copy, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = latents if args.precondition_outputs else (noise - latents)

                if args.train_text_encoder:
                    prompt_embeds, pooled_prompt_embeds = encode_prompt_sd3(models, batch["prompt"], device)
                else:
                    with torch.no_grad():
                        prompt_embeds, pooled_prompt_embeds = encode_prompt_sd3(models, batch["prompt"], device)
                
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                model_pred = models["transformer"](
                    hidden_states=noisy_latents, 
                    timestep=timesteps, 
                    encoder_hidden_states=prompt_embeds, 
                    pooled_projections=pooled_prompt_embeds
                )[0]
                
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_latents

                # Loss computation
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = weighted_loss(model_pred, target, weighting)
                
                accelerator.backward(loss)

                # Gradient clipping
                params_to_clip = list(filter(lambda p: p.requires_grad, itertools.chain(
                    models["transformer"].parameters(),
                    (models["text_encoder_one"].parameters() if args.train_text_encoder else []),
                    (models["text_encoder_two"].parameters() if args.train_text_encoder else [])
                )))
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # Save checkpoints by steps (independent of validation)
                if should_checkpoint_by_steps and global_step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_lora_weights(checkpoint_dir, accelerator, models, args)
                    accelerator.wait_for_everyone()
                
                # Run validation by steps
                if should_validate_by_steps and global_step % args.validation_steps == 0:
                    accelerator.wait_for_everyone()
                    # Use existing checkpoint if just saved, otherwise create validation checkpoint
                    if should_checkpoint_by_steps and global_step % args.checkpointing_steps == 0:
                        validation_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    else:
                        validation_checkpoint_dir = os.path.join(args.output_dir, f"validation-checkpoint-{global_step}")
                        save_lora_weights(validation_checkpoint_dir, accelerator, models, args)
                    accelerator.wait_for_everyone()
                    run_validation(args, accelerator, validation_checkpoint_dir, global_step)
                    
                    # Clean up validation-only checkpoint
                    if not (should_checkpoint_by_steps and global_step % args.checkpointing_steps == 0):
                        try:
                            import shutil
                            if os.path.exists(validation_checkpoint_dir):
                                shutil.rmtree(validation_checkpoint_dir)
                                logging.info(f"Cleaned up validation checkpoint: {validation_checkpoint_dir}")
                        except Exception as e:
                            logging.warning(f"Failed to clean up validation checkpoint: {e}")

            if global_step >= args.max_train_steps:
                break
        
        # Epoch-based validation
        if should_validate_by_epochs and (epoch + 1) % args.validation_epochs == 0:
            accelerator.wait_for_everyone()
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{(epoch+1)}")
            save_lora_weights(checkpoint_dir, accelerator, models, args)
            accelerator.wait_for_everyone()
            run_validation(args, accelerator, checkpoint_dir, global_step, epoch=epoch+1)

        if global_step >= args.max_train_steps:
            break

    # Save final model
    accelerator.wait_for_everyone()
    save_lora_weights(args.output_dir, accelerator, models, args)
    
    # Push to Hub if requested
    if args.push_to_hub:
        push_to_hub_if_needed(args, args.output_dir, accelerator)
    
    accelerator.end_training()

def main():
    args = parse_args()
    
    # Validate Hub arguments
    if args.push_to_hub:
        if args.hub_model_id is None:
            logging.warning("--push_to_hub is set but --hub_model_id is not provided. Will generate default name.")
        
        # Check if user is logged in to Hub
        try:
            from huggingface_hub import HfFolder
            token = args.hub_token or HfFolder.get_token()
            if token is None:
                raise ValueError("To push to Hub, you need to login first. Run `huggingface-cli login` or provide --hub_token")
        except ImportError:
            raise ImportError("To push to Hub, please install huggingface_hub: `pip install huggingface_hub`")
    
    if args.validation_prompt is None and (args.validation_steps is not None or args.validation_epochs is not None):
        args.validation_prompt = f"{args.instance_prompt.strip()}, high quality, detailed"
        logging.info(f"No validation prompt provided, using default: {args.validation_prompt}")
    if not os.path.exists(args.instance_data_dir):
        print(f"Instance data directory '{args.instance_data_dir}' not found. Creating a dummy dataset.")
        os.makedirs(args.instance_data_dir, exist_ok=True)
        dummy_image = Image.new('RGB', (args.resolution, args.resolution), 'blue')
        dummy_image.save(os.path.join(args.instance_data_dir, 'dummy_img.jpg'), 'JPEG')
        with open(os.path.join(args.instance_data_dir, 'dummy_img.txt'), 'w') as f:
            f.write('a blue square image')
        print("Dummy dataset created.")
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()