import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from pipeline_stable_diffusion_3_S import StableDiffusion3SPipeline

        >>> pipe = StableDiffusion3SPipeline.from_pretrained(
        ...     "./sd3-finetuned-S", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A beautiful woman with a gun in her hand, wearing a bikini."
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3_S_version.png")
        ```
"""

def calculate_shift(
    image_seq_len, base_seq_len: int = 256, max_seq_len: int = 4096, base_shift: float = 0.5, max_shift: float = 1.15
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler, num_inference_steps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None, sigmas: Optional[List[float]] = None, **kwargs,
):
    if timesteps is not None and sigmas is not None: raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class StableDiffusion3SPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, SD3IPAdapterMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->image_encoder->transformer->vae"
    _optional_components = ["text_encoder_3", "tokenizer_3", "image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel, scheduler: FlowMatchEulerDiscreteScheduler, vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection, tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel, tokenizer_3: T5TokenizerFast,
        image_encoder: SiglipVisionModel = None, feature_extractor: SiglipImageProcessor = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, text_encoder_3=text_encoder_3,
            tokenizer=tokenizer, tokenizer_2=tokenizer_2, tokenizer_3=tokenizer_3,
            transformer=transformer, scheduler=scheduler,
            image_encoder=image_encoder, feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        self.tokenizer_3_max_length = 256
        self.default_sample_size = self.transformer.config.sample_size if hasattr(self, "transformer") and self.transformer is not None else 128
        self.patch_size = self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2

    def _get_clip_prompt_embeds(self, prompt, num_images_per_prompt=1, device=None, clip_skip=None, clip_model_index=0):
        device = device or self._execution_device
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]
        tokenizer, text_encoder = clip_tokenizers[clip_model_index], clip_text_encoders[clip_model_index]
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = tokenizer(prompt, padding="max_length", max_length=self.tokenizer_max_length, truncation=True, return_tensors="pt")
        prompt_embeds_output = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds_output[0]
        if clip_skip is None: prompt_embeds = prompt_embeds_output.hidden_states[-2]
        else: prompt_embeds = prompt_embeds_output.hidden_states[-(clip_skip + 2)]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(len(prompt) * num_images_per_prompt, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1).view(len(prompt) * num_images_per_prompt, -1)
        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self, prompt, prompt_2=None, prompt_3=None, device=None, num_images_per_prompt=1, do_classifier_free_guidance=True,
        negative_prompt=None, negative_prompt_2=None, negative_prompt_3=None, prompt_embeds=None, negative_prompt_embeds=None,
        pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None, clip_skip=None, lora_scale=None,
    ):
        device = device or self._execution_device
        
        # [แก้ไข] เพิ่มการ scale LoRA ให้กับ Transformer
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.transformer and USE_PEFT_BACKEND: scale_lora_layers(self.transformer, lora_scale)
            if self.text_encoder and USE_PEFT_BACKEND: scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 and USE_PEFT_BACKEND: scale_lora_layers(self.text_encoder_2, lora_scale)

        if prompt is not None and isinstance(prompt, str): batch_size = 1
        elif prompt is not None and isinstance(prompt, list): batch_size = len(prompt)
        else: batch_size = prompt_embeds.shape[0]

        target_dim = self.transformer.config.joint_attention_dim

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_3 = prompt_3 or prompt
            
            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt, clip_skip=clip_skip, clip_model_index=0)
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(prompt=prompt_2, device=device, num_images_per_prompt=num_images_per_prompt, clip_skip=clip_skip, clip_model_index=1)
            clip_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

            text_inputs_3 = self.tokenizer_3(prompt_3, padding="max_length", max_length=self.tokenizer_3_max_length, truncation=True, return_tensors="pt")
            prompt_embeds_out_3 = self.text_encoder_3(text_inputs_3.input_ids.to(device))
            t5_embeds = prompt_embeds_out_3.last_hidden_state.repeat_interleave(num_images_per_prompt, dim=0)

            if clip_embeds.shape[-1] < target_dim: clip_embeds = torch.nn.functional.pad(clip_embeds, (0, target_dim - clip_embeds.shape[-1]))
            if t5_embeds.shape[-1] < target_dim: t5_embeds = torch.nn.functional.pad(t5_embeds, (0, target_dim - t5_embeds.shape[-1]))
            prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt
            
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            negative_prompt_3 = batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3

            neg_prompt_embed, neg_pooled_prompt_embed = self._get_clip_prompt_embeds(negative_prompt, device=device, num_images_per_prompt=num_images_per_prompt, clip_skip=None, clip_model_index=0)
            neg_prompt_2_embed, neg_pooled_prompt_2_embed = self._get_clip_prompt_embeds(negative_prompt_2, device=device, num_images_per_prompt=num_images_per_prompt, clip_skip=None, clip_model_index=1)
            neg_clip_embeds = torch.cat([neg_prompt_embed, neg_prompt_2_embed], dim=-1)
            negative_pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embed, neg_pooled_prompt_2_embed], dim=-1)

            neg_text_inputs_3 = self.tokenizer_3(negative_prompt_3, padding="max_length", max_length=self.tokenizer_3_max_length, truncation=True, return_tensors="pt")
            neg_prompt_embeds_out_3 = self.text_encoder_3(neg_text_inputs_3.input_ids.to(device))
            neg_t5_embeds = neg_prompt_embeds_out_3.last_hidden_state.repeat_interleave(num_images_per_prompt, dim=0)
            
            if neg_clip_embeds.shape[-1] < target_dim: neg_clip_embeds = torch.nn.functional.pad(neg_clip_embeds, (0, target_dim - neg_clip_embeds.shape[-1]))
            if neg_t5_embeds.shape[-1] < target_dim: neg_t5_embeds = torch.nn.functional.pad(neg_t5_embeds, (0, target_dim - neg_t5_embeds.shape[-1]))
            negative_prompt_embeds = torch.cat([neg_clip_embeds, neg_t5_embeds], dim=1)
        
        # [แก้ไข] เพิ่มการ unscale LoRA ให้กับ Transformer
        if self.transformer and isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND: unscale_lora_layers(self.transformer, lora_scale)
        if self.text_encoder and isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND: unscale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 and isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND: unscale_lora_layers(self.text_encoder_2, lora_scale)
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(self, prompt, prompt_2, prompt_3, height, width, negative_prompt=None, negative_prompt_2=None, negative_prompt_3=None,
        prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None):
        if height % (self.vae_scale_factor * self.patch_size) != 0 or width % (self.vae_scale_factor * self.patch_size) != 0: raise ValueError(f"`height` and `width` must be divisible by {self.vae_scale_factor * self.patch_size}")
        if callback_on_step_end_tensor_inputs is not None and not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs): raise ValueError("`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}")
        if prompt is not None and prompt_embeds is not None: raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None: raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if negative_prompt is not None and negative_prompt_embeds is not None: raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        if prompt_embeds is not None and pooled_prompt_embeds is None: raise ValueError("If `prompt_embeds` are provided, `pooled_prompt_embeds` must also be passed.")
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None: raise ValueError("If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` must also be passed.")

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if latents is not None: return latents.to(device=device, dtype=dtype)
        shape = (batch_size, num_channels_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self): return self._guidance_scale
    @property
    def clip_skip(self): return self._clip_skip
    @property
    def do_classifier_free_guidance(self): return self._guidance_scale > 1
    @property
    def joint_attention_kwargs(self): return self._joint_attention_kwargs
    @property
    def num_timesteps(self): return self._num_timesteps
    @property
    def interrupt(self): return self._interrupt
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self, prompt: Union[str, List[str]] = None, prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None, width: Optional[int] = None, num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None, guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None, negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None, prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None, pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None, output_type: Optional[str] = "pil",
        return_dict: bool = True, joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None, callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"], mu: Optional[float] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Examples:
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt, prompt_2, prompt_3, height, width, negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        batch_size = 1 if isinstance(prompt, str) else len(prompt) if prompt is not None else prompt_embeds.shape[0]
        device = self._execution_device
        
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3,
            negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2, negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, device=device, clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt, lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, h, w = latents.shape
            image_seq_len = (h // self.transformer.config.patch_size) * (w // self.transformer.config.patch_size)
            mu = calculate_shift(image_seq_len)
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
            
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt: continue
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.transformer(
                    hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds, joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent": image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict: return (image,)
        return StableDiffusion3PipelineOutput(images=image)