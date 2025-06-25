# ComfyUI Sana Custom Node

A custom node for ComfyUI that supports Sana text-to-image models (600M/1.6B parameters) with advanced features including LoRA support, PAG (Perturbed-Attention Guidance), and optimized VRAM usage.

## Features

- 🎨 **Sana Model Support**: Compatible with Sana 600M and 1.6B parameter models
- 💾 **VRAM Optimization**: Low VRAM mode (2-4GB) and High VRAM mode (12GB+)
- 🎭 **LoRA Support**: Load and apply LoRA weights with adjustable scaling
- 📐 **PAG Integration**: Perturbed-Attention Guidance for enhanced image quality
- ⚡ **Torch Compile**: Optional model compilation for improved performance
- 🔧 **CPU Offloading**: Automatic memory management for low VRAM systems

## Installation

1. **Clone to ComfyUI custom nodes directory:**
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/ComfyUI-Sana-Node
   cd ComfyUI-Sana-Node
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create model directories:**
   ```bash
   mkdir -p ComfyUI/models/sana
   mkdir -p ComfyUI/models/loras/sana
   ```

## Model Setup

### Download Sana Models

1. Download Sana models from Hugging Face:
   - [Sana 600M](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px)
   - [Sana 1.6B](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px)

2. Place model folders in `ComfyUI/models/sana/`:
   ```
   ComfyUI/models/sana/
   ├── Sana_600M_1024px/
   │   ├── text_encoder/
   │   ├── transformer/
   │   ├── vae/
   │   └── ...
   └── Sana_1600M_1024px/
       ├── text_encoder/
       ├── transformer/
       ├── vae/
       └── ...
   ```

### LoRA Setup (Optional)

Place LoRA files in `ComfyUI/models/loras/sana/`:
```
ComfyUI/models/loras/sana/
├── my_lora_1/
│   ├── pytorch_lora_weights.safetensors
│   └── adapter_config.json
└── my_lora_2/
    ├── pytorch_lora_weights.safetensors
    └── adapter_config.json
```

## Usage

The node package provides three main components:

### 1. Sana Model Loader
- **Purpose**: Load and configure Sana models
- **Options**:
  - `model_name`: Select from available Sana models
  - `vram_mode`: Choose "low" (2-4GB) or "high" (12GB+)
  - `use_pag`: Enable Perturbed-Attention Guidance
  - `torch_compile`: Enable model compilation for performance

### 2. Sana LoRA Loader
- **Purpose**: Apply LoRA weights to loaded models
- **Options**:
  - `lora_name`: Select LoRA or "None"
  - `lora_scale`: Adjust LoRA influence (0.0-2.0)
- **Note**: Cannot be used with PAG simultaneously

### 3. Sana Sampler
- **Purpose**: Generate images using the configured model
- **Options**:
  - `prompt`: Text description for generation
  - `negative_prompt`: What to avoid in generation
  - `width/height`: Image dimensions (512-2048px)
  - `guidance_scale`: Prompt adherence strength (1.0-20.0)
  - `pag_scale`: PAG strength (0.0-10.0, only with PAG enabled)
  - `num_inference_steps`: Generation steps (1-100)
  - `seed`: Random seed (-1 for random)

## Workflow Example

1. **Basic Generation:**
   ```
   Sana Model Loader → Sana Sampler → Preview Image
   ```

2. **With LoRA:**
   ```
   Sana Model Loader → Sana LoRA Loader → Sana Sampler → Preview Image
   ```

3. **Complete Workflow:**
   ```
   Sana Model Loader (use_pag=False) → Sana LoRA Loader → Sana Sampler
   ```

## VRAM Mode Guidelines

### Low VRAM Mode (2-4GB)
- Uses NF4 quantized text encoder
- Enables CPU offloading
- Automatic memory management
- Slower but memory efficient

### High VRAM Mode (12GB+)
- Full precision models
- Everything loaded to GPU
- Faster generation
- Requires sufficient VRAM

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure models are in `ComfyUI/models/sana/`
   - Check folder structure matches Hugging Face format

2. **CUDA Out of Memory**
   - Switch to "low" VRAM mode
   - Reduce image dimensions
   - Close other GPU applications

3. **LoRA Not Loading**
   - Verify LoRA files in `ComfyUI/models/loras/sana/`
   - Check adapter_config.json exists
   - Ensure not using PAG simultaneously

4. **Generation Fails**
   - Check prompt length (avoid extremely long prompts)
   - Verify all dependencies installed
   - Check ComfyUI console for detailed errors

### Performance Tips

- Use `torch_compile=True` for repeated generations
- Batch multiple images when possible
- Keep models loaded between generations
- Use appropriate VRAM mode for your hardware

## Advanced Configuration

### Custom LoRA Configuration

If adapter_config.json is missing, the node will create a basic one. For custom LoRA configurations, create your own:

```json
{
  "base_model_name_or_path": "your_model_name",
  "lora_alpha": 32,
  "lora_dropout": 0.0,
  "r": 16,
  "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
  "peft_type": "LORA"
}
```

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Control GPU usage
- `PYTORCH_CUDA_ALLOC_CONF`: Memory management settings

## License

This project follows the same license as the underlying Sana models and ComfyUI.

## Credits

- Based on the [Sana model architecture](https://github.com/NVlabs/Sana)
- Inspired by the original Gradio implementation
- Built for the ComfyUI ecosystem

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review ComfyUI console logs
3. Open an issue with detailed error information
