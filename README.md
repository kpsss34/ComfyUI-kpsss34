# ComfyUI Sana Custom Node

## Support my GPU rental fee for Finetune & Request the desired model

https://coff.ee/kpsss34

A custom node for ComfyUI that supports Sana text-to-image models (600M/1.6B parameters) with advanced features including LoRA support, PAG (Perturbed-Attention Guidance), and optimized VRAM usage.

## Installation

1. **Clone to ComfyUI custom nodes directory:**
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/kpsss34/ComfyUI-kpsss34-Sana.git
   cd ComfyUI-kpsss34-Sana
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
   - [FINETUNE MODELS](https://huggingface.co/kpsss34)
  
1.1 How to donwload

   Ex.repo [kpsss34/SANA600.fp8_illustrious_SFW_V1]
   
   in root/ComfyUI/models/sana
   
   - git clone https://huggingface.co/kpsss34/SANA600.fp8_illustrious_SFW_V1

2. Place model folders in `ComfyUI/models/sana/`:
   ```
   ComfyUI/models/sana/
   ├── SANA600.fp8_illustrious_SFW_V1/
   │   ├── text_encoder/
   │   ├── transformer/
   │   ├── vae/
   │   └── ...
   └── SANA600.fp8_illustrious_SFW_V2/
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
**Note**: Cannot be used with PAG simultaneously

### 3. Sana Sampler
- **Purpose**: Generate images using the configured model
- **Options**:
  - `prompt`: Text description for generation
  - `negative_prompt`: What to avoid in generation
  - `width/height`: Image dimensions (512px-2048px)
  - `guidance_scale`: Prompt adherence strength (3.0-7.0)
  - `pag_scale`: PAG strength (0.0-10.0, only with PAG enabled)
  **Note**:Cannot be used PAG with LoRA
  - `num_inference_steps`: Generation steps (1-100)
  - `seed`: Random seed (-1 for random)

## Workflow Example

![Screenshot (23)](https://github.com/user-attachments/assets/119288c6-ef4f-49d7-8869-cdb5b6d9f2cc)

