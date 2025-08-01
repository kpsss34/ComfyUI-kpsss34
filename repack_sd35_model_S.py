# --- START OF FILE repack_sd35_model_S.py ---

import os
import gc
import json
import torch
from safetensors.torch import save_file
import base64
from pathlib import Path
from huggingface_hub import snapshot_download

# ====================================================================
# CONFIGURATION - แก้ไขค่าต่างๆ ที่นี่
# ====================================================================

# 1. Path ไปยังโมเดลของคุณ
#    - Local Path: "./sd3-finetuned-S"
#    - HF Repo ID: "stabilityai/stable-diffusion-3-medium-diffusers"
MODEL_DIR = "stabilityai/stable-diffusion-3-medium-diffusers"

# 2. Path และชื่อไฟล์ .safetensors ที่ต้องการสร้าง
OUTPUT_PATH = "./repacked_models/sd3-medium-repacked.safetensors"

# 3. ความแม่นยำของตัวเลขที่ต้องการบันทึก ("bf16", "fp16", "fp32")
SAVE_PRECISION = "bf16"

# 4. (ไม่จำเป็น) Hugging Face Token สำหรับดาวน์โหลดโมเดล Private
#    - ใส่ Token ของคุณ (เช่น "hf_xxx...") ถ้าต้องการ
#    - หรือปล่อยเป็น None ถ้าโมเดลเป็น Public
HF_TOKEN = None

# ====================================================================
# (ไม่ต้องแก้ไขโค้ดด้านล่างนี้)
# ====================================================================

def get_full_path(path):
    return str(Path(path).expanduser().resolve())

def repack_model(model_dir: str, output_path: str, save_precision: str = "bf16", hf_token: str = None):
    
    if not os.path.isdir(model_dir):
        print(f"'{model_dir}' is not a local directory. Assuming it's a Hugging Face repo ID.")
        print(f"Attempting to download '{model_dir}'...")
        try:
            # --- แก้ไข: ส่ง hf_token เข้าไปใน snapshot_download ---
            # ถ้า hf_token เป็น None หรือ "" มันจะถูกเมินโดยอัตโนมัติ
            model_dir = snapshot_download(
                repo_id=model_dir,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
                token=hf_token
            )
            print(f"✅ Successfully downloaded to cache: {model_dir}")
        except Exception as e:
            print(f"❌ Failed to download repository '{model_dir}'.")
            print("   - If this is a private or gated model, make sure to provide a valid HF_TOKEN.")
            print(f"   - Error: {e}")
            return
    
    model_dir = get_full_path(model_dir)
    output_path = get_full_path(output_path)
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"Model directory not found: {model_dir}")

    print(f"🚀 Starting repack process for: {model_dir}")
    target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(save_precision, torch.bfloat16)
    print(f"Target save precision: {save_precision} ({target_dtype})")

    state_dict, metadata = {}, {"repacked": "true", "format": "sd35_S_v1"}
    
    components = {
        "transformer": "transformer", "vae": "vae",
        "text_encoder": "text_encoder", "text_encoder_2": "text_encoder_2",
        "scheduler": "scheduler"
    }
    tokenizer_components = {"tokenizer": "tokenizer", "tokenizer_2": "tokenizer_2"}
    
    for name, subfolder in components.items():
        print(f"  -> Processing component: {name}")
        component_path = os.path.join(model_dir, subfolder)
        if not os.path.isdir(component_path):
            print(f"    ⚠️ Component '{name}' not found, skipping.")
            continue
            
        config_path = os.path.join(component_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                metadata[f"{name}_config"] = f.read()
            print(f"    ✅ Config for '{name}' saved.")
        
        safetensors_path = os.path.join(component_path, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(safetensors_path):
            safetensors_path = os.path.join(component_path, "model.safetensors")
        
        if os.path.exists(safetensors_path):
            from safetensors.torch import safe_open
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[f"{name}.{key}"] = f.get_tensor(key).to(dtype=target_dtype)
            print(f"    ✅ Weights for '{name}' loaded.")
        gc.collect()

    for name, subfolder in tokenizer_components.items():
        print(f"  -> Processing tokenizer: {name}")
        tokenizer_path = os.path.join(model_dir, subfolder)
        if not os.path.isdir(tokenizer_path):
            print(f"    ⚠️ Tokenizer '{name}' not found, skipping.")
            continue

        for filename in os.listdir(tokenizer_path):
            if filename.endswith(('.json', '.model', '.txt')):
                full_path = os.path.join(tokenizer_path, filename)
                if filename.endswith('.model'):
                    with open(full_path, 'rb') as f:
                        content = f.read()
                    metadata[f"{name}_{filename}_b64"] = base64.b64encode(content).decode('utf-8')
                else:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    metadata[f"{name}_{filename}"] = content
                print(f"    ✅ Tokenizer file '{filename}' saved.")
            
    print(f"\n💾 Saving repacked model to: {output_path}")
    save_file(state_dict, output_path, metadata=metadata)
    print("\n🎉 Repack process completed successfully!")

if __name__ == "__main__":
    # --- แก้ไข: ส่ง HF_TOKEN เข้าไปในฟังก์ชัน ---
    repack_model(MODEL_DIR, OUTPUT_PATH, SAVE_PRECISION, HF_TOKEN)