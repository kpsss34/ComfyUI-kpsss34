# ComfyUI Custom Node

## Support me

https://coff.ee/kpsss34

### ComfyUI inference Supprot ###

I have completed a ComfyUI custom_node specifically designed to work with a direct pipeline implementation. Please note that this node is not compatible with standard ComfyUI nodes. If you wish to use the models I’ve trained, it is recommended to use this custom node for image generation.

Download checkpoints to ComfyUI/models/checkpoints

LINK: https://huggingface.co/kpsss34/Stable-Diffusion-3.5-Small-Preview1/resolve/main/SD35sPreview1_Custom-nodes.safetensors

### Ez TO USE ###
1.Install my custom node via the ComfyUI Manager.

<img width="1409" height="260" alt="Screenshot 2025-08-05 151706" src="https://github.com/user-attachments/assets/30c506f2-8228-49e8-8555-1b0aaf037708" />

2.Right-click to open the main menu, then go to [ Loaders → sd35s → Load and Sample SD35s.]

<img width="676" height="676" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/00da6a0e-7210-4b84-b823-cccab19ee9a8" />

3.Select a Checkpoint and configure the settings as needed. Then, connect my custom node to a Save Image or Preview Image node.

<img width="1270" height="790" alt="Screenshot 2025-08-05 151028" src="https://github.com/user-attachments/assets/4a4f54b4-0069-4652-9e2e-d1c2495f908c" />

Notes:

1.The model I repacked is not compatible with standard ComfyUI nodes. It only works with this custom node.

2.Unpacking the model into RAM may take a while on the first run — typically around 1–2 minutes, depending on your system.

<img width="319" height="68" alt="Screenshot 2025-08-05 150954" src="https://github.com/user-attachments/assets/adc7082d-62f0-4a8b-abd5-2cab91d54198" />

3.The model requires approximately 7–9GB of VRAM.


### LORA TRAINIG ###

Note: LoRA is not yet supported in ComfyUI.(Only Diffusers use)
If I have time, I’ll work on adding support.
If you’re able to help with this, I’d greatly appreciate it!

Install the required dependencies:
```python
git clone https://github.com/kpsss34/ComfyUI-kpsss34.git
cd ComfyUI-kpsss34
python -m venv venv
source venv/bin/activate
pip install -r requirements_lora.txt
pip install -U peft
```
# On Windows use: venv\Scripts\activate

- Basic argument for run

```python

accelerate launch dreambooth_lora_SD35S.py \
--pretrained_model_name_or_path="kpsss34/Stable-Diffusion-3.5-Small-Preview1" \
--train_text_encoder \
--output_dir="./lora_output" \
--instance_data_dir="./datasets" \
--rank=32 \
--instance_prompt="a photo of ... " \
--max_train_steps=1000 \
--warmup_steps=100 \
--use_8bit_adam \
--resolution=768 \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--learning_rate=1e-4 \
--gradient_checkpointing \
--mixed_precision="bf16" \
--validation_prompt="a photo of ... " \
--validation_steps=50 \
--num_validation_images=1 \
--push_to_hub \
--hub_model_id=""

```

⚠️ --instance_data_dir= (this directory should contain both image.jpg and image.txt)

⚠️ Each image must have a corresponding caption file (.txt with the same filename).

⚠️The file pipeline_stable_diffusion_3_S.py must be located in the same directory at all times.


LoRA Training Recommendations:

(1024px)I recommend 18-24GB of VRAM if you're including the validation step.

(1024px)If you skip validation, you can train with 16GB of VRAM.

If your available VRAM is lower than that, consider reducing the dataset image size to 768px or 512px, respectively.

I only have limited free time at the moment, and it may be a while before I can return to update this project again.

If anyone is willing to improve or optimize the code, I’d truly appreciate it.

As for the main model — it’s still a work in progress and not quite where I want it to be yet, but I’ll do my best to finish it soon.
