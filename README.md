# ComfyUI Custom Node

## Support me

https://coff.ee/kpsss34

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
