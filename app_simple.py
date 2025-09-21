"""
Simplified FluxGym - FLUX LoRA Training Application
This is a minimal version to test core functionality
"""

import os
import sys
import gradio as gr
import torch
from argparse import Namespace

# Set environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

def detect_gpu_info():
    """Detect GPU information including name, VRAM, and capabilities"""
    gpu_info = {
        'name': 'Unknown',
        'vram_gb': 8,  # Default fallback
        'cuda_available': False,
    }
    
    try:
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['name'] = torch.cuda.get_device_name(0)
            
            # Get VRAM information
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_info['vram_gb'] = round(vram_bytes / (1024**3))
            
        else:
            # CPU fallback
            gpu_info['name'] = 'CPU (No CUDA GPU detected)'
            gpu_info['vram_gb'] = 8  # Assume 8GB RAM for CPU
            
    except Exception as e:
        print(f"Error detecting GPU: {e}")
    
    return gpu_info

def get_available_models():
    """Scan models directories and return available files for each type"""
    models_base = os.path.join(os.path.dirname(__file__), "models")
    
    # Get UNET files
    unet_dir = os.path.join(models_base, "unet")
    unet_files = []
    if os.path.exists(unet_dir):
        for file in os.listdir(unet_dir):
            if file.endswith('.safetensors') or file.endswith('.sft'):
                unet_files.append(file)
    unet_files.sort()
    
    # Get VAE files
    vae_dir = os.path.join(models_base, "vae")
    vae_files = []
    if os.path.exists(vae_dir):
        for file in os.listdir(vae_dir):
            if file.endswith('.safetensors') or file.endswith('.sft'):
                vae_files.append(file)
    vae_files.sort()
    
    return {
        'unet': unet_files,
        'vae': vae_files
    }

def auto_configure_settings():
    """Auto-configure settings based on detected GPU"""
    gpu_info = detect_gpu_info()
    
    if gpu_info['cuda_available']:
        info_text = f"""**🎮 GPU Detected:** {gpu_info['name']}
**💾 VRAM:** {gpu_info['vram_gb']} GB
**⚡ Status:** Ready for training"""
    else:
        info_text = """**⚠️ No CUDA GPU detected**
Training will use CPU (very slow)
Consider using a CUDA-compatible GPU for faster training"""
    
    return info_text

def refresh_models():
    """Refresh the available models from the models directory"""
    available_models = get_available_models()
    return (
        gr.update(choices=available_models['unet'], value=available_models['unet'][0] if available_models['unet'] else None),
        gr.update(choices=available_models['vae'], value=available_models['vae'][0] if available_models['vae'] else None),
    )

# Simple theme
theme = gr.themes.Default()

with gr.Blocks(theme=theme, title="FluxGym Simple") as demo:
    gr.Markdown("# FluxGym - Simple Test Version")
    
    with gr.Group():
        gr.Markdown("**🎮 GPU Detection**")
        gpu_info_display = gr.Markdown("Detecting GPU...")
        auto_config_btn = gr.Button("🔧 Auto-Configure", variant="secondary")
    
    # Get available model files
    available_models = get_available_models()
    
    with gr.Group():
        gr.Markdown("**Model Selection**")
        with gr.Row():
            refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
        with gr.Row():
            unet_file = gr.Dropdown(
                label="UNET Model", 
                choices=available_models['unet'], 
                value=available_models['unet'][0] if available_models['unet'] else None,
                info="Main model file"
            )
            vae_file = gr.Dropdown(
                label="VAE Model", 
                choices=available_models['vae'], 
                value=available_models['vae'][0] if available_models['vae'] else None,
                info="VAE encoder/decoder"
            )
    
    with gr.Group():
        lora_name = gr.Textbox(
            label="LoRA Name",
            placeholder="e.g.: My Training",
        )
        resolution = gr.Number(value=512, label="Resolution")
        epochs = gr.Number(value=16, label="Epochs")
    
    status = gr.Textbox(label="Status", value="Ready")
    
    # Event handlers
    auto_config_btn.click(
        fn=auto_configure_settings,
        outputs=[gpu_info_display]
    )
    
    refresh_models_btn.click(
        fn=refresh_models, 
        outputs=[unet_file, vae_file]
    )
    
    # Auto-configure on load
    demo.load(
        fn=auto_configure_settings,
        outputs=[gpu_info_display]
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FluxGym Simple Test')
    parser.add_argument('--listen', '--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting FluxGym Simple on {args.listen}:{args.port}")
    
    demo.launch(
        server_name=args.listen,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True
    )