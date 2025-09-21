"""
FluxGym - FLUX LoRA Training Application

This application automatically scans the local models/ directory structure
to provide dropdowns for model selection instead of using a models.yaml configuration file.

Directory structure expected:
- models/unet/     - Main FLUX model files (.safetensors, .sft)
- models/clip/     - CLIP and T5 text encoder files (.safetensors)
- models/vae/      - VAE encoder/decoder files (.safetensors, .sft)
"""

import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
import subprocess
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
import requests
import base64
from io import BytesIO
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
import json
from library import flux_train_utils, huggingface_util
from argparse import Namespace
import train_network
import toml
import re
from datetime import datetime
MAX_IMAGES = 150

# Workflow automation features
def detect_interrupted_training():
    """Detect and list interrupted training sessions that can be resumed."""
    outputs_dir = resolve_path_without_quotes("outputs")
    interrupted_trainings = []
    
    if not os.path.exists(outputs_dir):
        return interrupted_trainings
    
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        if os.path.isdir(item_path):
            # Check for training artifacts but missing final model
            train_script = os.path.join(item_path, "train.bat") if sys.platform == "win32" else os.path.join(item_path, "train.sh")
            dataset_config = os.path.join(item_path, "dataset.toml")
            
            has_training_files = os.path.exists(train_script) and os.path.exists(dataset_config)
            
            # Look for .safetensors files (completed training)
            safetensors_files = [f for f in os.listdir(item_path) if f.endswith('.safetensors')]
            
            # Check for checkpoint/state files (interrupted training)
            state_files = [f for f in os.listdir(item_path) if 'epoch' in f.lower() or 'step' in f.lower()]
            
            if has_training_files and not safetensors_files and len(state_files) > 0:
                interrupted_trainings.append({
                    'name': item,
                    'path': item_path,
                    'last_checkpoint': max(state_files) if state_files else None,
                    'dataset_path': dataset_config,
                    'script_path': train_script
                })
    
    return interrupted_trainings

def create_batch_training_queue(queue_items):
    """Create a batch training queue to train multiple LoRAs sequentially."""
    queue_file = resolve_path_without_quotes("training_queue.json")
    
    queue_data = {
        'created_at': str(datetime.now()),
        'status': 'pending',
        'current_index': 0,
        'items': queue_items
    }
    
    with open(queue_file, 'w') as f:
        json.dump(queue_data, f, indent=2)
    
    return queue_file

def process_training_queue():
    """Process the training queue and train LoRAs sequentially."""
    queue_file = resolve_path_without_quotes("training_queue.json")
    
    if not os.path.exists(queue_file):
        return "No training queue found"
    
    with open(queue_file, 'r') as f:
        queue_data = json.load(f)
    
    if queue_data['status'] == 'completed':
        return "Queue already completed"
    
    # Process next item in queue
    current_index = queue_data['current_index']
    items = queue_data['items']
    
    if current_index >= len(items):
        queue_data['status'] = 'completed'
        with open(queue_file, 'w') as f:
            json.dump(queue_data, f, indent=2)
        return "Queue completed"
    
    current_item = items[current_index]
    
    # Start training for current item
    try:
        # Execute training script
        script_path = current_item['script_path']
        
        if sys.platform == "win32":
            result = subprocess.run(script_path, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(["bash", script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Mark current item as completed and move to next
            queue_data['current_index'] += 1
            queue_data['items'][current_index]['status'] = 'completed'
            queue_data['items'][current_index]['completed_at'] = str(datetime.now())
        else:
            # Mark as failed
            queue_data['items'][current_index]['status'] = 'failed'
            queue_data['items'][current_index]['error'] = result.stderr
        
        with open(queue_file, 'w') as f:
            json.dump(queue_data, f, indent=2)
        
        return f"Processed item {current_index + 1}/{len(items)}: {current_item['name']}"
        
    except Exception as e:
        queue_data['items'][current_index]['status'] = 'failed'
        queue_data['items'][current_index]['error'] = str(e)
        
        with open(queue_file, 'w') as f:
            json.dump(queue_data, f, indent=2)
        
        return f"Failed to process item {current_index + 1}: {str(e)}"

def auto_optimize_dataset(image_folder, target_resolution=512, auto_caption=True, concept_word=None):
    """Automatically optimize a dataset: resize images, generate captions, create TOML."""
    
    if not os.path.exists(image_folder):
        return "Image folder not found"
    
    optimized_folder = f"{image_folder}_optimized"
    os.makedirs(optimized_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
    
    if len(image_files) == 0:
        return "No image files found"
    
    processed_count = 0
    
    for img_file in image_files:
        try:
            # Resize and optimize image
            img_path = os.path.join(image_folder, img_file)
            img_name, img_ext = os.path.splitext(img_file)
            
            with Image.open(img_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize maintaining aspect ratio
                width, height = img.size
                if width > height:
                    new_width = target_resolution
                    new_height = int((target_resolution / width) * height)
                else:
                    new_height = target_resolution
                    new_width = int((target_resolution / height) * width)
                
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save optimized image
                optimized_img_path = os.path.join(optimized_folder, f"{img_name}.png")
                img_resized.save(optimized_img_path, "PNG", quality=95)
                
                # Generate caption if requested
                if auto_caption:
                    caption_path = os.path.join(optimized_folder, f"{img_name}.txt")
                    
                    # Check if caption already exists
                    if not os.path.exists(caption_path):
                        # Generate simple caption
                        base_caption = f"a high quality image"
                        if concept_word:
                            base_caption = f"{concept_word} {base_caption}"
                        
                        with open(caption_path, 'w', encoding='utf-8') as f:
                            f.write(base_caption)
                
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    # Generate optimized TOML config
    if processed_count > 0 and concept_word:
        toml_path = os.path.join(optimized_folder, "dataset.toml")
        toml_content = gen_toml(optimized_folder, target_resolution, concept_word, 10)
        
        with open(toml_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)
    
    return f"Optimized {processed_count} images in {optimized_folder}"

def smart_parameter_suggestions(gpu_vram_gb, dataset_size, image_resolution):
    """Provide intelligent parameter suggestions based on system specs and dataset."""
    suggestions = {}
    
    # Adjust batch size based on VRAM and resolution
    if gpu_vram_gb >= 24 and image_resolution <= 512:
        suggestions['batch_size'] = 2
        suggestions['vae_batch_size'] = 4
    elif gpu_vram_gb >= 16:
        suggestions['batch_size'] = 1
        suggestions['vae_batch_size'] = 2
    else:
        suggestions['batch_size'] = 1
        suggestions['vae_batch_size'] = 1
    
    # Adjust epochs based on dataset size
    if dataset_size < 5:
        suggestions['max_train_epochs'] = 20
        suggestions['num_repeats'] = 20
    elif dataset_size < 10:
        suggestions['max_train_epochs'] = 15
        suggestions['num_repeats'] = 15
    elif dataset_size < 20:
        suggestions['max_train_epochs'] = 12
        suggestions['num_repeats'] = 10
    else:
        suggestions['max_train_epochs'] = 10
        suggestions['num_repeats'] = 8
    
    # Network dimension based on complexity needed
    if dataset_size < 10:
        suggestions['network_dim'] = 8
    elif dataset_size < 20:
        suggestions['network_dim'] = 16
    else:
        suggestions['network_dim'] = 32
    
    # Learning rate based on dataset size
    if dataset_size < 10:
        suggestions['learning_rate'] = '1e-3'
    else:
        suggestions['learning_rate'] = '8e-4'
    
    # Caching recommendations
    if gpu_vram_gb < 16:
        suggestions['cache_latents_to_disk'] = True
        suggestions['cache_text_encoder_outputs_to_disk'] = True
    else:
        suggestions['cache_latents'] = True
        suggestions['cache_text_encoder_outputs'] = True
    
    return suggestions

def detect_gpu_info():
    """Detect GPU information including name, VRAM, and capabilities"""
    gpu_info = {
        'name': 'Unknown',
        'vram_gb': 8,  # Default fallback
        'compute_capability': None,
        'driver_version': None,
        'cuda_available': False,
        'recommended_settings': {}
    }
    
    try:
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['name'] = torch.cuda.get_device_name(0)
            
            # Get VRAM information
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_info['vram_gb'] = round(vram_bytes / (1024**3))
            
            # Get compute capability
            major, minor = torch.cuda.get_device_capability(0)
            gpu_info['compute_capability'] = f"{major}.{minor}"
            
            # Try to get CUDA version
            try:
                gpu_info['driver_version'] = torch.version.cuda
            except:
                try:
                    gpu_info['driver_version'] = torch.cuda.get_device_name(0)
                except:
                    pass
                
        else:
            # CPU fallback
            gpu_info['name'] = 'CPU (No CUDA GPU detected)'
            gpu_info['vram_gb'] = 8  # Assume 8GB RAM for CPU
            
    except Exception as e:
        print(f"Error detecting GPU: {e}")
    
    return gpu_info

def get_optimal_settings(gpu_info):
    """Generate optimal training settings based on GPU specifications"""
    vram_gb = gpu_info['vram_gb']
    gpu_name = gpu_info['name'].lower()
    
    # Base settings
    settings = {
        'vram_setting': '20G',
        'batch_size': 1,
        'network_dim': 16,
        'learning_rate': '8e-4',
        'max_train_epochs': 16,
        'save_every_n_epochs': 4,
        'resolution': 512,
        'workers': 2,
        'gradient_accumulation': 1,
        'mixed_precision': 'bf16',
        'optimizer': 'adamw8bit',
        'use_8bit_adam': True,
        'fp8_base': True,
        'cache_latents': True,
        'cache_text_encoder_outputs': True
    }
    
    # VRAM-based optimizations
    if vram_gb >= 24:
        # RTX 4090, RTX 6000 Ada, etc.
        settings.update({
            'vram_setting': '20G',
            'batch_size': 1,
            'network_dim': 32,
            'learning_rate': '1e-4',
            'workers': 4,
            'gradient_accumulation': 1,
            'optimizer': 'adamw8bit',
            'resolution': 1024
        })
    elif vram_gb >= 16:
        # RTX 4080, RTX 3090, etc.
        settings.update({
            'vram_setting': '16G',
            'batch_size': 1,
            'network_dim': 16,
            'learning_rate': '8e-4',
            'workers': 3,
            'gradient_accumulation': 1,
            'optimizer': 'adafactor',
            'resolution': 768
        })
    elif vram_gb >= 12:
        # RTX 4070 Ti, RTX 3080, etc.
        settings.update({
            'vram_setting': '12G',
            'batch_size': 1,
            'network_dim': 8,
            'learning_rate': '1e-3',
            'workers': 2,
            'gradient_accumulation': 2,
            'optimizer': 'adafactor',
            'resolution': 512
        })
    elif vram_gb >= 8:
        # RTX 4060 Ti, RTX 3070, etc.
        settings.update({
            'vram_setting': '12G',  # Use 12G settings but with lower params
            'batch_size': 1,
            'network_dim': 4,
            'learning_rate': '1e-3',
            'workers': 1,
            'gradient_accumulation': 4,
            'optimizer': 'adafactor',
            'resolution': 512
        })
    else:
        # Lower VRAM cards
        settings.update({
            'vram_setting': '12G',
            'batch_size': 1,
            'network_dim': 4,
            'learning_rate': '1e-3',
            'workers': 1,
            'gradient_accumulation': 8,
            'optimizer': 'adafactor',
            'resolution': 512
        })
    
    # GPU-specific optimizations
    if 'rtx 40' in gpu_name or 'rtx 50' in gpu_name:
        # Ada Lovelace architecture optimizations
        settings['mixed_precision'] = 'bf16'
        settings['fp8_base'] = True
    elif 'rtx 30' in gpu_name:
        # Ampere architecture optimizations
        settings['mixed_precision'] = 'bf16'
        settings['fp8_base'] = True
    elif 'rtx 20' in gpu_name or 'gtx' in gpu_name:
        # Turing/Pascal architecture
        settings['mixed_precision'] = 'fp16'
        settings['fp8_base'] = False
    elif 'tesla' in gpu_name or 'quadro' in gpu_name:
        # Professional cards
        settings['mixed_precision'] = 'bf16'
        settings['fp8_base'] = True
        settings['workers'] = min(settings['workers'], 2)  # More conservative
    
    # Compute capability optimizations
    if gpu_info.get('compute_capability'):
        cc_major = int(gpu_info['compute_capability'].split('.')[0])
        if cc_major >= 8:  # RTX 30 series and newer
            settings['fp8_base'] = True
        elif cc_major >= 7:  # RTX 20 series
            settings['fp8_base'] = True
        else:  # Older cards
            settings['fp8_base'] = False
            settings['mixed_precision'] = 'fp16'
    
    return settings

def format_gpu_info_display(gpu_info, optimal_settings):
    """Format GPU information for display in the UI"""
    if gpu_info['cuda_available']:
        info_text = f"""**🎮 GPU Detected:** {gpu_info['name']}
**💾 VRAM:** {gpu_info['vram_gb']} GB
**🔧 Compute Capability:** {gpu_info.get('compute_capability', 'Unknown')}
**⚡ Recommended VRAM Setting:** {optimal_settings['vram_setting']}
**📐 Recommended Resolution:** {optimal_settings['resolution']}px
**🧠 Recommended LoRA Rank:** {optimal_settings['network_dim']}"""
    else:
        info_text = """**⚠️ No CUDA GPU detected**
Training will use CPU (very slow)
Consider using a CUDA-compatible GPU for faster training"""
    
    return info_text

def get_available_models():
    """Scan models directories and return available files for each type"""
    models_base = resolve_path_without_quotes("models")
    
    # Get CLIP files
    clip_dir = os.path.join(models_base, "clip")
    clip_files = []
    if os.path.exists(clip_dir):
        for file in os.listdir(clip_dir):
            if file.endswith('.safetensors'):
                clip_files.append(file)
    clip_files.sort()
    
    # Get T5 files (also in clip directory)
    t5_files = []
    if os.path.exists(clip_dir):
        for file in os.listdir(clip_dir):
            if file.endswith('.safetensors') and ('t5' in file.lower() or 'qwen' in file.lower()):
                t5_files.append(file)
    t5_files.sort()
    
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
        'clip': clip_files,
        't5': t5_files,
        'unet': unet_files,
        'vae': vae_files
    }

def readme(unet_file, lora_name, instance_prompt, sample_prompts):

    # Determine base model and license info from UNET filename
    if "schnell" in unet_file.lower():
        base_model_name = "black-forest-labs/FLUX.1-schnell"
        license = "apache-2.0"
        license_name = None
        license_link = None
    else:
        # Default to FLUX.1-dev
        base_model_name = "black-forest-labs/FLUX.1-dev"
        license = "other"
        license_name = "flux-1-dev-non-commercial-license"
        license_link = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md"
    
    license_items = []
    if license:
        license_items.append(f"license: {license}")
    if license_name:
        license_items.append(f"license_name: {license_name}")
    if license_link:
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    print(f"license_items={license_items}")
    print(f"license_str = {license_str}")

    # tags
    tags = [ "text-to-image", "flux", "lora", "diffusers", "template:sd-lora", "fluxgym" ]

    # widgets
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            # Filename Schema: [name]_[steps]_[index]_[timestamp].png
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))

        # Sort by numeric index
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)

        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append(
                {
                    "text": prompt,
                    "output": {
                        "url": image_path
                    },
                }
            )
    except:
        print(f"no samples")
    dtype = "torch.bfloat16"
    # Construct the README content
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

A Flux LoRA trained on a local computer with [Fluxgym](https://github.com/cocktailpeanut/fluxgym)

<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.

"""
    return readme_content

def refresh_models():
    """Refresh the available models from the models directory"""
    available_models = get_available_models()
    return (
        gr.update(choices=available_models['unet'], value=available_models['unet'][0] if available_models['unet'] else None),
        gr.update(choices=available_models['vae'], value=available_models['vae'][0] if available_models['vae'] else None),
        gr.update(choices=available_models['clip'], value=available_models['clip'][0] if available_models['clip'] else None),
        gr.update(choices=available_models['t5'], value=available_models['t5'][0] if available_models['t5'] else None)
    )

def check_ollama_connection(base_url="http://localhost:11434"):
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_ollama_models(base_url="http://localhost:11434"):
    """Get list of available Ollama models with enhanced filtering"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]

            if not models:
                return []

            # Enhanced filtering for vision-capable models
            vision_keywords = [
                'llava', 'vision', 'minicpm', 'cogvlm', 'qwen-vl', 'internvl', 'moondream',
                'llama-vision', 'bakllava', 'llava-phi', 'llava-qwen', 'llava-llama',
                'gpt4v', 'claude-vision', 'gemini-vision', 'visual', 'image', 'caption'
            ]

            # First pass: strict vision model filtering
            vision_models = [m for m in models if any(keyword in m.lower() for keyword in vision_keywords)]

            # Second pass: if no vision models found, include models that might have vision capabilities
            if not vision_models:
                # Include models with common vision-related names or recent multimodal models
                potential_vision = [
                    m for m in models if any(keyword in m.lower() for keyword in
                    ['llama', 'qwen', 'mistral', 'gemma', 'phi', 'gemma2', 'mixtral', 'yi'])
                ]
                vision_models = potential_vision if potential_vision else models

            # Sort models alphabetically for better UX
            vision_models.sort()

            return vision_models
        return []
    except requests.exceptions.RequestException as e:
        print(f"Network error connecting to Ollama: {e}")
        return []
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []

def encode_image_to_base64(image_path):
    """Convert image to base64 for Ollama API"""
    with Image.open(image_path) as img:
        # Resize if too large to avoid API limits
        if img.width > 1024 or img.height > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

def caption_with_ollama(image_path, model_name, prompt, base_url="http://localhost:11434"):
    """Generate caption using Ollama vision model"""
    try:
        image_b64 = encode_image_to_base64(image_path)
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
        
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def refresh_ollama_models(ollama_url):
    """Refresh available Ollama models with better error handling"""
    try:
        models = get_ollama_models(ollama_url)
        if not models:
            gr.Warning("No Ollama models found. Make sure Ollama is running and has vision-capable models installed.")
            return gr.update(choices=[], value=None)

        # Select the first model as default
        default_model = models[0] if models else None

        gr.Info(f"Successfully loaded {len(models)} Ollama models")
        return gr.update(choices=models, value=default_model)

    except Exception as e:
        gr.Error(f"Failed to refresh Ollama models: {str(e)}")
        return gr.update(choices=[], value=None)

def auto_load_ollama_models():
    """Automatically load Ollama models on app startup if Ollama is available"""
    try:
        if check_ollama_connection():
            models = get_ollama_models()
            if models:
                print(f"✅ Auto-loaded {len(models)} Ollama models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
                return models, models[0] if models else None
        return [], None
    except Exception as e:
        print(f"⚠️  Could not auto-load Ollama models: {e}")
        return [], None

def toggle_captioning_method(method):
    """Toggle between Florence-2 and Ollama captioning options"""
    if method == "Ollama":
        # Auto-refresh Ollama models when switching to Ollama
        try:
            models = get_ollama_models()
            default_model = models[0] if models else None
            return gr.update(visible=True), gr.update(choices=models, value=default_model)
        except Exception as e:
            print(f"⚠️ Could not load Ollama models: {e}")
            return gr.update(visible=True), gr.update(choices=[], value=None)
    else:
        return gr.update(visible=False), gr.update(choices=[], value=None)

def auto_configure_settings():
    """Auto-configure settings based on detected GPU"""
    gpu_info = detect_gpu_info()
    optimal_settings = get_optimal_settings(gpu_info)
    gpu_display = format_gpu_info_display(gpu_info, optimal_settings)
    
    return (
        gpu_display,
        optimal_settings['vram_setting'],
        optimal_settings['resolution'],
        optimal_settings['network_dim'],
        optimal_settings['learning_rate'],
        optimal_settings['max_train_epochs'],
        optimal_settings['save_every_n_epochs'],
        optimal_settings['workers']
    )

def account_hf():
    try:
        with open("HF_TOKEN", "r") as file:
            token = file.read()
            api = HfApi(token=token)
            try:
                account = api.whoami()
                return { "token": token, "account": account['name'] }
            except:
                return None
    except:
        return None

"""
hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def logout_hf():
    os.remove("HF_TOKEN")
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)


"""
hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
"""
def login_hf(hf_token):
    api = HfApi(token=hf_token)
    try:
        account = api.whoami()
        if account != None:
            if "name" in account:
                with open("HF_TOKEN", "w") as file:
                    file.write(hf_token)
                global current_account
                current_account = account_hf()
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
        return gr.update(), gr.update(), gr.update(), gr.update()
    except:
        print(f"incorrect hf_token")
        return gr.update(), gr.update(), gr.update(), gr.update()

def upload_hf(unet_file, lora_rows, repo_owner, repo_name, repo_visibility, hf_token):
    src = lora_rows
    repo_id = f"{repo_owner}/{repo_name}"
    gr.Info(f"Uploading to Huggingface. Please Stand by...")
    args = Namespace(
        huggingface_repo_id=repo_id,
        huggingface_repo_type="model",
        huggingface_repo_visibility=repo_visibility,
        huggingface_path_in_repo="",
        huggingface_token=hf_token,
        async_upload=False
    )
    print(f"upload_hf args={args}")
    huggingface_util.upload(args=args, src=src)
    gr.Info(f"[Upload Complete] https://huggingface.co/{repo_id}")

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    if len(uploaded_images) <= 1:
        raise gr.Error(
            "Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    
    # Update for the captioning_area
    updates.append(gr.update(visible=True))

    # Auto-load Ollama models if available
    ollama_models, default_model = auto_load_ollama_models()
    updates.append(gr.update(visible=False))  # ollama_settings initially hidden
    
    # Update Ollama model dropdown with loaded models
    updates.append(gr.update(choices=ollama_models, value=default_model))    # Update visibility and image for each captioning row and image
    for i in range(1, MAX_IMAGES + 1):
        # Determine if the current row and image should be visible
        visible = i <= len(uploaded_images)

        # Update visibility of the captioning row
        updates.append(gr.update(visible=visible))

        # Update for image component - display image if available, otherwise hide
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))

        corresponding_caption = False
        if(image_value):
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()

        # Update value of captioning area
        text_value = corresponding_caption if visible and corresponding_caption else concept_sentence if visible and concept_sentence else None
        updates.append(gr.update(value=text_value, visible=visible))

    # Update for the sample caption area
    updates.append(gr.update(visible=True))
    updates.append(gr.update(visible=True))

    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(destination_folder, size, *inputs):
    print("Creating dataset")
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        # copy the images to the datasets folder
        new_image_path = shutil.copy(image, destination_folder)

        # if it's a caption text file skip the next bit
        ext = os.path.splitext(new_image_path)[-1].lower()
        if ext == '.txt':
            continue

        # resize the images
        resize_image(new_image_path, new_image_path, size)

        # copy the captions

        original_caption = inputs[index + 1]

        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
        print(f"image_path={new_image_path}, caption_path = {caption_path}, original_caption={original_caption}")
        # if caption_path exists, do not write
        if os.path.exists(caption_path):
            print(f"{caption_path} already exists. use the existing .txt file")
        else:
            print(f"{caption_path} create a .txt caption file")
            with open(caption_path, 'w') as file:
                file.write(original_caption)

    print(f"destination_folder {destination_folder}")
    return destination_folder


def run_captioning(images, concept_sentence, captioning_method, ollama_model, ollama_url, *captions):
    print(f"run_captioning: method={captioning_method}")
    print(f"concept sentence: {concept_sentence}")
    print(f"captions: {captions}")
    
    captions = list(captions)
    
    if captioning_method == "Ollama":
        # Use Ollama for captioning
        if not check_ollama_connection(ollama_url):
            gr.Error(f"Cannot connect to Ollama at {ollama_url}. Please ensure Ollama is running.")
            return captions
        
        if not ollama_model:
            gr.Error("Please select an Ollama model for captioning.")
            return captions
        
        ollama_prompt = "Describe this image in detail, focusing on the visual elements, composition, style, and any notable features. Be descriptive but concise."
        
        for i, image_path in enumerate(images):
            print(f"Processing image {i+1}/{len(images)} with Ollama")
            if isinstance(image_path, str):
                try:
                    caption_text = caption_with_ollama(image_path, ollama_model, ollama_prompt, ollama_url)
                    
                    if caption_text.startswith("Error:"):
                        gr.Warning(f"Failed to caption image {i+1}: {caption_text}")
                        continue
                    
                    # Add concept sentence if provided
                    if concept_sentence:
                        caption_text = f"{concept_sentence} {caption_text}"
                    
                    captions[i] = caption_text
                    yield captions
                    
                except Exception as e:
                    gr.Warning(f"Error processing image {i+1}: {str(e)}")
                    continue
    
    else:
        # Use Florence-2 (robust with fallbacks)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device={device}")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        florence_repo = "multimodalart/Florence-2-large-no-flash-attn"
        model = None
        processor = None

        def unload():
            try:
                if model is not None:
                    model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Load processor first (cheaper) so we can still fallback
        try:
            processor = AutoProcessor.from_pretrained(florence_repo, trust_remote_code=True)
        except Exception as e:
            gr.Warning(f"Processor load failed: {e}. Falling back to simple heuristic captions.")
        
        # Try progressive model loading strategies
        load_attempt_errors = []
        if processor is not None:
            load_strategies = [
                dict(attn_implementation="eager"),
                dict(),  # default
            ]
            for strat in load_strategies:
                if model is not None:
                    break
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        florence_repo,
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                        **strat,
                    )
                    model.to(device)
                except Exception as ex:
                    load_attempt_errors.append(str(ex))
                    model = None

        if model is None or processor is None:
            # Final fallback: simple heuristic captioning
            for i, image_path in enumerate(images):
                try:
                    base_caption = "a training reference image"
                    if isinstance(image_path, str):
                        with Image.open(image_path) as im:
                            w, h = im.size
                            base_caption = f"an image ({w}x{h}) with subject" if w and h else base_caption
                    if concept_sentence:
                        base_caption = f"{concept_sentence} {base_caption}"
                    captions[i] = base_caption
                except Exception as fe:
                    captions[i] = concept_sentence or "image"
                yield captions
            gr.Warning("Florence model unavailable. Used heuristic fallback captions. Errors: " + "; ".join(load_attempt_errors[-2:]))
            unload()
            return

        # Inference loop
        prompt_token = "<DETAILED_CAPTION>"
        for i, image_path in enumerate(images):
            print(f"Processing image {i+1}/{len(images)} with Florence-2 (robust)")
            if isinstance(image_path, str):
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as ie:
                    gr.Warning(f"Failed to open image {image_path}: {ie}")
                    captions[i] = concept_sentence or "image"
                    yield captions
                    continue
            else:
                # Unsupported type
                captions[i] = concept_sentence or "image"
                yield captions
                continue

            try:
                inputs = processor(text=prompt_token, images=image, return_tensors="pt").to(device, torch_dtype)
                # Some Florence variants might not expose generate; fallback to language_model.generate or manual forward
                generate_fn = None
                if hasattr(model, "generate"):
                    generate_fn = model.generate
                elif hasattr(getattr(model, "language_model", None), "generate"):
                    generate_fn = model.language_model.generate  # type: ignore

                if generate_fn is None:
                    raise AttributeError("No generate method available on Florence model")

                generated_ids = generate_fn(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=256,
                    num_beams=3,
                )
                decoded = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(
                    decoded, task=prompt_token, image_size=(image.width, image.height)
                )
                caption_text = parsed.get(prompt_token, "").replace("The image shows ", "").strip()
                if not caption_text:
                    caption_text = "a detailed image"
            except Exception as gen_ex:
                gr.Warning(f"Caption generation failed (fallback applied): {gen_ex}")
                caption_text = "a training image"

            if concept_sentence:
                caption_text = f"{concept_sentence} {caption_text}".strip()
            captions[i] = caption_text
            yield captions

        unload()

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""
def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(
    unet_file,
    clip_file,
    t5_file,
    vae_file,
    output_name,
    resolution,
    seed,
    workers,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    sample_prompts,
    sample_every_n_steps,
    lr_scheduler,
    max_train_steps,
    train_batch_size,
    reg_data_dir,
    cache_latents,
    cache_latents_to_disk,
    cache_text_encoder_outputs,
    cache_text_encoder_outputs_to_disk,
    skip_cache_check,
    vae_batch_size,
    cache_info,
    *advanced_components
):

    print(f"gen_sh: network_dim:{network_dim}, max_train_epochs={max_train_epochs}, save_every_n_epochs={save_every_n_epochs}, timestep_sampling={timestep_sampling}, guidance_scale={guidance_scale}, vram={vram}, sample_prompts={sample_prompts}, sample_every_n_steps={sample_every_n_steps}")

    output_dir = resolve_path(f"outputs/{output_name}")
    sample_prompts_path = resolve_path(f"outputs/{output_name}/sample_prompts.txt")

    line_break = "\\"
    file_type = "sh"
    if sys.platform == "win32":
        line_break = "^"
        file_type = "bat"

    ############# Sample args ########################
    sample = ""
    if len(sample_prompts) > 0 and sample_every_n_steps > 0:
        sample = f"""--sample_prompts={sample_prompts_path} --sample_every_n_steps="{sample_every_n_steps}" {line_break}"""


    ############# Optimizer args ########################
#    if vram == "8G":
#        optimizer = f"""--optimizer_type adafactor {line_break}
#    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
#        --split_mode {line_break}
#        --network_args "train_blocks=single" {line_break}
#        --lr_scheduler constant_with_warmup {line_break}
#        --max_grad_norm 0.0 {line_break}"""
    if vram == "16G":
        # 16G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    elif vram == "12G":
      # 12G VRAM
        optimizer = f"""--optimizer_type adafactor {line_break}
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" {line_break}
  --split_mode {line_break}
  --network_args "train_blocks=single" {line_break}
  --lr_scheduler constant_with_warmup {line_break}
  --max_grad_norm 0.0 {line_break}"""
    else:
        # 20G+ VRAM
        optimizer = f"--optimizer_type adamw8bit {line_break}"


    #######################################################
    # Build paths from selected files
    pretrained_model_path = resolve_path(f"models/unet/{unet_file}")
    clip_path = resolve_path(f"models/clip/{clip_file}")
    t5_path = resolve_path(f"models/clip/{t5_file}")
    ae_path = resolve_path(f"models/vae/{vae_file}")
    
    sh = f"""accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  {optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {resolve_path(f"outputs/{output_name}/dataset.toml")} {line_break}
  --output_dir {output_dir} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}"""
   
    # Add new training parameters
    if lr_scheduler and lr_scheduler != "constant":
        sh += f"  --lr_scheduler {lr_scheduler} {line_break}"
    
    if max_train_steps is not None and max_train_steps > 0:
        sh += f"  --max_train_steps {max_train_steps} {line_break}"
    
    if train_batch_size and train_batch_size != 1:
        sh += f"  --train_batch_size {train_batch_size} {line_break}"
    
    if reg_data_dir and reg_data_dir.strip():
        reg_data_path = resolve_path(reg_data_dir.strip())
        sh += f"  --reg_data_dir {reg_data_path} {line_break}"

    # Add caching options
    if cache_latents:
        sh += f"  --cache_latents {line_break}"
    
    if cache_latents_to_disk:
        sh += f"  --cache_latents_to_disk {line_break}"
    
    if cache_text_encoder_outputs:
        sh += f"  --cache_text_encoder_outputs {line_break}"
    
    if cache_text_encoder_outputs_to_disk:
        sh += f"  --cache_text_encoder_outputs_to_disk {line_break}"
    
    if skip_cache_check:
        sh += f"  --skip_cache_check {line_break}"
    
    if vae_batch_size and vae_batch_size != 1:
        sh += f"  --vae_batch_size {vae_batch_size} {line_break}"
    
    if cache_info:
        sh += f"  --cache_info {line_break}"


    ############# Advanced args ########################
    global advanced_component_ids
    global original_advanced_component_values
   
    # check dirty
    print(f"original_advanced_component_values = {original_advanced_component_values}")
    advanced_flags = []
    for i, current_value in enumerate(advanced_components):
#        print(f"compare {advanced_component_ids[i]}: old={original_advanced_component_values[i]}, new={current_value}")
        if original_advanced_component_values[i] != current_value:
            # dirty
            if current_value == True:
                # Boolean
                advanced_flags.append(advanced_component_ids[i])
            else:
                # string
                advanced_flags.append(f"{advanced_component_ids[i]} {current_value}")

    if len(advanced_flags) > 0:
        advanced_flags_str = f" {line_break}\n  ".join(advanced_flags)
        sh = sh + "\n  " + advanced_flags_str

    return sh

def gen_toml(
  dataset_folder,
  resolution,
  class_tokens,
  num_repeats
):
    toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""
    return toml

def update_total_steps(max_train_epochs, num_repeats, images):
    try:
        num_images = len(images)
        total_steps = max_train_epochs * num_images * num_repeats
        print(f"max_train_epochs={max_train_epochs} num_images={num_images}, num_repeats={num_repeats}, total_steps={total_steps}")
        return gr.update(value = total_steps)
    except:
        print("")

def set_repo(lora_rows):
    selected_name = os.path.basename(lora_rows)
    return gr.update(value=selected_name)

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except Exception as e:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

def start_training(
    unet_file,
    clip_file,
    t5_file,
    vae_file,
    lora_name,
    train_script,
    train_config,
    sample_prompts,
):
    # write custom script and toml
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Verify that selected model files exist
    models_base = resolve_path_without_quotes("models")
    unet_path = os.path.join(models_base, "unet", unet_file)
    clip_path = os.path.join(models_base, "clip", clip_file)
    t5_path = os.path.join(models_base, "clip", t5_file)
    vae_path = os.path.join(models_base, "vae", vae_file)
    
    missing_files = []
    if not os.path.exists(unet_path):
        missing_files.append(f"UNET: {unet_file}")
    if not os.path.exists(clip_path):
        missing_files.append(f"CLIP: {clip_file}")
    if not os.path.exists(t5_path):
        missing_files.append(f"T5: {t5_file}")
    if not os.path.exists(vae_path):
        missing_files.append(f"VAE: {vae_file}")
    
    if missing_files:
        gr.Error(f"Missing model files: {', '.join(missing_files)}")
        return

    file_type = "sh"
    if sys.platform == "win32":
        file_type = "bat"

    sh_filename = f"train.{file_type}"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/{sh_filename}")
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"Generated train script at {sh_filename}")


    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"Generated dataset.toml")

    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"Generated sample_prompts.txt")

    # Train
    if sys.platform == "win32":
        command = sh_filepath
    else:
        command = f"bash \"{sh_filepath}\""

    # Use subprocess to run the command and capture output
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"Started training")
    
    try:
        # Run the training command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=cwd,
            bufsize=1,
            universal_newlines=True
        )
        
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Yield the accumulated output
                yield "\n".join(output_lines[-50:])  # Show last 50 lines
        
        rc = process.poll()
        if rc == 0:
            output_lines.append("\n✅ Training completed successfully!")
        else:
            output_lines.append(f"\n❌ Training failed with exit code: {rc}")
        
        yield "\n".join(output_lines[-50:])
        
    except Exception as e:
        error_msg = f"\n❌ Error running training: {str(e)}"
        yield error_msg

    # Generate Readme
    config = toml.loads(train_config)
    concept_sentence = config['datasets'][0]['subsets'][0]['class_tokens']
    print(f"concept_sentence={concept_sentence}")
    print(f"lora_name {lora_name}, concept_sentence={concept_sentence}, output_name={output_name}")
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sample_prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    md = readme(unet_file, lora_name, concept_sentence, sample_prompts)
    readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)

    gr.Info(f"Training Complete. Check the outputs folder for the LoRA files.")


def update(
    unet_file,
    clip_file,
    t5_file,
    vae_file,
    lora_name,
    resolution,
    seed,
    workers,
    class_tokens,
    learning_rate,
    network_dim,
    max_train_epochs,
    save_every_n_epochs,
    timestep_sampling,
    guidance_scale,
    vram,
    num_repeats,
    sample_prompts,
    sample_every_n_steps,
    lr_scheduler,
    max_train_steps,
    train_batch_size,
    reg_data_dir,
    cache_latents,
    cache_latents_to_disk,
    cache_text_encoder_outputs,
    cache_text_encoder_outputs_to_disk,
    skip_cache_check,
    vae_batch_size,
    cache_info,
    *advanced_components,
):
    output_name = slugify(lora_name)
    dataset_folder = str(f"datasets/{output_name}")
    sh = gen_sh(
        unet_file,
        clip_file,
        t5_file,
        vae_file,
        output_name,
        resolution,
        seed,
        workers,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        sample_prompts,
        sample_every_n_steps,
        lr_scheduler,
        max_train_steps,
        train_batch_size,
        reg_data_dir,
        cache_latents,
        cache_latents_to_disk,
        cache_text_encoder_outputs,
        cache_text_encoder_outputs_to_disk,
        skip_cache_check,
        vae_batch_size,
        cache_info,
        *advanced_components,
    )
    toml = gen_toml(
        dataset_folder,
        resolution,
        class_tokens,
        num_repeats
    )
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

"""
demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, hf_account])
"""
def loaded():
    global current_account
    current_account = account_hf()
    print(f"current_account={current_account}")
    if current_account != None:
        return gr.update(value=current_account["token"]), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
    else:
        return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def update_sample(concept_sentence):
    return gr.update(value=concept_sentence)

def refresh_publish_tab():
    loras = get_loras()
    return gr.Dropdown(label="Trained LoRAs", choices=loras)

def init_advanced():
    """Initialize advanced components with safe, static configuration"""
    # Create a simplified set of advanced components to avoid dynamic parsing issues
    advanced_components = []
    advanced_component_ids = []
    
    # Define commonly used advanced training options manually
    advanced_options = [
        {"id": "--use_8bit_adam", "label": "Use 8-bit Adam", "type": "checkbox", "help": "Use 8-bit Adam optimizer"},
        {"id": "--xformers", "label": "Use Xformers", "type": "checkbox", "help": "Use xformers for memory efficiency"},
        {"id": "--gradient_accumulation_steps", "label": "Gradient Accumulation Steps", "type": "number", "help": "Number of steps to accumulate gradients"},
        {"id": "--clip_skip", "label": "CLIP Skip", "type": "number", "help": "Number of CLIP layers to skip"},
        {"id": "--noise_offset", "label": "Noise Offset", "type": "number", "help": "Noise offset for training"},
        {"id": "--adaptive_noise_scale", "label": "Adaptive Noise Scale", "type": "number", "help": "Adaptive noise scale"},
        {"id": "--multires_noise_iterations", "label": "Multires Noise Iterations", "type": "number", "help": "Multi-resolution noise iterations"},
        {"id": "--multires_noise_discount", "label": "Multires Noise Discount", "type": "number", "help": "Multi-resolution noise discount"},
    ]
    
    for option in advanced_options:
        with gr.Column(min_width=300):
            if option["type"] == "checkbox":
                component = gr.Checkbox(
                    label=option["label"],
                    value=False,
                    info=option.get("help", ""),
                    elem_classes=["advanced"]
                )
            elif option["type"] == "number":
                component = gr.Number(
                    label=option["label"],
                    value=0,
                    info=option.get("help", ""),
                    elem_classes=["advanced"]
                )
            else:
                component = gr.Textbox(
                    label=option["label"],
                    value="",
                    info=option.get("help", ""),
                    elem_classes=["advanced"]
                )
            
            component.elem_id = option["id"]
            advanced_components.append(component)
            advanced_component_ids.append(option["id"])
    
    return advanced_components, advanced_component_ids


theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
@keyframes rotate {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
#advanced_options .advanced:nth-child(even) { background: rgba(0,0,100,0.04) !important; }
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px); }
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px; }
nav img { height: 40px; width: 40px; border-radius: 40px; }
nav img.rotate { animation: rotate 2s linear infinite; }
.flexible { flex-grow: 1; }
.tast-details { margin: 10px 0 !important; }
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px); }
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px; }
.toast-body { border: none !important; }
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03); }
#terminal .generating { border: none !important; }
#terminal label { position: absolute !important; }
.tabs { margin-top: 50px; }
.hidden { display: none !important; }
.codemirror-wrapper .cm-line { font-size: 12px !important; }
label { font-weight: bold !important; }
#start_training.clicked { background: silver; color: black; }
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) {
        window.clearInterval(window.iidxx);
    }
    window.iidxx = window.setInterval(function() {
        let text=document.querySelector(".codemirror-wrapper .cm-line").innerText.trim()
        let img = document.querySelector("#logo")
        if (text.length > 0) {
            autoscroll.classList.remove("hidden")
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON"
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate")
            } else {
                autoscroll.textContent = "Autoscroll OFF"
                img.classList.remove("rotate")
            }
        }
    }, 500);
    console.log("autoscroll", autoscroll)
    autoscroll.addEventListener("click", (e) => {
        autoscroll.classList.toggle("on")
    })
    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    function handleClick() {
        console.log("refresh")
        document.querySelector("#refresh").click();
    }
    const debouncedClick = debounce(handleClick, 1000);
    document.addEventListener("input", debouncedClick);

    document.querySelector("#start_training").addEventListener("click", (e) => {
      e.target.classList.add("clicked")
      e.target.innerHTML = "Training..."
    })

}
"""

current_account = account_hf()
print(f"current_account={current_account}")

with gr.Blocks(elem_id="app", theme=theme, css=css) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Gym"):
            output_components = []
            with gr.Row():
                gr.HTML("""<nav>
            <img id='logo' src='/file=icon.png' width='80' height='80'>
            <div class='flexible'></div>
            <button id='autoscroll' class='on hidden'></button>
        </nav>
        """)
            with gr.Row(elem_id='container'):
                with gr.Column():
                    gr.Markdown(
                        """# Step 1. LoRA Info
        <p style="margin-top:0">Configure your LoRA train settings.</p>
        """, elem_classes="group_padding")
                    
                    # GPU Detection and Auto-Configuration
                    with gr.Group():
                        gr.Markdown("**🎮 GPU Detection & Auto-Configuration**")
                        gpu_info_display = gr.Markdown("Detecting GPU...")
                        auto_config_btn = gr.Button("🔧 Auto-Configure Optimal Settings", variant="secondary")
                    
                    lora_name = gr.Textbox(
                        label="The name of your LoRA",
                        info="This has to be a unique name",
                        placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
                    )
                    concept_sentence = gr.Textbox(
                        elem_id="--concept_sentence",
                        label="Trigger word/sentence",
                        info="Trigger word or sentence to be used",
                        placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                        interactive=True,
                    )
                    
                    # Get available model files
                    available_models = get_available_models()
                    
                    # Model selection dropdowns
                    with gr.Group():
                        gr.Markdown("**Model Selection** - Select files from your local models/ directory")
                        with gr.Row():
                            refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
                        with gr.Row():
                            with gr.Column():
                                unet_file = gr.Dropdown(
                                    label="UNET Model", 
                                    choices=available_models['unet'], 
                                    value=available_models['unet'][0] if available_models['unet'] else None,
                                    info="Main model file (.safetensors or .sft)"
                                )
                            with gr.Column():
                                vae_file = gr.Dropdown(
                                    label="VAE Model", 
                                    choices=available_models['vae'], 
                                    value=available_models['vae'][0] if available_models['vae'] else None,
                                    info="VAE encoder/decoder"
                                )
                        with gr.Row():
                            with gr.Column():
                                clip_file = gr.Dropdown(
                                    label="CLIP Model", 
                                    choices=available_models['clip'], 
                                    value=available_models['clip'][0] if available_models['clip'] else None,
                                    info="CLIP text encoder"
                                )
                            with gr.Column():
                                t5_file = gr.Dropdown(
                                    label="T5 Model", 
                                    choices=available_models['t5'], 
                                    value=available_models['t5'][0] if available_models['t5'] else None,
                                    info="T5 text encoder"
                                )
                    
                    vram = gr.Radio(["20G", "16G", "12G" ], value="20G", label="VRAM", info="GPU memory optimization setting (auto-detected)", interactive=True)
                    num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", info="Higher values = more training on each image", interactive=True)
                    max_train_epochs = gr.Number(label="Max Train Epochs", value=16, info="Total training epochs (auto-optimized)", interactive=True)
                    total_steps = gr.Number(0, interactive=False, label="Expected training steps")
                    sample_prompts = gr.Textbox("", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True)
                    sample_every_n_steps = gr.Number(0, precision=0, label="Sample Image Every N Steps", interactive=True)
                    resolution = gr.Number(value=512, precision=0, label="Resize dataset images", info="Training resolution (auto-optimized for GPU)", interactive=True)
                with gr.Column():
                    gr.Markdown(
                        """# Step 2. Dataset
        <p style="margin-top:0">Make sure the captions include the trigger word.</p>
        """, elem_classes="group_padding")
                    with gr.Group():
                        images = gr.File(
                            file_types=["image", ".txt"],
                            label="Upload your images",
                            #info="If you want, you can also manually upload caption files that match the image names (example: img0.png => img0.txt)",
                            file_count="multiple",
                            interactive=True,
                            visible=True,
                            scale=1,
                        )
                    with gr.Group(visible=False) as captioning_area:
                        with gr.Row():
                            captioning_method = gr.Radio(
                                choices=["Florence-2", "Ollama"], 
                                value="Florence-2", 
                                label="Captioning Method",
                                info="Choose between Florence-2 (local) or Ollama (requires Ollama server)"
                            )
                        
                        with gr.Row():
                            with gr.Column():
                                do_captioning = gr.Button("Add AI captions", variant="primary")
                            with gr.Column(visible=False) as ollama_settings:
                                ollama_url = gr.Textbox(
                                    value="http://localhost:11434",
                                    label="Ollama URL",
                                    info="Ollama server URL"
                                )
                                ollama_model = gr.Dropdown(
                                    choices=[],
                                    label="Ollama Vision Model",
                                    info="Select a vision-capable model (auto-loaded from Ollama)"
                                )
                                refresh_ollama = gr.Button("🔄 Refresh Models", size="sm")
                        
                        output_components.append(captioning_area)
                        output_components.append(ollama_settings)
                        output_components.append(ollama_model)  # Add ollama_model to output components
                        caption_list = []
                        for i in range(1, MAX_IMAGES + 1):
                            locals()[f"captioning_row_{i}"] = gr.Row(visible=False)
                            with locals()[f"captioning_row_{i}"]:
                                locals()[f"image_{i}"] = gr.Image(
                                    type="filepath",
                                    width=111,
                                    height=111,
                                    min_width=111,
                                    interactive=False,
                                    scale=2,
                                    show_label=False,
                                    show_share_button=False,
                                    show_download_button=False,
                                )
                                locals()[f"caption_{i}"] = gr.Textbox(
                                    label=f"Caption {i}", scale=15, interactive=True
                                )

                            output_components.append(locals()[f"captioning_row_{i}"])
                            output_components.append(locals()[f"image_{i}"])
                            output_components.append(locals()[f"caption_{i}"])
                            caption_list.append(locals()[f"caption_{i}"])
                with gr.Column():
                    gr.Markdown(
                        """# Step 3. Train
        <p style="margin-top:0">Press start to start training.</p>
        """, elem_classes="group_padding")
                    refresh = gr.Button("Refresh", elem_id="refresh", visible=False)
                    start = gr.Button("Start training", visible=False, elem_id="start_training")
                    output_components.append(start)
                    train_script = gr.Textbox(label="Train script", max_lines=100, interactive=True)
                    train_config = gr.Textbox(label="Train config", max_lines=100, interactive=True)
            with gr.Accordion("Advanced options", elem_id='advanced_options', open=False):
                gr.Markdown("""
                **💡 Auto-Configuration Info:**
                - Settings are automatically optimized based on your GPU and VRAM
                - Higher VRAM allows larger batch sizes, higher resolution, and better quality
                - Network Dim (LoRA Rank) affects model complexity vs. training speed
                - Learning rate is adjusted for optimal convergence
                """)
                with gr.Row():
                    with gr.Column(min_width=300):
                        seed = gr.Number(label="--seed", info="Seed", value=42, interactive=True)
                    with gr.Column(min_width=300):
                        workers = gr.Number(label="--max_data_loader_n_workers", info="Number of Workers (auto-optimized)", value=2, interactive=True)
                    with gr.Column(min_width=300):
                        learning_rate = gr.Textbox(label="--learning_rate", info="Learning Rate (auto-optimized)", value="8e-4", interactive=True)
                    with gr.Column(min_width=300):
                        save_every_n_epochs = gr.Number(label="--save_every_n_epochs", info="Save every N epochs", value=4, interactive=True)
                    with gr.Column(min_width=300):
                        guidance_scale = gr.Number(label="--guidance_scale", info="Guidance Scale", value=1.0, interactive=True)
                    with gr.Column(min_width=300):
                        timestep_sampling = gr.Textbox(label="--timestep_sampling", info="Timestep Sampling", value="shift", interactive=True)
                    with gr.Column(min_width=300):
                        network_dim = gr.Number(label="--network_dim", info="LoRA Rank (auto-optimized for GPU)", value=4, minimum=4, maximum=128, step=4, interactive=True)
                
                # New advanced training parameters
                with gr.Row():
                    with gr.Column(min_width=300):
                        lr_scheduler = gr.Dropdown(
                            label="--lr_scheduler", 
                            choices=["constant", "cosine", "cosine_with_restarts", "polynomial", "linear", "constant_with_warmup", "adafactor"],
                            value="constant",
                            info="Learning rate scheduler type",
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        max_train_steps = gr.Number(
                            label="--max_train_steps", 
                            info="Maximum training steps (overrides max_train_epochs)", 
                            value=None, 
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        train_batch_size = gr.Number(
                            label="--train_batch_size", 
                            info="Batch size for training", 
                            value=1, 
                            minimum=1, 
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        reg_data_dir = gr.Textbox(
                            label="--reg_data_dir", 
                            info="Directory for regularization/reference images", 
                            value="", 
                            interactive=True
                        )
                
                # Caching options for training optimization
                with gr.Row():
                    with gr.Column(min_width=300):
                        cache_latents = gr.Checkbox(
                            label="--cache_latents", 
                            info="Cache latents to main memory to reduce VRAM usage", 
                            value=False, 
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        cache_latents_to_disk = gr.Checkbox(
                            label="--cache_latents_to_disk", 
                            info="Cache latents to disk to reduce VRAM usage", 
                            value=False, 
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        cache_text_encoder_outputs = gr.Checkbox(
                            label="--cache_text_encoder_outputs", 
                            info="Cache text encoder outputs", 
                            value=False, 
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        cache_text_encoder_outputs_to_disk = gr.Checkbox(
                            label="--cache_text_encoder_outputs_to_disk", 
                            info="Cache text encoder outputs to disk", 
                            value=False, 
                            interactive=True
                        )
                
                with gr.Row():
                    with gr.Column(min_width=300):
                        skip_cache_check = gr.Checkbox(
                            label="--skip_cache_check", 
                            info="Skip content validation of cache", 
                            value=False, 
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        vae_batch_size = gr.Number(
                            label="--vae_batch_size", 
                            info="Batch size for caching latents", 
                            value=1, 
                            minimum=1, 
                            interactive=True
                        )
                    with gr.Column(min_width=300):
                        cache_info = gr.Checkbox(
                            label="--cache_info", 
                            info="Cache meta information for faster dataset loading (DreamBooth)", 
                            value=False, 
                            interactive=True
                        )
                
                advanced_components, advanced_component_ids = init_advanced()
            with gr.Row():
                terminal = gr.Textbox(
                    label="Training Log", 
                    elem_id="terminal",
                    lines=20,
                    max_lines=50,
                    interactive=False,
                    show_copy_button=True,
                    container=True
                )
            with gr.Row():
                gallery = gr.Gallery(label="Samples", columns=6)

        with gr.TabItem("Publish") as publish_tab:
            hf_token = gr.Textbox(label="Huggingface Token")
            hf_login = gr.Button("Login")
            hf_logout = gr.Button("Logout")
            with gr.Row() as row:
                gr.Markdown("**LoRA**")
                gr.Markdown("**Upload**")
            loras = get_loras()
            with gr.Row():
                lora_rows = refresh_publish_tab()
                with gr.Column():
                    with gr.Row():
                        repo_owner = gr.Textbox(label="Account", interactive=False)
                        repo_name = gr.Textbox(label="Repository Name")
                    repo_visibility = gr.Textbox(label="Repository Visibility ('public' or 'private')", value="public")
                    upload_button = gr.Button("Upload to HuggingFace")
                    upload_button.click(
                        fn=upload_hf,
                        inputs=[
                            unet_file,
                            lora_rows,
                            repo_owner,
                            repo_name,
                            repo_visibility,
                            hf_token,
                        ]
                    )
            hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
            hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])

        with gr.TabItem("🚀 Workflow Automation"):
            gr.Markdown("### Automate your training workflow with these powerful features")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔄 Smart Training Resume")
                    gr.Markdown("Automatically detect and resume interrupted training sessions")
                    resume_scan_btn = gr.Button("Scan for Interrupted Training", variant="secondary")
                    resume_list = gr.Dropdown(label="Select Training to Resume", choices=[], interactive=True)
                    resume_btn = gr.Button("Resume Training", variant="primary")
                    resume_status = gr.Textbox(label="Resume Status", interactive=False)
                    
                with gr.Column():
                    gr.Markdown("#### 📋 Batch Training Queue")
                    gr.Markdown("Queue multiple LoRA training jobs to run sequentially")
                    queue_name = gr.Textbox(label="Queue Item Name", placeholder="character_lora_1")
                    queue_script = gr.File(label="Training Script", file_types=[".sh", ".bat"])
                    add_to_queue_btn = gr.Button("Add to Queue", variant="secondary")
                    queue_list = gr.Dataframe(
                        headers=["Name", "Status", "Added"],
                        datatype=["str", "str", "str"],
                        label="Training Queue"
                    )
                    start_queue_btn = gr.Button("Start Batch Training", variant="primary")
                    queue_status = gr.Textbox(label="Queue Status", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 Auto Dataset Optimizer")
                    gr.Markdown("Automatically resize images, generate captions, and optimize datasets")
                    dataset_folder = gr.Textbox(label="Dataset Folder Path", placeholder="C:/datasets/my_character")
                    optimize_resolution = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Target Resolution")
                    auto_caption_enable = gr.Checkbox(label="Auto-generate Captions", value=True)
                    concept_word_input = gr.Textbox(label="Concept Word (for captions)", placeholder="character_name")
                    optimize_btn = gr.Button("Optimize Dataset", variant="primary")
                    optimize_status = gr.Textbox(label="Optimization Status", interactive=False)
                    
                with gr.Column():
                    gr.Markdown("#### 🧠 Smart Parameter Suggestions")
                    gr.Markdown("Get AI-powered parameter recommendations")
                    suggest_vram = gr.Slider(minimum=4, maximum=48, value=12, step=2, label="GPU VRAM (GB)")
                    suggest_dataset_size = gr.Number(label="Dataset Size (number of images)", value=10)
                    suggest_resolution = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Image Resolution")
                    get_suggestions_btn = gr.Button("Get Smart Suggestions", variant="secondary")
                    suggestions_output = gr.JSON(label="Recommended Parameters")
            
            # Event handlers for workflow automation
            def scan_interrupted():
                interrupted = detect_interrupted_training()
                choices = [f"{item['name']} (Last checkpoint: {item['last_checkpoint']})" for item in interrupted]
                return gr.update(choices=choices)
            
            def resume_training(selected_training):
                if not selected_training:
                    return "Please select a training to resume"
                
                # Extract training name
                training_name = selected_training.split(" (")[0]
                interrupted = detect_interrupted_training()
                
                for item in interrupted:
                    if item['name'] == training_name:
                        try:
                            # Execute the training script
                            if sys.platform == "win32":
                                subprocess.Popen(item['script_path'], shell=True)
                            else:
                                subprocess.Popen(["bash", item['script_path']])
                            return f"Resumed training: {training_name}"
                        except Exception as e:
                            return f"Failed to resume training: {str(e)}"
                
                return "Training session not found"
            
            def add_to_training_queue(name, script_file):
                if not name or not script_file:
                    return "Please provide name and script file", gr.update()
                
                queue_file = resolve_path_without_quotes("training_queue.json")
                
                # Load existing queue or create new
                if os.path.exists(queue_file):
                    with open(queue_file, 'r') as f:
                        queue_data = json.load(f)
                else:
                    queue_data = {
                        'created_at': str(datetime.now()),
                        'status': 'pending',
                        'current_index': 0,
                        'items': []
                    }
                
                # Add new item
                new_item = {
                    'name': name,
                    'script_path': script_file.name if hasattr(script_file, 'name') else str(script_file),
                    'added_at': str(datetime.now()),
                    'status': 'pending'
                }
                
                queue_data['items'].append(new_item)
                
                with open(queue_file, 'w') as f:
                    json.dump(queue_data, f, indent=2)
                
                # Update display
                queue_display = [[item['name'], item['status'], item['added_at']] for item in queue_data['items']]
                
                return f"Added {name} to queue", gr.update(value=queue_display)
            
            def optimize_dataset_handler(folder, resolution, auto_caption, concept_word):
                return auto_optimize_dataset(folder, resolution, auto_caption, concept_word)
            
            def get_parameter_suggestions(vram_gb, dataset_size, resolution):
                suggestions = smart_parameter_suggestions(vram_gb, dataset_size, resolution)
                return suggestions
            
            resume_scan_btn.click(scan_interrupted, outputs=resume_list)
            resume_btn.click(resume_training, inputs=resume_list, outputs=resume_status)
            add_to_queue_btn.click(add_to_training_queue, inputs=[queue_name, queue_script], outputs=[queue_status, queue_list])
            start_queue_btn.click(process_training_queue, outputs=queue_status)
            optimize_btn.click(optimize_dataset_handler, inputs=[dataset_folder, optimize_resolution, auto_caption_enable, concept_word_input], outputs=optimize_status)
            get_suggestions_btn.click(get_parameter_suggestions, inputs=[suggest_vram, suggest_dataset_size, suggest_resolution], outputs=suggestions_output)


    publish_tab.select(refresh_publish_tab, outputs=lora_rows)
    lora_rows.select(fn=set_repo, inputs=[lora_rows], outputs=[repo_name])

    dataset_folder = gr.State()

    listeners = [
        unet_file,
        clip_file,
        t5_file,
        vae_file,
        lora_name,
        resolution,
        seed,
        workers,
        concept_sentence,
        learning_rate,
        network_dim,
        max_train_epochs,
        save_every_n_epochs,
        timestep_sampling,
        guidance_scale,
        vram,
        num_repeats,
        sample_prompts,
        sample_every_n_steps,
        lr_scheduler,
        max_train_steps,
        train_batch_size,
        reg_data_dir,
        cache_latents,
        cache_latents_to_disk,
        cache_text_encoder_outputs,
        cache_text_encoder_outputs_to_disk,
        skip_cache_check,
        vae_batch_size,
        cache_info,
        *advanced_components
    ]
    advanced_component_ids = [x.elem_id for x in advanced_components]
    original_advanced_component_values = [comp.value for comp in advanced_components]
    images.upload(
        load_captioning,
        inputs=[images, concept_sentence],
        outputs=output_components
    )
    images.clear(
        hide_captioning,
        outputs=[captioning_area, start]
    )
    max_train_epochs.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    num_repeats.change(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.upload(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    images.clear(
        fn=update_total_steps,
        inputs=[max_train_epochs, num_repeats, images],
        outputs=[total_steps]
    )
    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)
    
    # Captioning method toggle
    captioning_method.change(
        fn=toggle_captioning_method,
        inputs=[captioning_method], 
        outputs=[ollama_settings, ollama_model]
    )
    
    # Ollama model refresh
    refresh_ollama.click(
        fn=refresh_ollama_models,
        inputs=[ollama_url],
        outputs=[ollama_model]
    )
    
    start.click(fn=create_dataset, inputs=[dataset_folder, resolution, images] + caption_list, outputs=dataset_folder).then(
        fn=start_training,
        inputs=[
            unet_file,
            clip_file,
            t5_file,
            vae_file,
            lora_name,
            train_script,
            train_config,
            sample_prompts,
        ],
        outputs=terminal,
    )
    do_captioning.click(
        fn=run_captioning, 
        inputs=[images, concept_sentence, captioning_method, ollama_model, ollama_url] + caption_list, 
        outputs=caption_list
    )
    demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, repo_owner])
    
    # Auto-configure settings on load and when button is clicked
    demo.load(
        fn=auto_configure_settings,
        outputs=[gpu_info_display, vram, resolution, network_dim, learning_rate, max_train_epochs, save_every_n_epochs, workers]
    )
    auto_config_btn.click(
        fn=auto_configure_settings,
        outputs=[gpu_info_display, vram, resolution, network_dim, learning_rate, max_train_epochs, save_every_n_epochs, workers]
    )
    
    refresh.click(update, inputs=listeners, outputs=[train_script, train_config, dataset_folder])
    refresh_models_btn.click(refresh_models, outputs=[unet_file, vae_file, clip_file, t5_file])
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FluxGym - FLUX LoRA Training Application')
    parser.add_argument('--listen', '--host', type=str, default='127.0.0.1', 
                       help='Host to bind the server to (default: 127.0.0.1, use 0.0.0.0 for all interfaces)')
    parser.add_argument('--port', type=int, default=7860, 
                       help='Port to run the server on (default: 7860)')
    parser.add_argument('--share', action='store_true', 
                       help='Create a public Gradio link')
    parser.add_argument('--auth', type=str, 
                       help='Authentication in format "username:password"')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Parse authentication if provided
    auth = None
    if args.auth:
        try:
            username, password = args.auth.split(':')
            auth = (username, password)
        except ValueError:
            print("Error: Auth format should be 'username:password'")
            exit(1)
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    print(f"🚀 Starting FluxGym on {args.listen}:{args.port}")
    if args.share:
        print("🌐 Creating public share link...")
    if auth:
        print(f"🔐 Authentication enabled for user: {auth[0]}")
    
    demo.launch(
        server_name=args.listen,
        server_port=args.port,
        share=args.share,
        auth=auth,
        debug=args.debug,
        show_error=True,
        allowed_paths=[cwd]
    )
