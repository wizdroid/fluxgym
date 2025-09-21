#!/usr/bin/env python3
"""
Test the main app with debug output
"""

import os
import sys

print("🔧 Setting up environment...")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))

print("📦 Importing required modules...")

try:
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
    print("✅ Basic imports successful")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    print("✅ Transformers imported")
except Exception as e:
    print(f"⚠️ Transformers import warning: {e}")

try:
    from gradio_logsview import LogsView, LogsViewRunner
    print("✅ Gradio LogsView imported")
except Exception as e:
    print(f"❌ Gradio LogsView failed: {e}")
    # This might be the issue - let's continue without it for now

try:
    from huggingface_hub import hf_hub_download, HfApi
    print("✅ HuggingFace Hub imported")
except Exception as e:
    print(f"⚠️ HuggingFace Hub warning: {e}")

try:
    from library import flux_train_utils, huggingface_util
    print("✅ SD-Scripts library imported")
except Exception as e:
    print(f"❌ SD-Scripts library failed: {e}")

try:
    from argparse import Namespace
    import toml
    import re
    print("✅ Additional utilities imported")
except Exception as e:
    print(f"❌ Additional utilities failed: {e}")

try:
    import train_network
    print("✅ Train network imported")
except Exception as e:
    print(f"❌ Train network failed: {e}")

print("\n🎯 Testing core functions...")

# Test GPU detection
def test_detect_gpu_info():
    print("Testing GPU detection...")
    try:
        gpu_info = {
            'name': 'Unknown',
            'vram_gb': 8,
            'compute_capability': None,
            'driver_version': None,
            'cuda_available': False,
            'recommended_settings': {}
        }
        
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['name'] = torch.cuda.get_device_name(0)
            
            # Get VRAM information
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_info['vram_gb'] = round(vram_bytes / (1024**3))
            
            # Get compute capability
            major, minor = torch.cuda.get_device_capability(0)
            gpu_info['compute_capability'] = f"{major}.{minor}"
        
        print(f"✅ GPU Detection: {gpu_info}")
        return gpu_info
    except Exception as e:
        print(f"❌ GPU Detection failed: {e}")
        return None

gpu_info = test_detect_gpu_info()

print("\n🚀 Attempting to start main application...")

try:
    # Try importing and running the main app
    import sys
    import importlib.util
    
    # Load app.py as module
    spec = importlib.util.spec_from_file_location("app", "app.py")
    app_module = importlib.util.module_from_spec(spec)
    
    print("📝 App module loaded, executing...")
    
    # This will execute the app.py file
    spec.loader.exec_module(app_module)
    
except Exception as e:
    print(f"❌ Main app execution failed: {e}")
    import traceback
    traceback.print_exc()

print("🏁 Debug test completed")