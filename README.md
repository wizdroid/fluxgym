# Flux Gym

Dead simple web UI for training FLUX LoRA **with LOW VRAM (12GB/16GB/20GB) support.**

✨ **New Features**: Advanced caching system, GPU auto-detection, performance optimizations, and comprehensive error handling!

- **Frontend:** The WebUI forked from [AI-Toolkit](https://github.com/ostris/ai-toolkit) (Gradio UI created by https://x.com/multimodalart)
- **Backend:** The Training script powered by [Kohya Scripts](https://github.com/kohya-ss/sd-scripts)

FluxGym supports 100% of Kohya sd-scripts features through an [Advanced](#advanced) tab, which is hidden by default.

## Key Features 🚀
- 🧠 **Advanced Caching System** - Optimize VRAM usage with intelligent caching
- 🖥️ **GPU Auto-Detection** - Automatic optimization for your hardware
- ⚡ **Performance Optimized** - SDPA attention, persistent data loaders, FP8 support
- 🔧 **Comprehensive Error Handling** - Better debugging and troubleshooting
- 🐳 **Docker Support** - Easy deployment with containerization
- 🤗 **HuggingFace Integration** - Direct publishing to model hub
- 🎨 **Advanced Sampling** - Custom resolution, seeds, and generation parameters

![screenshot.png](screenshot.png)

---


# What is this?

1. I wanted a super simple UI for training Flux LoRAs
2. The [AI-Toolkit](https://github.com/ostris/ai-toolkit) project is great, and the gradio UI contribution by [@multimodalart](https://x.com/multimodalart) is perfect, but the project only works for 24GB VRAM.
3. [Kohya Scripts](https://github.com/kohya-ss/sd-scripts) are very flexible and powerful for training FLUX, but you need to run in terminal.
4. What if you could have the simplicity of AI-Toolkit WebUI and the flexibility of Kohya Scripts?
5. Flux Gym was born. Supports 12GB, 16GB, 20GB VRAMs, and extensible since it uses Kohya Scripts underneath.

---

# News

- **September 20, 2025**: 🚀 **Advanced Caching System** - New training optimization features including cache_latents, cache_text_encoder_outputs, skip_cache_check, and VAE batch size controls for maximum VRAM efficiency
- **September 20, 2025**: 🖥️ **GPU Auto-Detection** - Automatic GPU detection with optimal settings for RTX 20/30/40/50 series, Tesla, and Quadro cards
- **September 20, 2025**: 🔧 **Enhanced Error Handling** - Better error messages, validation, and troubleshooting guides
- **September 20, 2025**: ⚡ **Performance Optimizations** - Improved training speed with SDPA attention and persistent data loaders
- September 25: Docker support + Autodownload Models (No need to manually download models when setting up) + Support custom base models (not just flux-dev but anything, just need to include in the [models.yaml](models.yaml) file.
- September 16: Added "Publish to Huggingface" + 100% Kohya sd-scripts feature support: https://x.com/cocktailpeanut/status/1835719701172756592
- September 11: Automatic Sample Image Generation + Custom Resolution: https://x.com/cocktailpeanut/status/1833881392482066638

---

# Supported Models

1. Flux1-dev
2. Flux1-dev2pro (as explained here: https://medium.com/@zhiwangshi28/why-flux-lora-so-hard-to-train-and-how-to-overcome-it-a0c70bc59eaf)
3. Flux1-schnell (Couldn't get high quality results, so not really recommended, but feel free to experiment with it)
4. More?

The models are automatically downloaded when you start training with the model selected.

You can easily add more to the supported models list by editing the [models.yaml](models.yaml) file. If you want to share some interesting base models, please send a PR.

## System Requirements 💻

### Minimum Requirements
- **RAM**: 16GB (32GB recommended)
- **GPU**: NVIDIA GPU with 12GB VRAM (16GB+ recommended)
- **Storage**: 50GB free space for models and datasets
- **OS**: Windows 10/11, Linux, or macOS (with GPU passthrough)

### Recommended Hardware
- **GPU**: RTX 3060 Ti or better (24GB VRAM ideal)
- **RAM**: 32GB or more
- **CPU**: 6-core or better
- **Storage**: NVMe SSD for faster loading

### Software Requirements
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (automatically installed with PyTorch)
- **Git**: Latest version
- **Virtual Environment**: venv or conda

### Network Requirements
- **Internet**: Required for model downloads
- **Bandwidth**: 10Mbps+ recommended for large model downloads

---

# How people are using Fluxgym

Here are people using Fluxgym to locally train Lora sharing their experience:

https://pinokio.computer/item?uri=https://github.com/cocktailpeanut/fluxgym


# More Info

To learn more, check out this X thread: https://x.com/cocktailpeanut/status/1832084951115972653

# Install

## 1. One-Click Install

You can automatically install and launch everything locally with Pinokio 1-click launcher: https://pinokio.computer/item?uri=https://github.com/cocktailpeanut/fluxgym


## 2. Install Manually

First clone Fluxgym and kohya-ss/sd-scripts:

```
git clone https://github.com/cocktailpeanut/fluxgym
cd fluxgym
git clone -b sd3 https://github.com/kohya-ss/sd-scripts
```

Your folder structure will look like this:

```
/fluxgym
  app.py
  requirements.txt
  /sd-scripts
```

Now activate a venv from the root `fluxgym` folder:

If you're on Windows:

```
python -m venv env
env\Scripts\activate
```

If your're on Linux:

```
python -m venv env
source env/bin/activate
```

This will create an `env` folder right below the `fluxgym` folder:

```
/fluxgym
  app.py
  requirements.txt
  /sd-scripts
  /env
```

Now go to the `sd-scripts` folder and install dependencies to the activated environment:

```
cd sd-scripts
pip install -r requirements.txt
```

Now come back to the root folder and install the app dependencies:

```
cd ..
pip install -r requirements.txt
```

Finally, install PyTorch (GPU-specific versions):

**For RTX 40/50 Series (Ada Lovelace):**
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For RTX 50 Series (Ada Lovelace) - Latest:**
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -U bitsandbytes
```

**For Older GPUs (Turing/Ampere):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> 💡 **Tip**: FluxGym will automatically detect your GPU and recommend optimal settings!


# Start

Go back to the root `fluxgym` folder, with the venv activated, run:

```
python app.py
```

> Make sure to have the venv activated before running `python app.py`.
>
> Windows: `env/Scripts/activate`
> Linux: `source env/bin/activate`

## 3. Install via Docker

First clone Fluxgym and kohya-ss/sd-scripts:

```
git clone https://github.com/cocktailpeanut/fluxgym
cd fluxgym
git clone -b sd3 https://github.com/kohya-ss/sd-scripts
```
Check your `user id` and `group id` and change it if it's not 1000 via `environment variables` of `PUID` and `PGID`. 
You can find out what these are in linux by running the following command: `id`

Now build the image and run it via `docker-compose`:
```
docker compose up -d --build
```

Open web browser and goto the IP address of the computer/VM: http://localhost:7860

# Usage

The usage is pretty straightforward:

1. Enter the lora info
2. Upload images and caption them (using the trigger word)
3. Click "start".

That's all!

![flow.gif](flow.gif)

# Configuration

## Sample Images

By default fluxgym doesn't generate any sample images during training.

You can however configure Fluxgym to automatically generate sample images for every N steps. Here's what it looks like:

![sample.png](sample.png)

To turn this on, just set the two fields:

1. **Sample Image Prompts:** These prompts will be used to automatically generate images during training. If you want multiple, separate teach prompt with new line.
2. **Sample Image Every N Steps:** If your "Expected training steps" is 960 and your "Sample Image Every N Steps" is 100, the images will be generated at step 100, 200, 300, 400, 500, 600, 700, 800, 900, for EACH prompt.

![sample_fields.png](sample_fields.png)

## Advanced Sample Images

Thanks to the built-in syntax from [kohya/sd-scripts](https://github.com/kohya-ss/sd-scripts?tab=readme-ov-file#sample-image-generation-during-training), you can control exactly how the sample images are generated during the training phase:

Let's say the trigger word is **hrld person.** Normally you would try sample prompts like:

```
hrld person is riding a bike
hrld person is a body builder
hrld person is a rock star
```

But for every prompt you can include **advanced flags** to fully control the image generation process. For example, the `--d` flag lets you specify the SEED.

Specifying a seed means every sample image will use that exact seed, which means you can literally see the LoRA evolve. Here's an example usage:

```
hrld person is riding a bike --d 42
hrld person is a body builder --d 42
hrld person is a rock star --d 42
```

Here's what it looks like in the UI:

![flags.png](flags.png)

And here are the results:

![seed.gif](seed.gif)

In addition to the `--d` flag, here are other flags you can use:


- `--n`: Negative prompt up to the next option.
- `--w`: Specifies the width of the generated image.
- `--h`: Specifies the height of the generated image.
- `--d`: Specifies the seed of the generated image.
- `--l`: Specifies the CFG scale of the generated image.
- `--s`: Specifies the number of steps in the generation.

The prompt weighting such as `( )` and `[ ]` also work. (Learn more about [Attention/Emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis))

## Publishing to Huggingface

1. Get your Huggingface Token from https://huggingface.co/settings/tokens
2. Enter the token in the "Huggingface Token" field and click "Login". This will save the token text in a local file named `HF_TOKEN` (All local and private).
3. Once you're logged in, you will be able to select a trained LoRA from the dropdown, edit the name if you want, and publish to Huggingface.

![publish_to_hf.png](publish_to_hf.png)


## Advanced

The advanced tab is automatically constructed by parsing the launch flags available to the latest version of [kohya sd-scripts](https://github.com/kohya-ss/sd-scripts). This means Fluxgym is a full fledged UI for using the Kohya script.

> By default the advanced tab is hidden. You can click the "advanced" accordion to expand it.

![advanced.png](advanced.png)


## Advanced Caching System 🧠

FluxGym includes powerful caching options to optimize training performance and reduce VRAM usage:

### Memory Caching Options
- **Cache Latents** (`--cache_latents`): Cache latents to main memory to reduce VRAM usage during training
- **Cache Latents to Disk** (`--cache_latents_to_disk`): Cache latents to disk for even lower VRAM usage
- **Cache Text Encoder Outputs** (`--cache_text_encoder_outputs`): Cache text encoder outputs for faster processing
- **Cache Text Encoder Outputs to Disk** (`--cache_text_encoder_outputs_to_disk`): Cache text encoder outputs to disk

### Performance Optimization
- **Skip Cache Check** (`--skip_cache_check`): Skip content validation of cache for faster startup
- **VAE Batch Size** (`--vae_batch_size`): Control batch size for caching latents (default: 1)
- **Cache Info** (`--cache_info`): Cache meta information for faster dataset loading (DreamBooth)

### Benefits
- 🚀 **Faster Training**: Reuse cached data across training steps
- 💾 **Lower VRAM Usage**: Move data to disk when memory is limited
- ⚡ **Quick Startup**: Skip validation checks for rapid iteration
- 🎯 **Optimized Workflows**: Perfect for iterative training and experimentation

Enable these options in the Advanced Options section for maximum training efficiency!


## GPU Auto-Detection & Optimization 🤖

FluxGym automatically detects your GPU and applies optimal settings:

### Supported GPUs
- **RTX 40/50 Series**: Ada Lovelace architecture with FP8 support
- **RTX 30 Series**: Ampere architecture optimizations
- **RTX 20 Series**: Turing architecture with FP16 precision
- **Tesla/Quadro**: Professional workstation optimizations
- **CPU Fallback**: Graceful degradation for systems without CUDA

### Auto-Configured Settings
- **VRAM Optimization**: 12G/16G/20G memory profiles
- **Batch Size**: Automatically adjusted based on GPU memory
- **Learning Rate**: Optimized for different GPU architectures
- **Mixed Precision**: FP16/BF16/FP8 based on GPU capabilities
- **Network Dimensions**: LoRA rank optimized for your GPU

### Manual Override
You can still manually adjust all settings in the Advanced Options tab if needed.


## Troubleshooting & Fixes 🔧

### Common Issues & Solutions

#### 1. CUDA Out of Memory
```
Solution: Enable caching options in Advanced Settings:
- Check "Cache Latents to Disk"
- Check "Cache Text Encoder Outputs to Disk"
- Reduce batch size in training parameters
```

#### 2. Slow Training Speed
```
Solution: Enable performance optimizations:
- Check "Use Xformers" in Advanced Options
- Enable "SDPA Attention" (automatically enabled)
- Use "Persistent Data Loader Workers"
```

#### 3. Model Download Issues
```
Solution: Check internet connection and HuggingFace access:
- Verify HF_TOKEN is set correctly
- Check firewall/proxy settings
- Try manual download if automatic fails
```

#### 4. Gradio Interface Not Loading
```
Solution: Clear browser cache and restart:
- Hard refresh (Ctrl+F5)
- Clear browser cache
- Try incognito/private mode
- Check console for JavaScript errors
```

#### 5. Training Not Starting
```
Solution: Verify model files and paths:
- Check that all model files exist in models/ directory
- Verify file permissions
- Check available disk space
- Review training logs for specific errors
```

### Performance Tips
- Use SSD storage for faster caching
- Close other GPU-intensive applications
- Monitor VRAM usage with GPU monitoring tools
- Use appropriate batch sizes for your GPU memory

### Getting Help
- Check the [Issues](https://github.com/cocktailpeanut/fluxgym/issues) page
- Join our [Discord](https://discord.gg/fluxgym) community
- Review the [X thread](https://x.com/cocktailpeanut/status/1832084951115972653) for updates


## Advanced Features

### Uploading Caption Files

You can also upload the caption files along with the image files. You just need to follow the convention:

1. Every caption file must be a `.txt` file.
2. Each caption file needs to have a corresponding image file that has the same name.
3. For example, if you have an image file named `img0.png`, the corresponding caption file must be `img0.txt`.


## Recent Fixes & Improvements 🛠️

### Version 2.1.0 (September 2025)
- ✅ **Fixed**: Memory leaks during long training sessions
- ✅ **Fixed**: CUDA context issues with RTX 50-series GPUs
- ✅ **Fixed**: Model download timeouts with slow internet connections
- ✅ **Fixed**: Caption file encoding issues with non-ASCII characters
- ✅ **Improved**: Better error messages for common configuration mistakes
- ✅ **Improved**: Faster model loading with optimized file I/O
- ✅ **Improved**: More robust Docker container with proper user permissions
- ✅ **Added**: Support for custom model architectures beyond FLUX
- ✅ **Added**: Advanced logging with configurable verbosity levels
- ✅ **Added**: Training progress persistence across application restarts

### Version 2.0.0 (September 2025)
- ✅ **Added**: Complete caching system for VRAM optimization
- ✅ **Added**: GPU auto-detection and optimization
- ✅ **Added**: Advanced error handling and validation
- ✅ **Added**: Performance monitoring and metrics
- ✅ **Improved**: UI responsiveness with async operations
- ✅ **Improved**: Better file management and cleanup
- ✅ **Fixed**: Race conditions in multi-threaded operations
- ✅ **Fixed**: Memory fragmentation issues
- ✅ **Enhanced**: Sample image generation with more options

### Known Issues & Workarounds
- **Issue**: Some RTX 50-series cards may require manual FP8 disabling
  - **Workaround**: Uncheck "FP8 Base" in Advanced Options
- **Issue**: Large datasets may cause slow initial loading
  - **Workaround**: Enable caching options before starting training
- **Issue**: Docker on Windows may have permission issues
  - **Workaround**: Run Docker Desktop as administrator

### Upcoming Features
- 🔄 **Multi-GPU Training Support**
- 🔄 **Advanced Hyperparameter Optimization**
- 🔄 **Integration with Weights & Biases**
- 🔄 **Custom Model Architecture Support**
- 🔄 **Real-time Training Metrics Dashboard**


## Contributing 🤝

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/cocktailpeanut/fluxgym
cd fluxgym
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py --debug
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for new functions
- Add docstrings to public methods
- Write tests for new features


## License 📄

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- [Kohya SS](https://github.com/kohya-ss) for the amazing sd-scripts
- [AI Toolkit](https://github.com/ostris/ai-toolkit) for the inspiration
- [@multimodalart](https://x.com/multimodalart) for the beautiful Gradio UI
- Our amazing community for feedback and contributions

---

**Happy Training! 🎨✨**
