import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image

# Import app functions for testing
import app


class TestModelScanning:
    """Test model directory scanning functionality."""
    
    def test_get_available_models(self, temp_output_dir, mock_available_models):
        """Test model directory scanning."""
        # Create mock model directories
        models_dir = os.path.join(temp_output_dir, "models")
        for model_type, files in mock_available_models.items():
            type_dir = os.path.join(models_dir, model_type)
            os.makedirs(type_dir, exist_ok=True)
            for filename in files:
                filepath = os.path.join(type_dir, filename)
                with open(filepath, 'w') as f:
                    f.write("dummy model file")
        
        # Mock the resolve_path_without_quotes function
        with patch('app.resolve_path_without_quotes', return_value=models_dir):
            available = app.get_available_models()
            
            assert 'unet' in available
            assert 'clip' in available
            assert 't5' in available
            assert 'vae' in available
            
            # Check that files are detected
            assert len(available['unet']) >= 1
            assert 'flux1-dev-fp8.safetensors' in available['unet']


class TestGPUDetection:
    """Test GPU detection and optimization."""
    
    def test_detect_gpu_info(self, mock_torch, mock_gpu_info):
        """Test GPU detection functionality."""
        gpu_info = app.detect_gpu_info()
        
        assert gpu_info['cuda_available'] is True
        assert 'name' in gpu_info
        assert 'vram_gb' in gpu_info
        assert gpu_info['vram_gb'] > 0
    
    def test_get_optimal_settings(self, mock_gpu_info):
        """Test optimal settings generation."""
        settings = app.get_optimal_settings(mock_gpu_info)
        
        assert 'vram_setting' in settings
        assert 'batch_size' in settings
        assert 'network_dim' in settings
        assert 'learning_rate' in settings
        assert settings['batch_size'] >= 1
        assert settings['network_dim'] >= 4


class TestScriptGeneration:
    """Test training script generation."""
    
    def test_gen_sh_basic(self, sample_training_config):
        """Test basic script generation."""
        config = sample_training_config
        
        script = app.gen_sh(
            config['unet_file'],
            config['clip_file'], 
            config['t5_file'],
            config['vae_file'],
            config['output_name'],
            config['resolution'],
            config['seed'],
            config['workers'],
            config['learning_rate'],
            config['network_dim'],
            config['max_train_epochs'],
            config['save_every_n_epochs'],
            config['timestep_sampling'],
            config['guidance_scale'],
            config['vram'],
            config['sample_prompts'],
            config['sample_every_n_steps'],
            config['lr_scheduler'],
            config['max_train_steps'],
            config['train_batch_size'],
            config['reg_data_dir'],
            config['cache_latents'],
            config['cache_latents_to_disk'],
            config['cache_text_encoder_outputs'],
            config['cache_text_encoder_outputs_to_disk'],
            config['skip_cache_check'],
            config['vae_batch_size'],
            config['cache_info']
        )
        
        assert 'accelerate launch' in script
        assert 'flux_train_network.py' in script
        assert f"--seed {config['seed']}" in script
        assert f"--network_dim {config['network_dim']}" in script
        assert f"--learning_rate {config['learning_rate']}" in script
    
    def test_caching_flags_inclusion(self, sample_training_config):
        """Test that caching flags are properly included."""
        config = sample_training_config.copy()
        config['cache_latents'] = True
        config['cache_text_encoder_outputs'] = True
        config['skip_cache_check'] = True
        
        script = app.gen_sh(
            config['unet_file'], config['clip_file'], config['t5_file'], config['vae_file'],
            config['output_name'], config['resolution'], config['seed'], config['workers'],
            config['learning_rate'], config['network_dim'], config['max_train_epochs'],
            config['save_every_n_epochs'], config['timestep_sampling'], config['guidance_scale'],
            config['vram'], config['sample_prompts'], config['sample_every_n_steps'],
            config['lr_scheduler'], config['max_train_steps'], config['train_batch_size'],
            config['reg_data_dir'], config['cache_latents'], config['cache_latents_to_disk'],
            config['cache_text_encoder_outputs'], config['cache_text_encoder_outputs_to_disk'],
            config['skip_cache_check'], config['vae_batch_size'], config['cache_info']
        )
        
        assert '--cache_latents' in script
        assert '--cache_text_encoder_outputs' in script
        assert '--skip_cache_check' in script
    
    def test_scheduler_inclusion(self, sample_training_config):
        """Test that learning rate scheduler is included when specified."""
        config = sample_training_config.copy()
        config['lr_scheduler'] = 'cosine_with_restarts'
        
        script = app.gen_sh(
            config['unet_file'], config['clip_file'], config['t5_file'], config['vae_file'],
            config['output_name'], config['resolution'], config['seed'], config['workers'],
            config['learning_rate'], config['network_dim'], config['max_train_epochs'],
            config['save_every_n_epochs'], config['timestep_sampling'], config['guidance_scale'],
            config['vram'], config['sample_prompts'], config['sample_every_n_steps'],
            config['lr_scheduler'], config['max_train_steps'], config['train_batch_size'],
            config['reg_data_dir'], config['cache_latents'], config['cache_latents_to_disk'],
            config['cache_text_encoder_outputs'], config['cache_text_encoder_outputs_to_disk'],
            config['skip_cache_check'], config['vae_batch_size'], config['cache_info']
        )
        
        assert '--lr_scheduler cosine_with_restarts' in script


class TestDatasetCreation:
    """Test dataset creation functionality."""
    
    def test_create_dataset(self, sample_images, temp_output_dir):
        """Test dataset creation with images and captions."""
        destination = os.path.join(temp_output_dir, "dataset")
        
        # Prepare inputs: images list + captions
        image_paths = [img['image'] for img in sample_images]
        captions = [f"test caption {i}" for i in range(len(sample_images))]
        
        # Call create_dataset function
        result_dir = app.create_dataset(destination, 512, image_paths, *captions)
        
        assert os.path.exists(result_dir)
        assert result_dir == destination
        
        # Check that images and captions were created
        created_files = os.listdir(destination)
        image_files = [f for f in created_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        caption_files = [f for f in created_files if f.endswith('.txt')]
        
        assert len(image_files) > 0
        assert len(caption_files) > 0
        assert len(image_files) == len(caption_files)
        
        # Check caption content
        for caption_file in caption_files:
            caption_path = os.path.join(destination, caption_file)
            with open(caption_path, 'r') as f:
                content = f.read().strip()
                assert content.startswith('test caption')
    
    def test_image_resizing(self, sample_images, temp_output_dir):
        """Test that images are properly resized."""
        destination = os.path.join(temp_output_dir, "dataset_resize")
        target_size = 768
        
        image_paths = [sample_images[0]['image']]  # Test with one image
        captions = ["resize test"]
        
        app.create_dataset(destination, target_size, image_paths, *captions)
        
        # Check resized image
        created_files = os.listdir(destination)
        image_files = [f for f in created_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        assert len(image_files) == 1
        
        resized_image_path = os.path.join(destination, image_files[0])
        with Image.open(resized_image_path) as img:
            width, height = img.size
            # Check that the larger dimension matches target_size
            assert max(width, height) == target_size


class TestTOMLGeneration:
    """Test TOML configuration generation."""
    
    def test_gen_toml(self):
        """Test TOML generation for dataset configuration."""
        dataset_folder = "test/dataset"
        resolution = 512
        class_tokens = "test_subject"
        num_repeats = 10
        
        toml_content = app.gen_toml(dataset_folder, resolution, class_tokens, num_repeats)
        
        assert '[general]' in toml_content
        assert '[[datasets]]' in toml_content
        assert f'resolution = {resolution}' in toml_content
        assert f"class_tokens = '{class_tokens}'" in toml_content
        assert f'num_repeats = {num_repeats}' in toml_content
        assert 'shuffle_caption = false' in toml_content
