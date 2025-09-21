import pytest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
from PIL import Image
import sys

# Add app to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="fluxgym_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_images(temp_output_dir):
    """Generate sample synthetic images for testing."""
    from tests.assets.generate_test_images import generate_test_dataset
    
    images_dir = os.path.join(temp_output_dir, "test_images")
    images = generate_test_dataset(images_dir, count=5, patterns=["geometric", "gradient"])
    return images


@pytest.fixture
def mock_gpu_info():
    """Mock GPU detection for consistent testing."""
    mock_info = {
        'name': 'Test GPU RTX 4090',
        'vram_gb': 24,
        'compute_capability': '8.9',
        'driver_version': '12.1',
        'cuda_available': True,
        'recommended_settings': {
            'vram_setting': '20G',
            'batch_size': 1,
            'network_dim': 32,
            'learning_rate': '1e-4',
            'max_train_epochs': 16,
            'save_every_n_epochs': 4,
            'resolution': 1024,
            'workers': 4,
            'gradient_accumulation': 1,
            'optimizer': 'adamw8bit',
            'mixed_precision': 'bf16',
            'fp8_base': True,
            'cache_latents': True,
            'cache_text_encoder_outputs': True
        }
    }
    return mock_info


@pytest.fixture
def mock_florence_model():
    """Mock Florence model for captioning tests."""
    class MockProcessor:
        def __call__(self, text, images, return_tensors="pt"):
            return {
                "input_ids": MagicMock(),
                "pixel_values": MagicMock()
            }
        
        def batch_decode(self, ids, skip_special_tokens=False):
            return ["<DETAILED_CAPTION>a synthetic test image with geometric patterns</DETAILED_CAPTION>"]
        
        def post_process_generation(self, text, task, image_size):
            return {"<DETAILED_CAPTION>": "a synthetic test image with geometric patterns"}
    
    class MockModel:
        def __init__(self):
            self.device = "cpu"
        
        def to(self, device):
            self.device = device
            return self
        
        def generate(self, input_ids, pixel_values, max_new_tokens=256, num_beams=3):
            return MagicMock()
    
    return MockModel(), MockProcessor()


@pytest.fixture
def mock_available_models():
    """Mock available model files for testing."""
    return {
        'unet': ['flux1-dev-fp8.safetensors', 'test-model.safetensors'],
        'clip': ['clip_l.safetensors', 't5xxl_fp16.safetensors'],
        't5': ['t5xxl_fp16.safetensors', 'qwen_2.5_vl_fp16.safetensors'],
        'vae': ['ae.safetensors', 'flux-vae.safetensors']
    }


@pytest.fixture
def mock_torch():
    """Mock torch operations for testing without GPU."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.get_device_name', return_value='Test GPU'), \
         patch('torch.cuda.get_device_properties') as mock_props, \
         patch('torch.cuda.get_device_capability', return_value=(8, 9)), \
         patch('torch.cuda.empty_cache'):
        
        mock_props.return_value.total_memory = 24 * 1024**3  # 24GB
        yield


@pytest.fixture
def sample_training_config():
    """Sample training configuration for testing."""
    return {
        'unet_file': 'flux1-dev-fp8.safetensors',
        'clip_file': 'clip_l.safetensors',
        't5_file': 't5xxl_fp16.safetensors',
        'vae_file': 'ae.safetensors',
        'output_name': 'test_lora',
        'resolution': 512,
        'seed': 42,
        'workers': 2,
        'learning_rate': '1e-4',
        'network_dim': 16,
        'max_train_epochs': 10,
        'save_every_n_epochs': 5,
        'timestep_sampling': 'shift',
        'guidance_scale': 1.0,
        'vram': '20G',
        'sample_prompts': 'test prompt 1\ntest prompt 2',
        'sample_every_n_steps': 100,
        'lr_scheduler': 'cosine',
        'max_train_steps': None,
        'train_batch_size': 1,
        'reg_data_dir': '',
        'cache_latents': True,
        'cache_latents_to_disk': False,
        'cache_text_encoder_outputs': True,
        'cache_text_encoder_outputs_to_disk': False,
        'skip_cache_check': False,
        'vae_batch_size': 1,
        'cache_info': False
    }
