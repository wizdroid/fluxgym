import pytest
from unittest.mock import patch, MagicMock, Mock
from PIL import Image
import tempfile
import os

# Import app functions for testing
import app


class TestCaptioning:
    """Test captioning functionality including fallbacks."""
    
    def test_ollama_captioning_success(self, sample_images):
        """Test successful Ollama captioning."""
        images = [img['image'] for img in sample_images[:2]]
        concept_sentence = "test concept"
        captions = ["", ""]  # Initial empty captions
        
        # Mock successful Ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "a colorful geometric pattern"}
        
        with patch('app.check_ollama_connection', return_value=True), \
             patch('requests.post', return_value=mock_response), \
             patch('app.encode_image_to_base64', return_value="fake_base64"):
            
            # Test the generator
            result_generator = app.run_captioning(
                images, concept_sentence, "Ollama", "test_model", "http://localhost:11434", *captions
            )
            
            # Get final result
            final_captions = None
            for result in result_generator:
                final_captions = result
            
            assert final_captions is not None
            assert len(final_captions) == 2
            assert "test concept" in final_captions[0]
            assert "geometric pattern" in final_captions[0]
    
    def test_ollama_connection_failure(self, sample_images):
        """Test Ollama captioning when connection fails."""
        images = [sample_images[0]['image']]
        captions = [""]
        
        with patch('app.check_ollama_connection', return_value=False):
            result_generator = app.run_captioning(
                images, "test", "Ollama", "test_model", "http://localhost:11434", *captions
            )
            
            # Should return original captions when connection fails
            final_result = None
            for result in result_generator:
                final_result = result
            
            # Should return original captions (might be empty)
            assert final_result == captions
    
    def test_florence_captioning_success(self, sample_images, mock_florence_model):
        """Test successful Florence-2 captioning."""
        images = [sample_images[0]['image']]
        concept_sentence = "test subject"
        captions = [""]
        
        mock_model, mock_processor = mock_florence_model
        
        with patch('app.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
             patch('app.AutoProcessor.from_pretrained', return_value=mock_processor), \
             patch('torch.cuda.is_available', return_value=True):
            
            result_generator = app.run_captioning(
                images, concept_sentence, "Florence-2", None, None, *captions
            )
            
            final_captions = None
            for result in result_generator:
                final_captions = result
            
            assert final_captions is not None
            assert len(final_captions) == 1
            assert "test subject" in final_captions[0]
            assert "synthetic test image" in final_captions[0]
    
    def test_florence_model_loading_failure(self, sample_images):
        """Test Florence-2 fallback when model loading fails."""
        images = [sample_images[0]['image']]
        concept_sentence = "test subject"
        captions = [""]
        
        # Mock model loading failure
        with patch('app.AutoProcessor.from_pretrained', side_effect=Exception("Model not found")), \
             patch('torch.cuda.is_available', return_value=True):
            
            result_generator = app.run_captioning(
                images, concept_sentence, "Florence-2", None, None, *captions
            )
            
            final_captions = None
            for result in result_generator:
                final_captions = result
            
            assert final_captions is not None
            assert len(final_captions) == 1
            # Should use fallback caption
            assert "test subject" in final_captions[0]
    
    def test_florence_generate_method_missing(self, sample_images):
        """Test Florence-2 fallback when generate method is missing (the original error)."""
        images = [sample_images[0]['image']]
        concept_sentence = "test subject"
        captions = [""]
        
        # Create a mock model without generate method
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        # Explicitly remove generate method
        if hasattr(mock_model, 'generate'):
            delattr(mock_model, 'generate')
        # Also remove language_model.generate
        mock_model.language_model = Mock()
        if hasattr(mock_model.language_model, 'generate'):
            delattr(mock_model.language_model, 'generate')
        
        mock_processor = Mock()
        mock_processor.return_value = {"input_ids": Mock(), "pixel_values": Mock()}
        
        with patch('app.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
             patch('app.AutoProcessor.from_pretrained', return_value=mock_processor), \
             patch('torch.cuda.is_available', return_value=True):
            
            result_generator = app.run_captioning(
                images, concept_sentence, "Florence-2", None, None, *captions
            )
            
            final_captions = None
            for result in result_generator:
                final_captions = result
            
            assert final_captions is not None
            assert len(final_captions) == 1
            # Should use fallback caption when generate fails
            assert "test subject" in final_captions[0]
    
    def test_image_loading_failure(self, temp_output_dir):
        """Test captioning when image loading fails."""
        # Create a fake image path that doesn't exist
        fake_image = os.path.join(temp_output_dir, "nonexistent.jpg")
        concept_sentence = "test subject"
        captions = [""]
        
        mock_model, mock_processor = Mock(), Mock()
        mock_model.to.return_value = mock_model
        
        with patch('app.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
             patch('app.AutoProcessor.from_pretrained', return_value=mock_processor), \
             patch('torch.cuda.is_available', return_value=True):
            
            result_generator = app.run_captioning(
                [fake_image], concept_sentence, "Florence-2", None, None, *captions
            )
            
            final_captions = None
            for result in result_generator:
                final_captions = result
            
            assert final_captions is not None
            assert len(final_captions) == 1
            # Should use fallback when image loading fails
            assert final_captions[0] == concept_sentence or "image"


class TestOllamaUtilities:
    """Test Ollama utility functions."""
    
    def test_check_ollama_connection_success(self):
        """Test successful Ollama connection check."""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('requests.get', return_value=mock_response):
            result = app.check_ollama_connection()
            assert result is True
    
    def test_check_ollama_connection_failure(self):
        """Test failed Ollama connection check."""
        with patch('requests.get', side_effect=Exception("Connection failed")):
            result = app.check_ollama_connection()
            assert result is False
    
    def test_get_ollama_models_success(self):
        """Test successful Ollama model retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llava:latest"},
                {"name": "vision-model:latest"}
            ]
        }
        
        with patch('requests.get', return_value=mock_response):
            models = app.get_ollama_models()
            assert len(models) >= 1
            assert any('llava' in model.lower() for model in models)
    
    def test_encode_image_to_base64(self, sample_images):
        """Test image encoding to base64."""
        image_path = sample_images[0]['image']
        
        result = app.encode_image_to_base64(image_path)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Basic check that it looks like base64
        import base64
        try:
            base64.b64decode(result)
            assert True
        except Exception:
            assert False, "Result is not valid base64"
