# FluxGym Test Suite

This directory contains a comprehensive test suite for FluxGym's core functionality.

## Test Structure

- `conftest.py` - Pytest fixtures and test configuration
- `test_core_functionality.py` - Core application tests (model scanning, GPU detection, script generation)
- `test_captioning.py` - Image captioning tests (Florence-2, Ollama, fallbacks)
- `assets/` - Test assets and utilities
- `assets/generate_test_images.py` - Synthetic image generator for testing

## Ethical Testing Approach

### Synthetic Test Images

This test suite uses **synthetic geometric pattern images** instead of real photographs to:

- ✅ Avoid privacy concerns with demographic content
- ✅ Ensure consistent, reproducible test results
- ✅ Eliminate copyright/licensing issues
- ✅ Focus testing on technical functionality rather than content

The synthetic images include:
- **Geometric patterns** - Circles, rectangles, triangles in various colors
- **Gradients** - Smooth color transitions for testing image processing
- **Noise patterns** - Random pixel patterns for edge case testing
- **Mandala designs** - Symmetric patterns for testing symmetry detection

### Using Your Own Images

If you want to test with your own images:

1. Create a `tests/assets/custom_images/` directory
2. Add your images (PNG/JPG format)
3. Add corresponding `.txt` caption files with the same base name
4. Modify test fixtures in `conftest.py` to use your custom images

**Important**: Only use images you own or have permission to use for testing.

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-mock pillow

# Ensure you're in the FluxGym virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m captioning    # Captioning tests only
```

### Run Specific Test Files

```bash
# Test core functionality
pytest tests/test_core_functionality.py

# Test captioning features
pytest tests/test_captioning.py

# Test specific function
pytest tests/test_core_functionality.py::TestScriptGeneration::test_caching_flags_inclusion
```

### Generate Test Images

```bash
# Generate synthetic test images
python tests/assets/generate_test_images.py --output tests/assets/test_images --count 10

# Generate specific patterns
python tests/assets/generate_test_images.py --patterns geometric mandala --count 5
```

## Test Coverage

### Core Functionality Tests

- ✅ **Model Directory Scanning** - Tests `get_available_models()`
- ✅ **GPU Detection** - Tests `detect_gpu_info()` and `get_optimal_settings()`
- ✅ **Script Generation** - Tests `gen_sh()` with various configurations
- ✅ **Caching Flags** - Ensures caching options appear in generated scripts
- ✅ **Dataset Creation** - Tests `create_dataset()` with image resizing
- ✅ **TOML Generation** - Tests `gen_toml()` configuration output

### Captioning Tests

- ✅ **Ollama Integration** - Tests successful and failed Ollama captioning
- ✅ **Florence-2 Fallbacks** - Tests the robust Florence-2 implementation
- ✅ **Model Loading Errors** - Tests graceful degradation when models fail to load
- ✅ **Generate Method Missing** - Tests the specific AttributeError fix
- ✅ **Image Loading Failures** - Tests handling of corrupted/missing images
- ✅ **Utility Functions** - Tests Ollama connection checking and image encoding

### Mocking Strategy

- **GPU Operations** - Mocked to work on CPU-only systems
- **Model Loading** - Mocked to avoid downloading large models
- **Network Requests** - Mocked for Ollama API calls
- **File Operations** - Uses temporary directories

## Adding New Tests

### Test File Naming

- `test_*.py` for test modules
- `Test*` for test classes
- `test_*` for test functions

### Writing Tests

```python
import pytest
import app

class TestNewFeature:
    """Test description."""
    
    def test_feature_success(self, fixture_name):
        """Test successful case."""
        result = app.feature_function("input")
        assert result == "expected"
    
    def test_feature_failure(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            app.feature_function("invalid_input")
```

### Using Fixtures

```python
def test_with_images(self, sample_images, temp_output_dir):
    """Test using synthetic images and temp directory."""
    # sample_images provides synthetic test images
    # temp_output_dir provides a clean temporary directory
    pass
```

## Test Environment

### Isolated Testing

- Tests run in isolation from the main application
- Uses temporary directories for file operations
- Mocks external dependencies (GPU, network, models)
- Doesn't require actual FLUX models to be downloaded

### CI/CD Compatibility

- Tests can run on CPU-only systems
- No GPU required (GPU operations are mocked)
- No network access required (external APIs are mocked)
- Fast execution (avoids model downloads)

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure you're in the right directory
cd /path/to/fluxgym
pytest
```

**Missing Dependencies**:
```bash
pip install pytest pytest-mock pillow
```

**GPU-Related Errors**:
- Tests should work without GPU (operations are mocked)
- If you see CUDA errors, check that mocking is working correctly

**File Permission Errors**:
- Ensure write permissions in the test directory
- Tests clean up temporary files automatically

### Debug Mode

```bash
# Run tests with debug output
pytest -v -s --tb=long

# Run single test with debugging
pytest -v -s tests/test_core_functionality.py::TestScriptGeneration::test_gen_sh_basic
```

## Contributing

When adding new features to FluxGym:

1. **Add corresponding tests** for new functionality
2. **Update fixtures** if new mock data is needed
3. **Maintain ethical standards** - use synthetic/open-source test data only
4. **Document test purpose** clearly in docstrings
5. **Ensure tests are deterministic** and don't rely on external state

---

**Note**: This test suite prioritizes ethical testing practices by using synthetic images and avoiding real demographic content. All test images are procedurally generated geometric patterns designed specifically for technical validation.
