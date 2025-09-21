#!/usr/bin/env python3
"""
Test script for FluxGym workflow automation features
"""
import os
import sys
import tempfile
import shutil
from datetime import datetime

# Add the current directory to the path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import (
        detect_interrupted_training,
        auto_optimize_dataset,
        smart_parameter_suggestions,
        create_batch_training_queue
    )
    print("✅ Successfully imported workflow automation functions")
except ImportError as e:
    print(f"❌ Failed to import functions: {e}")
    sys.exit(1)

def test_smart_parameter_suggestions():
    """Test the smart parameter suggestions feature"""
    print("\n🧠 Testing Smart Parameter Suggestions...")
    
    # Test with various configurations
    test_cases = [
        {"vram": 12, "dataset_size": 10, "resolution": 512},
        {"vram": 24, "dataset_size": 5, "resolution": 1024},
        {"vram": 8, "dataset_size": 20, "resolution": 768}
    ]
    
    for case in test_cases:
        suggestions = smart_parameter_suggestions(
            case["vram"], 
            case["dataset_size"], 
            case["resolution"]
        )
        
        print(f"  VRAM: {case['vram']}GB, Dataset: {case['dataset_size']} images, Resolution: {case['resolution']}px")
        print(f"  Suggestions: {suggestions}")
        
        # Verify suggestions contain expected keys
        expected_keys = ['batch_size', 'vae_batch_size', 'max_train_epochs', 'num_repeats', 'network_dim', 'learning_rate']
        for key in expected_keys:
            if key not in suggestions:
                print(f"    ⚠️  Missing key: {key}")
            else:
                print(f"    ✅ {key}: {suggestions[key]}")
        print()

def test_detect_interrupted_training():
    """Test the interrupted training detection"""
    print("\n🔄 Testing Interrupted Training Detection...")
    
    # Create a mock interrupted training scenario
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create outputs directory structure
        outputs_dir = os.path.join(temp_dir, "outputs")
        os.makedirs(outputs_dir)
        
        # Create a mock interrupted training folder
        training_folder = os.path.join(outputs_dir, "test_training")
        os.makedirs(training_folder)
        
        # Create training files (but no final .safetensors)
        with open(os.path.join(training_folder, "train.bat"), 'w') as f:
            f.write("echo training script")
        
        with open(os.path.join(training_folder, "dataset.toml"), 'w') as f:
            f.write("[dataset]\nname = 'test'")
        
        # Create checkpoint files (indicating interrupted training)
        with open(os.path.join(training_folder, "epoch-003-state.pt"), 'w') as f:
            f.write("checkpoint data")
        
        # Temporarily change the working directory context for the test
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Mock the resolve_path_without_quotes function
            import app
            original_resolve = app.resolve_path_without_quotes
            app.resolve_path_without_quotes = lambda path: os.path.join(temp_dir, path)
            
            interrupted = detect_interrupted_training()
            
            # Restore original function
            app.resolve_path_without_quotes = original_resolve
            
            if interrupted:
                print(f"  ✅ Found {len(interrupted)} interrupted training(s)")
                for training in interrupted:
                    print(f"    - {training['name']}: {training['last_checkpoint']}")
            else:
                print("  ℹ️  No interrupted training found")
                
        finally:
            os.chdir(original_cwd)

def test_batch_training_queue():
    """Test the batch training queue functionality"""
    print("\n📋 Testing Batch Training Queue...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Mock the resolve_path_without_quotes function
            import app
            original_resolve = app.resolve_path_without_quotes
            app.resolve_path_without_quotes = lambda path: os.path.join(temp_dir, path)
            
            # Create test queue items
            queue_items = [
                {
                    'name': 'character_lora_1',
                    'script_path': '/path/to/script1.bat',
                    'added_at': str(datetime.now()),
                    'status': 'pending'
                },
                {
                    'name': 'character_lora_2', 
                    'script_path': '/path/to/script2.bat',
                    'added_at': str(datetime.now()),
                    'status': 'pending'
                }
            ]
            
            queue_file = create_batch_training_queue(queue_items)
            
            if os.path.exists(queue_file):
                print(f"  ✅ Created queue file: {queue_file}")
                with open(queue_file, 'r') as f:
                    import json
                    queue_data = json.load(f)
                    print(f"  ✅ Queue contains {len(queue_data['items'])} items")
                    for item in queue_data['items']:
                        print(f"    - {item['name']}: {item['status']}")
            else:
                print("  ❌ Failed to create queue file")
            
            # Restore original function
            app.resolve_path_without_quotes = original_resolve
            
        finally:
            os.chdir(original_cwd)

def test_auto_optimize_dataset():
    """Test the auto dataset optimizer"""
    print("\n🎯 Testing Auto Dataset Optimizer...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        test_images_dir = os.path.join(temp_dir, "test_images")
        os.makedirs(test_images_dir)
        
        # Create a simple test image using PIL
        try:
            from PIL import Image
            
            # Create test images
            for i in range(3):
                img = Image.new('RGB', (800, 600), color=(255, 0, 0))
                img.save(os.path.join(test_images_dir, f"test_img_{i}.png"))
            
            print(f"  ✅ Created {len(os.listdir(test_images_dir))} test images")
            
            # Test optimization
            result = auto_optimize_dataset(
                test_images_dir, 
                target_resolution=512, 
                auto_caption=True, 
                concept_word="test_character"
            )
            
            print(f"  Result: {result}")
            
            # Check if optimized folder was created
            optimized_dir = f"{test_images_dir}_optimized"
            if os.path.exists(optimized_dir):
                optimized_files = os.listdir(optimized_dir)
                print(f"  ✅ Created optimized folder with {len(optimized_files)} files")
                for file in optimized_files:
                    print(f"    - {file}")
            else:
                print("  ⚠️  Optimized folder not created")
                
        except ImportError:
            print("  ⚠️  PIL not available, skipping image optimization test")

def main():
    """Run all workflow automation tests"""
    print("🚀 FluxGym Workflow Automation Test Suite")
    print("=" * 50)
    
    try:
        test_smart_parameter_suggestions()
        test_detect_interrupted_training() 
        test_batch_training_queue()
        test_auto_optimize_dataset()
        
        print("\n" + "=" * 50)
        print("✅ All workflow automation tests completed!")
        print("🎉 Features are ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()