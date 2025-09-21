#!/usr/bin/env python3
"""
Generate synthetic test images for FluxGym test suite.
Uses geometric patterns and abstract shapes to avoid demographic content.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random
import math


def create_synthetic_image(width=512, height=512, pattern="geometric", seed=None):
    """Create a synthetic test image with geometric patterns."""
    if seed is not None:
        random.seed(seed)
    
    # Create base image
    img = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    if pattern == "geometric":
        # Draw geometric shapes
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                 (255, 255, 100), (255, 100, 255), (100, 255, 255)]
        
        for _ in range(random.randint(5, 15)):
            color = random.choice(colors)
            shape_type = random.choice(['rectangle', 'circle', 'triangle'])
            
            x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
            
            if shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            elif shape_type == 'circle':
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            elif shape_type == 'triangle':
                x3, y3 = random.randint(x1, x2), random.randint(y1, y2)
                draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=color, outline=(0, 0, 0))
    
    elif pattern == "gradient":
        # Create gradient pattern
        for y in range(height):
            for x in range(width):
                r = int(255 * (x / width))
                g = int(255 * (y / height))
                b = int(255 * ((x + y) / (width + height)))
                img.putpixel((x, y), (r, g, b))
    
    elif pattern == "noise":
        # Create colored noise pattern
        for y in range(height):
            for x in range(width):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                img.putpixel((x, y), (r, g, b))
    
    elif pattern == "mandala":
        # Create mandala-like pattern
        center_x, center_y = width // 2, height // 2
        max_radius = min(center_x, center_y) - 10
        
        for angle in range(0, 360, 10):
            for radius in range(20, max_radius, 20):
                x = int(center_x + radius * math.cos(math.radians(angle)))
                y = int(center_y + radius * math.sin(math.radians(angle)))
                color = (
                    int(128 + 127 * math.sin(angle * 0.1)),
                    int(128 + 127 * math.cos(angle * 0.1)),
                    int(128 + 127 * math.sin(radius * 0.1))
                )
                if 0 <= x < width and 0 <= y < height:
                    draw.ellipse([x-5, y-5, x+5, y+5], fill=color)
    
    # Add text label to identify the pattern
    try:
        # Try to use a basic font, fallback to default if not available
        font = ImageFont.load_default()
        draw.text((10, 10), f"{pattern}_{width}x{height}", fill=(0, 0, 0), font=font)
    except:
        draw.text((10, 10), f"{pattern}_{width}x{height}", fill=(0, 0, 0))
    
    return img


def generate_test_dataset(output_dir, count=10, patterns=None):
    """Generate a set of test images for FluxGym testing."""
    if patterns is None:
        patterns = ["geometric", "gradient", "noise", "mandala"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    images_created = []
    
    for i in range(count):
        pattern = patterns[i % len(patterns)]
        
        # Vary image sizes for testing
        sizes = [(512, 512), (768, 768), (1024, 1024), (640, 480)]
        width, height = sizes[i % len(sizes)]
        
        # Create image
        img = create_synthetic_image(width, height, pattern, seed=i)
        
        # Save image
        filename = f"test_image_{i:03d}_{pattern}_{width}x{height}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, "PNG")
        
        # Create corresponding caption file
        caption_file = os.path.join(output_dir, f"test_image_{i:03d}_{pattern}_{width}x{height}.txt")
        caption = f"a {pattern} pattern image with colorful shapes and designs"
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption)
        
        images_created.append({
            'image': filepath,
            'caption': caption_file,
            'pattern': pattern,
            'size': (width, height)
        })
        
        print(f"Created: {filename}")
    
    return images_created


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic test images for FluxGym")
    parser.add_argument("--output", "-o", default="./test_images", 
                       help="Output directory for test images")
    parser.add_argument("--count", "-c", type=int, default=10,
                       help="Number of images to generate")
    parser.add_argument("--patterns", nargs="+", 
                       choices=["geometric", "gradient", "noise", "mandala"],
                       default=["geometric", "gradient", "noise", "mandala"],
                       help="Patterns to generate")
    
    args = parser.parse_args()
    
    print(f"Generating {args.count} test images in {args.output}")
    images = generate_test_dataset(args.output, args.count, args.patterns)
    print(f"Generated {len(images)} test images successfully!")