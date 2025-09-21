#!/usr/bin/env python3
"""
Debug optimal settings
"""

def get_optimal_settings(vram_gb):
    settings = {
        'vram_setting': '20G',
        'network_dim': 16,
        'resolution': 512,
        'workers': 2,
    }
    
    if vram_gb >= 24:
        settings.update({'vram_setting': '20G', 'network_dim': 32, 'resolution': 1024})
    elif vram_gb >= 16:
        settings.update({'vram_setting': '16G', 'network_dim': 16, 'resolution': 768})
    elif vram_gb >= 12:
        settings.update({'vram_setting': '12G', 'network_dim': 8, 'resolution': 512})
    
    return settings

for vram in [24, 16, 12, 8]:
    settings = get_optimal_settings(vram)
    print(f"{vram}GB: {settings}")