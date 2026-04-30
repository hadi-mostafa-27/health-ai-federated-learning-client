import os
from pathlib import Path
from PIL import Image, ImageDraw
import random

def generate_sample_dataset(base_dir: str | Path, num_samples: int = 40):
    base_dir = Path(base_dir)
    normal_dir = base_dir / "NORMAL"
    pneumonia_dir = base_dir / "PNEUMONIA"
    
    normal_dir.mkdir(parents=True, exist_ok=True)
    pneumonia_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate some simple mock chest x-rays
    # Normal: fewer white blobs
    # Pneumonia: more white/cloudy blobs
    for i in range(num_samples // 2):
        _create_mock_xray(normal_dir / f"normal_{i}.png", is_pneumonia=False)
        _create_mock_xray(pneumonia_dir / f"pneumonia_{i}.png", is_pneumonia=True)

    return str(base_dir)

def _create_mock_xray(path: Path, is_pneumonia: bool):
    # Dark grey background
    img = Image.new('L', (224, 224), color=40)
    draw = ImageDraw.Draw(img)
    
    # Draw spine/ribs mock lines
    for y in range(40, 200, 20):
        draw.line([(60, y), (160, y)], fill=80, width=4)
        
    draw.line([(110, 20), (110, 200)], fill=100, width=8) # spine
    
    # Draw some "lungs"
    draw.ellipse([40, 40, 100, 180], outline=60, width=2)
    draw.ellipse([120, 40, 180, 180], outline=60, width=2)
    
    num_blobs = random.randint(5, 15) if is_pneumonia else random.randint(0, 3)
    for _ in range(num_blobs):
        x = random.randint(40, 180)
        y = random.randint(60, 160)
        r = random.randint(10, 30)
        intensity = random.randint(120, 200)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=intensity)
        
    img.save(path)

if __name__ == "__main__":
    generate_sample_dataset("sample_data/chest_xray")
