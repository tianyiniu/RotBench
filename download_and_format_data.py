from datasets import load_dataset
import os
from PIL import Image

def download_dataset_to_directories(dataset_name, base_dir="RotBench-Data"):
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Load datasets
    print("Loading large dataset...")
    large_dataset = load_dataset(dataset_name, split="large")
    
    print("Loading small dataset...")
    small_dataset = load_dataset(dataset_name, split="small")
    
    # Create subdirectories
    large_dir = os.path.join(base_dir, "RotBench_large")
    small_dir = os.path.join(base_dir, "RotBench_small")
    
    os.makedirs(large_dir, exist_ok=True)
    os.makedirs(small_dir, exist_ok=True)
    
    # Download large dataset images
    print(f"Downloading {len(large_dataset)} images to {large_dir}...")
    for i, sample in enumerate(large_dataset):
        image = sample["image"]
        image_name = sample["image_name"]
        
        # Save image with original filename
        image_path = os.path.join(large_dir, image_name)
        image.save(image_path)
        
        if (i + 1) % 50 == 0:  # Progress indicator
            print(f"Downloaded {i + 1}/{len(large_dataset)} large images...")
    
    # Download small dataset images
    print(f"Downloading {len(small_dataset)} images to {small_dir}...")
    for i, sample in enumerate(small_dataset):
        image = sample["image"]
        image_name = sample["image_name"]
        
        # Save image with original filename
        image_path = os.path.join(small_dir, image_name)
        image.save(image_path)
        
        if (i + 1) % 10 == 0:  # Progress indicator
            print(f"Downloaded {i + 1}/{len(small_dataset)} small images...")
    
    print(f"Download complete")
    print(f"Large dataset: {len(large_dataset)} images in {large_dir}")
    print(f"Small dataset: {len(small_dataset)} images in {small_dir}")
    
    return large_dir, small_dir

if __name__ == "__main__":
    large_dir, small_dir = download_dataset_to_directories("tianyin/RotBench")
    
    # Verify the structure
    print("\nDirectory structure created:")
    print("RotBench-Data/")
    print(f"├── RotBench_large/ ({len(os.listdir(large_dir))} files)")
    print(f"└── RotBench_small/ ({len(os.listdir(small_dir))} files)")