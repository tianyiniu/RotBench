"""Given a directory of images, create parallel directories that rotates it 0, 90, 180, 270 degrees."""

import os
from PIL import Image

def rotate_images(input_dir, output_dir, degrees):
    """Rotates all .jpg/.png images in input_dir by the given degrees and saves them in output_dir."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with Image.open(input_path) as img:
                rotated_img = img.rotate(degrees)
                rotated_img.save(output_path)


if __name__ == "__main__":
    os.makedirs("./Coco", exist_ok=True)
    for deg in [0, 90, 180, 270]:
        input_dir = f"./MS_COCO"
        rotate_images(input_dir, f"./Coco/{deg}_imgs", deg)