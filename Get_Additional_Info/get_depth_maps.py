import os, shutil, argparse

from tqdm import tqdm
from PIL import Image
from pathlib import Path 
from transformers import pipeline

def flush_directory(dir_path):
    """
    Deletes all contents of a directory if it exists.
    The directory itself is not removed.
    """
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use", choices=["small", "large"])
    args = parser.parse_args()
    ds = args.dataset

    image_names = os.listdir(f"./{ds}/0_imgs")
    aux_save_dir = "Aux_small" if ds == "small" else "Aux_large"
    os.makedirs(aux_save_dir, exist_ok=True)

    pipe = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")

    for rot in [0, 90, 180, 270]:

        source_dir = f"./{ds}/{rot}_imgs"
        output_dir = f"./{aux_save_dir}/depth_maps/rot{rot}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        else: 
            flush_directory(output_dir)

        # Get the 0 degree (right-side up depth map)
        for image_name in tqdm(os.listdir(source_dir)):
            image_path = os.path.join(source_dir, image_name)
            image = Image.open(image_path).convert("RGB")

            result = pipe(image)
            depth = result["depth"]
            depth.save(os.path.join(output_dir, image_name))
