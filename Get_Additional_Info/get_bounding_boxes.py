import torch, os, json, base64, shutil, argparse
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-base"
DEVICE = "cuda"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)


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


def encode_image(image_path: str) -> str:
    """Base-64-encode an image so it can be embedded in the prompt."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_bounding_boxes(image_path: str, detected_objs: list[str]):
    image = Image.open(image_path).convert("RGB")
    inputs = PROCESSOR(images=image, text=detected_objs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(**inputs)

    results = PROCESSOR.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    result = results[0]
    width, height = image.size
    detected_boxes = {}
    for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
        x_min, y_min, x_max, y_max = box.tolist()
        norm_box = [
            round(x_min / width, 3),
            round(y_min / height, 3),
            round(x_max / width, 3),
            round(y_max / height, 3)
        ]
        cleaned_norm_box = []
        for coor in norm_box: 
            if coor < 0 and abs(coor - 0.0) <= 0.01: 
                cleaned_norm_box.append(0.0)
            elif coor > 1 and abs(coor-1.0) <= 0.01:
                cleaned_norm_box.append(1.0)
            else: 
                cleaned_norm_box.append(round(coor, 2))
        detected_boxes[label] = cleaned_norm_box
    return detected_boxes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use", choices=["small", "large"])
    args = parser.parse_args()
    ds = args.dataset

    image_names = os.listdir(f"./{ds}/0_imgs")
    aux_save_dir = "Aux_small" if ds == "small" else "Aux_large"
    os.makedirs(aux_save_dir, exist_ok=True)

    for rot in [0, 90, 180, 270]:

        with open(f"./{aux_save_dir}/subjects/rot{rot}/GPT4o.json", "r") as f:
            subject_dict = json.load(f)

        output_json = {}
        for image_name in tqdm(image_names):

            image_subjects = subject_dict[image_name]
            image_filepath = f"./{ds}/{rot}_imgs/" + image_name

            detected_boxes = get_bounding_boxes(image_filepath, image_subjects)

            output_json[image_name] = detected_boxes

        os.makedirs(f"./{aux_save_dir}/bounding_boxes/rot{rot}", exist_ok=True)
        with open(f"./{aux_save_dir}/bounding_boxes/rot{rot}/GPT4o.json", "w") as f:
            json.dump(output_json, f, indent=4)
