import cv2, json, shutil, os, argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from sam2.sam2_image_predictor import SAM2ImagePredictor

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


def get_and_save_segmap(image_name, image_path, save_path, predictor, bb_dict):
    img_pil = Image.open(image_path).convert("RGB")
    img_cv  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W   = img_cv.shape[:2]

    img_bbs = bb_dict[image_name]
    subjects, boxes = [], []
    for sub, box in img_bbs.items():
        subjects.append(sub)
        xb1, yb1, xb2, yb2 = box
        x1, y1 = int(xb1 * W), int(yb1 * H)
        x2, y2 = int(xb2 * W), int(yb2 * H)
        boxes.append([x1, y1, x2, y2])


    predictor.set_image(img_cv) 

    masks = []
    for box in boxes:
        mask_batch, _, _ = predictor.predict(
            box=np.array(box),
            multimask_output=False
        )
        masks.append(mask_batch[0])  

    # Build a color mask & overlay
    color_mask = np.zeros_like(img_cv, dtype=np.uint8)
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 256, size=(len(subjects), 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color_mask[mask.astype(bool)] = palette[i]

    overlay = cv2.addWeighted(img_cv, 0.2, color_mask, 0.8, 0)
    cv2.imwrite(save_path, overlay)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use", choices=["small", "large"])
    args = parser.parse_args()
    ds = args.dataset

    image_names = os.listdir(f"./{ds}/0_imgs")
    aux_save_dir = "Aux_small" if ds == "small" else "Aux_large"
    os.makedirs(aux_save_dir, exist_ok=True)


    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2.1-hiera-base-plus",
        trust_remote_code=True 
    )

    for rot in [0, 90, 180, 270]:

        with open(f"./{aux_save_dir}/bounding_boxes/rot{rot}/GPT4o.json", "r") as f: 
            bb_dict = json.load(f)

        source_dir = f"./{ds}/{rot}_imgs"
        output_dir = f"./{aux_save_dir}/seg_maps/rot{rot}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        else: 
            flush_directory(output_dir)
    
        for image_name in tqdm(os.listdir(source_dir)):
            image_path = os.path.join(source_dir, image_name)
            output_path = os.path.join(output_dir, image_name)
            get_and_save_segmap(image_name, image_path, output_path, predictor, bb_dict)

