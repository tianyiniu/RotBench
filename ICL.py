import os, math
from random import shuffle

import json, argparse, random, base64
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info

ROTATIONS = [0, 90, 180, 270]
CHOICE_LETTERS = ['A', 'B', 'C', 'D']
SYS_PROMPT = """You are an intelligent AI assistant that specializes in identifying rotation in images. You will be given an image and a multiple choice question. Each choice corresponds to the number of degrees the image has been rotated. A 90° rotation is a quarter-turn counter-clockwise; 270° is a quarter-turn clockwise. A 0° rotation indicates the image is right-side up; a 180° rotation indicates the image is up-side down."""


ONE_IMG_NAMES = ['149f7ae687.jpg', 'e298fa33df.jpg', '8087a6bee7.jpg', '6e5ecee526.jpg', 'df28cb4ea7.jpg', '27a00d0461.jpg', '78ee80415b.jpg', '38ae229755.jpg', '849480c30d.jpg', '255821e8dc.jpg']
TWO_IMG_NAMES = ['aa4543f91d.jpg', '31f7f4ab8f.jpg', '266d98da16.jpg', '82eebc2489.jpg', 'dbf447b049.jpg', '585a585006.jpg', '4576ea2bcb.jpg', '2377a92081.jpg', '8a5769619f.jpg', 'ac44ea8a69.jpg']
assert len(ONE_IMG_NAMES) == len(TWO_IMG_NAMES) == 10

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_img_mime(path: str) -> str:
    return "image/jpeg" if path.lower().endswith(".jpg") else "image/png"


def new_choice_mapping():
    shuffled = ROTATIONS.copy()
    random.shuffle(shuffled)
    mapping, lines = {}, []
    for i, deg in enumerate(shuffled):
        letter = CHOICE_LETTERS[i]
        mapping[letter] = deg
        lines.append(f"{letter}. {deg}")
    return mapping, "\n".join(lines)


def mean_and_std(numbers):
    n = len(numbers)
    if n == 0:
        raise ValueError("mean_and_std requires at least one data point")
    mu = sum(numbers) / n
    var = sum((x - mu) ** 2 for x in numbers) / n
    sigma = math.sqrt(var)
    return mu, sigma


def build_prompt(choices_text: str) -> str:
    return f"""\
Identify whether the image has been rotated.
        
Response with a SINGLE LETTER, either A, B, C, or D, representing the correct rotation. You must select one of these choices even if you are uncertain. DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE.

The rotation of the image is:
{choices_text}

Answer: """


def load_icl_prompt(num_icl_examples):
    assert num_icl_examples <= 10
    counter = 0
    temp_msgs = []
    for file_list, img_dir in [(ONE_IMG_NAMES, "OOD_one"), (TWO_IMG_NAMES, "OOD_two")]:
        filenames = file_list[:num_icl_examples]
        for img_name in filenames: 
            for rot in [0, 90, 180, 270]:
                img_path = f"./Data/{img_dir}_rot{rot}_imgs/{img_name}"
                img_type = get_img_mime(img_name)
                encoded_img = encode_image(img_path)
                temp_msgs.append(
                    (
                        {"type": "text", "text": f"\nBelow is an example of a image rotated {rot} degrees:\n"}, 
                        {"type": "image", "image": f"data:{img_type};base64,{encoded_img}"}
                    )
                )
                counter += 1
    shuffle(temp_msgs)
    msgs = []
    for text_msg, img_msg in temp_msgs:
        msgs.append(text_msg)
        msgs.append(img_msg)
    print(f"Loaded {counter} images into prompt")
    return msgs


def infer_rotation(icl_msgs, model, processor, img_path, device):
    mapping, choices = new_choice_mapping()
    b64 = encode_image(img_path)
    mime = get_img_mime(img_path)
    messages = [
        {"role": "system", "content": SYS_PROMPT}, 
        {
            "role": "user",
            "content": icl_msgs + [
                {"type": "image", "image": f"data:{mime};base64,{b64}"},
                {"type": "text",  "text": build_prompt(choices)},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)

    out_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=50,
    )

    prompt_len = inputs.input_ids.shape[1]
    gen = out_ids[:, prompt_len:]
    text_out = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    letter = text_out[0] if text_out and text_out[0] in mapping else None
    pred_deg = mapping.get(letter, -1)
    return text_out, pred_deg


def process_image(img_name, img_path, icl_msgs, gt, model, processor, device):
    raw, pred = infer_rotation(icl_msgs, model, processor, img_path, device)
    return img_name, pred, gt, raw


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num_examples", type=int, required=True)
    p.add_argument("-w", "--max_workers", type=int, default=3)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    fixed_pixels = 224 * 224
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast=True,
        min_pixels=fixed_pixels,
        max_pixels=fixed_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval().to(device)
    print("Model loaded.")

    icl_msgs = load_icl_prompt(args.num_examples)

    results = {rot: [] for rot in ROTATIONS}
    for _ in range(3):
        for rot in ROTATIONS:
            all_results, total_correct, total_images = [], 0, 0
            d = f"./large/{rot}_imgs"
            imgs = os.listdir(d)
            print(f"Eval rotation={rot}° on {len(imgs)} images")

            with ThreadPoolExecutor(max_workers=args.max_workers) as exe:
                futures = {
                    exe.submit(process_image, img, os.path.join(d, img), icl_msgs, rot, model, processor, device): img
                    for img in imgs
                }
                for fut in tqdm(as_completed(futures), total=len(futures)):
                    img_name, pred, gt, raw = fut.result()
                    correct = (pred == gt)
                    total_correct += int(correct)
                    total_images  += 1
                    all_results.append({
                        "image": img_name,
                        "gt":      gt,
                        "pred":    pred,
                        "out":     raw,
                        "correct": correct
                    })

            acc = total_correct / total_images if total_images else 0.0
            print(f"Accuracy: {acc:.2f}%  ({total_correct}/{total_images})")
            results[rot].append(acc)
    for rot, accs in results.items():
        m, s = mean_and_std(accs)
        print(f"Rot {rot}: Mean: {m:.2f}, Std dev: {s:.2f}") 
