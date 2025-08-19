import os
import json, argparse, random, base64
from tqdm import tqdm
from PIL import Image
from common import encode_image, get_img_mime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info


# Config
lin_points = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000]
lin_points2 = [e+4000 for e in lin_points[1:]]
save_points = sorted(set(lin_points + lin_points2))


ROTATIONS = [0, 90, 180, 270]
CHOICE_LETTERS = ['A', 'B', 'C', 'D']
SYS_PROMPT = """You are an intelligent AI assistant that specializes in identifying rotation in images. You will be given an image and a multiple choice question. Each choice corresponds to the number of degrees the image has been rotated. A 90° rotation is a quarter-turn counter-clockwise; 270° is a quarter-turn clockwise. A 0° rotation indicates the image is rightside-up; a 180° rotation indicates the image is up-side down."""

def new_choice_mapping():
    shuffled = ROTATIONS.copy()
    random.shuffle(shuffled)
    mapping, lines = {}, []
    for i, deg in enumerate(shuffled):
        letter = CHOICE_LETTERS[i]
        mapping[letter] = deg
        lines.append(f"{letter}. {deg}")
    return mapping, "\n".join(lines)


def build_prompt(choices_text: str) -> str:
    return f"""\
Identify whether the image has been rotated.
        
Response with a SINGLE LETTER, either A, B, C, or D, representing the correct rotation. You must select one of these choices even if you are uncertain. DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE.

The rotation of the image is:
{choices_text}

Answer: """


# Inference
def infer_rotation(model, processor, img_path, device):
    mapping, choices = new_choice_mapping()
    b64 = encode_image(img_path)
    mime = get_img_mime(img_path)
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": [
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

    # Expect single letter
    letter = text_out[0] if text_out and text_out[0] in mapping else None
    pred_deg = mapping.get(letter, -1)
    return text_out, pred_deg

def process_image(img_name, img_path, gt, model, processor, device):
    raw, pred = infer_rotation(model, processor, img_path, device)
    return img_name, pred, gt, raw


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("-r", "--run", type=int, required=True)
    p.add_argument("-w", "--max_workers", type=int, default=5)
    p.add_argument("-fp", "--ft_model_path", type=str, required=True)
    args = p.parse_args()
    random.seed(args.run)
    torch.manual_seed(args.run)
    torch.cuda.manual_seed_all(args.run)

    import numpy as np
    np.random.seed(args.run)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    all_results_dict = {}
    for save in tqdm(save_points):
        assert save not in all_results_dict
        all_results_dict[save] = {0: None, 90: None, 180: None, 270: None}
        ft_model_path = args.ft_model_path
        processor = AutoProcessor.from_pretrained(ft_model_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, ft_model_path, torch_dtype=torch.bfloat16)
        model.eval().to(device)


        for rot in ROTATIONS:
            accs = []
            for _ in range(1):
                all_results, total_correct, total_images = [], 0, 0
                d = f"./large/{rot}_imgs"
                imgs = os.listdir(d)
                print(f"Eval rotation={rot}° on {len(imgs)} images")

                with ThreadPoolExecutor(max_workers=args.max_workers) as exe:
                    futures = {
                        exe.submit(process_image, img, os.path.join(d, img), rot, model, processor, device): img
                        for img in imgs
                    }
                    for fut in as_completed(futures):
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

                acc = round(total_correct / total_images if total_images else 0.0, 2)
                accs.append(acc)
            all_results_dict[save][rot] = round(sum(accs)/len(accs), 2)

    with open(f"./ft_results_run{args.run}.json", "w") as f: 
        json.dump(all_results_dict, f, indent=4)
