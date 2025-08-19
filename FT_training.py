import argparse
import os, random, torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info  
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from common import encode_image, get_img_mime, new_choice_mapping


lin_points = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000]
lin_points2 = [e+4000 for e in lin_points[1:]]
save_points = sorted(set(lin_points + lin_points2))

# Configuration
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training arguments
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
LOGGING_STEPS = 50
MAX_LENGTH = 2048


# Rotation setup
ROTATIONS = [0, 90, 180, 270]
CHOICE_LETTERS = ['A', 'B', 'C', 'D']

SYS_PROMPT = """You are an intelligent AI assistant that specializes in identifying rotation in images. You will be given an image and a multiple choice question. Each choice corresponds to the number of degrees the image has been rotated. A 90° rotation is a quarter-turn counter-clockwise; 270° is a quarter-turn clockwise. A 0° rotation indicates the image is right-side up; a 180° rotation indicates the image is up-side down."""


def build_prompt(choices_text: str) -> str:
    return f"""\
Identify whether the image has been rotated.

Response with a SINGLE LETTER, either A, B, C, or D, representing the correct rotation. You must select one of these choices even if you are uncertain. DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE.

The rotation of the image is:
{choices_text}

Answer: """


class RotationDataset(Dataset):
    def __init__(self, processor):
        self.processor = processor
        self.data = []
        for rot in ROTATIONS:
            dir1 = f"./Coco/{rot}_imgs"
            for img in os.listdir(dir1):
                self.data.append({"image_path": os.path.join(dir1, img), "rotation": rot})
        print(f"### Total {len(self.data)} images found in dataset ###")
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        rot_label = item["rotation"]
        b64 = encode_image(img_path)
        mapping, choices = new_choice_mapping()

        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:{get_img_mime(img_path)};base64,{b64}"},
                    {"type": "text", "text": build_prompt(choices)},
                ],
            },
            {"role": "assistant", "content": mapping[rot_label]},
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )


        labels = inputs["input_ids"].clone()
        prelim = self.processor.apply_chat_template(
            messages[:-1],  # drop assistant
            tokenize=False,
            add_generation_prompt=True
        )
        usr_only = self.processor(
            text=prelim,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        prefix_len = usr_only["input_ids"].shape[1]
        labels[0, :prefix_len] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels.squeeze(0)

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.squeeze(0)
        return inputs


def collate_fn(batch):
    max_len = max(x["input_ids"].size(0) for x in batch)
    pad_id  = processor.tokenizer.pad_token_id

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, x in enumerate(batch):
        L = x["input_ids"].size(0)
        input_ids[i, :L] = x["input_ids"]
        attention_mask[i, :L] = x["attention_mask"]
        labels[i, :L] = x["labels"]

    pixel_values = torch.stack([x["pixel_values"]    for x in batch], dim=0)  # (B, C, H, W)
    image_grid_thw = torch.stack([x["image_grid_thw"] for x in batch], dim=0)  # (B, 3)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"trainable params: {trainable} / {total} ({100*trainable/total:.2f}%)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, required=True)
    parser.add_argument("-fp", "--output_file_path", type=str, required=True)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_file_path

    seed = args.run
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        skip_modules=["visual"]
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",         
        trust_remote_code=True,
    )
    fixed_pixels = 224 * 224
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True,
        min_pixels=fixed_pixels,
        max_pixels=fixed_pixels,
    )
    model.to(device)
    model.config.use_cache = False
    # model.visual.requires_grad_(False)
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    print_trainable_parameters(model)

    train_ds = RotationDataset(processor)
    train_dl = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count()//2 or 0,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    images_seen = 0
    next_save_idx = 0 

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        step_count = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, batch in enumerate(pbar, 1):
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

                # Accumulate the actual loss (not scaled)
                total_loss += outputs.loss.item()
                step_count += 1
                images_seen += batch["input_ids"].size(0)

                                
                # Only step optimizer after accumulation
                if step % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Logging
                if step % LOGGING_STEPS == 0:
                    avg_loss = total_loss / step_count
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                while next_save_idx < len(save_points) and images_seen >= save_points[next_save_idx]:
                    sp = save_points[next_save_idx]
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"{sp}_images")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    processor.save_pretrained(ckpt_dir)
                    print(f"▶ Saved checkpoint at {sp} images to {ckpt_dir}")
                    next_save_idx += 1
                
            except Exception as e:
                print(f"Error in training step {step}: {e}")
                continue
        
        if step % GRADIENT_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuning complete, saved to {OUTPUT_DIR}")
