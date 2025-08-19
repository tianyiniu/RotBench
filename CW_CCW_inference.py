import os, json, argparse, shutil, base64, random, re
import numpy as np 

from tqdm import tqdm
from openai import OpenAI
from common import encode_image, get_img_mime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential

from dotenv import load_dotenv
load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model Calls
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_answer(
        model_name: str, 
        img_path: str, 
        client: OpenAI) -> tuple[str, int, int]:

    sys_prompt = "You are an intelligent AI assistant that specializes in identifying rotation in images. You will be given an image that has been rotated 90 degrees. Your task is to identify whether the image has been rotated 90 degrees clockwise or counter-clockwise. Suppose you are given a right-side up portrait of a man. If the image has been rotated 90 degrees clockwise, the man's feet would be left of the man's head. If the rotation is counter-clockwise, and the man's head would be left of his feet. "

    user_prompt = "Your task is to identify whether the image has been rotated 90 clockwise or counter-clockwise. Examine the image closely and identify the rotation. Repond with 'cw' for clockwise and 'ccw' for counter-clockwise. Let's think step-by-step."


    msg_content = []

    encoded_img = encode_image(img_path)
    img_type = get_img_mime(img_path)
    msg_content.append({"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{encoded_img}"}})
    msg_content.append({"type": "text", "text": user_prompt})
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": msg_content}
    ]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=0.3,
        max_tokens=1024
    )

    response_text = chat_response.choices[0].message.content
    prompt_tokens = chat_response.usage.prompt_tokens
    completion_tokens = chat_response.usage.completion_tokens
    return response_text, prompt_tokens, completion_tokens


def process_image(img_name: str, img_path: str, client, model_name: str):
    """End-to-end processing of a single image. Returns (img_name, answer, mapping, p_tk, c_tk)."""
    ans, p_tk, c_tk = openai_answer(model_name, img_path, client)
    return img_name, ans, p_tk, c_tk


# Main entry point
if __name__ == "__main__":

    nick, max_workers = "GPT4o", 10

    nick_to_name_port = {
        "Qwen7": ("Qwen/Qwen2.5-VL-7B-Instruct", 7472),
        "GPT4o": ("gpt-4o", 0),
    }

    model_name, port = nick_to_name_port[nick]
    # Initialize API clients
    base_url = "https://api.openai.com/v1" if "gpt" in model_name.lower() else f"http://localhost:{port}/v1"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)


    for rot_deg in [90, 270]:
        total_p_tk, total_c_tk = 0, 0
        data_dir = f"./large/{rot_deg}_imgs"
        img_names = os.listdir(data_dir)
        print(f"Currently on rotation {rot_deg}, {data_dir}")

        
        L_out = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_image,
                    img_name,
                    os.path.join(data_dir, img_name),
                    client,
                    model_name,
                ): img_name
                for img_name in img_names
            }

            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    L_out.append(fut.result())
                except Exception as e:
                    print(f"Error processing {futures[fut]}: {e}")

        # Write to json 
        output_json = []
        for (img_name, ans, p_tk, c_tk) in L_out: 
            total_p_tk += p_tk
            total_c_tk += c_tk
            output_json.append({
                "image_name": img_name,
                "model_response": ans,
            })
        
        save_filename = f"./CW_CCW/{nick}/rot{rot_deg}.json"
        with open(save_filename, "w") as f: 
            json.dump(output_json, f, indent=4)

        print(f"{rot_deg} degrees used {total_p_tk=} {total_c_tk=}")
