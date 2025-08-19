import os, json, argparse, base64, argparse
import PIL.Image
import numpy as np

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential,
)
from dotenv import load_dotenv
load_dotenv()

CLIENT = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
MODEL_NICKNAME = "GPT4o"
MAX_WORKERS = 3


def logprobs_to_perplexity(logprobs):
    """Input a list of logprobs, output a single value for perplexity"""
    return np.exp(-np.mean(logprobs))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_caption(img_path, img_name, client):
    img_type = "image/jpeg" if img_path.endswith("jpg") else "image/png"
    prompt = f"Generate a detailed caption for this image. Do not include any preceeding text before the caption."
    encoded_img = encode_image(img_path)
    msgs = [
        {
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{encoded_img}"}},
                {"type": "text", "text": prompt}
        ]}
    ]
    chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=msgs,
            temperature=0,
            logprobs=True,
            top_logprobs=1
    )
    response_text = chat_response.choices[0].message.content
    prompt_tokens = chat_response.usage.prompt_tokens
    completion_tokens = chat_response.usage.completion_tokens

    logprobs = [tk.logprob for tk in chat_response.choices[0].logprobs.content]
    ppl = logprobs_to_perplexity(logprobs)
    return response_text, prompt_tokens, completion_tokens, logprobs, ppl, img_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use", choices=["small", "large"])
    args = parser.parse_args()
    ds = args.dataset

    total_prompt_tk, total_completion_tk = 0, 0
    image_names = os.listdir(f"./{ds}/0_imgs")
    aux_save_dir = "Aux_small" if ds == "small" else "Aux_large"
    os.makedirs(aux_save_dir, exist_ok=True)

    for rot in [0, 90, 180, 270]:
        data_dir = f"./{ds}/{rot}_imgs"

        L_out = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(get_caption, os.path.join(data_dir, image_name), image_name, CLIENT): image_name for image_name in image_names}

            for future in tqdm(as_completed(future_to_item), total=len(future_to_item)):
                result = future.result()
                if result:
                    L_out.append(result)

        ppl_data = {}
        caption_data = {}
        for (ans, prompt_tk, completion_tk, logprobs, ppl, image_name) in L_out:

            caption_data[image_name] = ans
            ppl_data[image_name] = logprobs

            total_completion_tk += completion_tk
            total_prompt_tk += prompt_tk

        os.makedirs(f"./{aux_save_dir}/captions/rot{rot}", exist_ok=True)
        os.makedirs(f"./{aux_save_dir}/logits/rot{rot}", exist_ok=True)

        with open(f"./{aux_save_dir}/captions/rot{rot}/GPT4o.json", "w") as f: 
            json.dump(caption_data, f, indent=4)
        with open(f"./{aux_save_dir}/logits/rot{rot}/GPT4o.json", "w") as f: 
            json.dump(ppl_data, f)

    print(f"Total prompt tokens: {total_prompt_tk}; total completion tokens: {total_completion_tk}")

