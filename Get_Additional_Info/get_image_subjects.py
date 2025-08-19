"""
Calls GPT 4o to identify the primary subjects in each images. Loads images in "images.json" and saves a list of subjects (list[str]) into each json object.
"""

import os, ast, re, json, base64, argparse
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  


def encode_image(image_path: str) -> str:
    """Base-64-encode an image so it can be embedded in the prompt."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_list_from_markdown(markdown_str):
    """Extract content from ```python['object 1', 'object 2', ...]``` and converts it into a python list object."""

    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, markdown_str, re.DOTALL)
    
    if not match:
        print(markdown_str)
        raise ValueError("Input does not contain a valid Python markdown code block.")
    list_str = match.group(1)
    try:
        return ast.literal_eval(list_str)
    except (ValueError, SyntaxError):
        print(markdown_str)
        raise ValueError("Extracted content is not a valid Python list literal.")
    

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(4))
def get_openai_img_subjects(image_path, client, model_name):
    """Gets a list of subjects in the image.""" 
    img_type = "image/jpeg" if image_path.lower().endswith("jpg") else "image/png"
    encoded_img = encode_image(image_path)
    response = client.chat.completions.create(
        model=model_name,
        messages= [{
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{encoded_img}"}},
                {"type": "text", "text": "Return a list of objects in this image. The list will later be passed to a bounding box model to extract bounding boxes for each detected object. Format your task as a python list, surrounded with a python markdown fence. For example: ```python\n['fedora', 'woman in green dress', 'man in red suit', ...]\n``` Each object should have a distinct name. ENSURE YOUR RESPONSE FOLLOWS THE FORMATTING REQUIREMENTS!"}
                ]
        }]
    )
    text = response.choices[0].message.content.strip()
    prompt_tks = response.usage.prompt_tokens
    completion_tks = response.usage.completion_tokens
    return text, prompt_tks, completion_tks


def process_image(image_path, client, model_name): 
    process_func = get_openai_img_subjects
    response_text, p_tks, c_tks = process_func(image_path, client, model_name)
    extracted_obj = extract_list_from_markdown(response_text)
    image_name = image_path.split("/")[-1]
    return extracted_obj, p_tks, c_tks, image_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use", choices=["small", "large"])
    args = parser.parse_args()
    ds = args.dataset

    image_names = os.listdir(f"./{ds}/0_imgs")
    aux_save_dir = "Aux_small" if ds == "small" else "Aux_large"
    os.makedirs(aux_save_dir, exist_ok=True)

    model_nickname = "GPT4o"
    model_name = "gpt-4o"
    num_workers = 3

    base_url = "https://api.openai.com/v1" if "gpt" in model_name.lower() else f"http://localhost:{port}/v1"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)



    total_prompt_tks, total_completion_tks = 0, 0
    for rot in [0, 90, 180, 270]:
        L_out = []
        img_paths = [f"./{ds}/{rot}_imgs/{name}" for name in image_names]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_image,img_filepath, client, model_name): img_filepath for img_filepath in img_paths}

            for fut in tqdm(as_completed(futures), total=len(futures)):
                L_out.append(fut.result())


        output_json = {}
        for (extracted_obj, p_tks, c_tks, image_name) in L_out:
            output_json[image_name] = extracted_obj
            total_prompt_tks += p_tks
            total_completion_tks += c_tks
        
        os.makedirs(f"./{aux_save_dir}/subjects/rot{rot}", exist_ok=True)
        with open(f"./{aux_save_dir}/subjects/rot{rot}/{model_nickname}.json", "w") as f: 
            json.dump(output_json, f, indent=4)

    print(f"{total_prompt_tks=}, {total_completion_tks=}")