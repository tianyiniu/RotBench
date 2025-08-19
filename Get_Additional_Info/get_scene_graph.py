""" 
Generates scene graphs
"""
import os, re, ast, json, base64, argparse
from openai import OpenAI 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv 
load_dotenv() 

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

def get_narrow_scene_graph_prompt(objects: list[str]):
    return f"""\
Task: Given the image and key objects, generate a scene graph for this image. Represent each relationship as a three-element tuple with ('subject_id', 'predicate', 'object_id'). Extract a set of words describing the location, orientation, directions and spatial or positional relations between key objects in the image. Your answer should be a list of values that are in format of (object1, relation, object2). The relation MUST be one of [left, right, above, below, facing left, facing right, front, behind]. You are to interpret the image literally. If you see a sky below a mountain, your scene graph must reflect that. Format your response as a Python list of tuples, surrounded by a markdown fence. Example formatting: 

```python
[("object1", "predicate1", "object2"), "object2", "predicate2", "object3"), ...]
```

Key objects in the image: {objects}
"""

def get_broad_scene_graph_prompt(objects: list[str]):
    return f"""\
Task: Given the image and key objects, generate a scene graph for this image. Represent each relationship as a three-element tuple with ('subject_id', 'predicate', 'object_id'). Format your response as a Python list of tuples, surrounded by a markdown fence. Example formatting: 

```python
[("object1", "predicate1", "object2"), "object2", "predicate2", "object3"), ...]
```

Key objects in the image: {objects}
"""

MODEL_NAME = "gpt-4o"
CLIENT = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))


def parse_llm_output(llm_response: str) -> list[tuple[str, str, str]]:
    """
    Args:
        output: Raw string from the LLM, e.g.:
            ```python
            [("obj1","on","obj2"),("obj2","next_to","obj3")]
            ```
    Returns:
        A list of (subject_id, predicate, object_id) tuples.
    Raises:
        ValueError if parsing fails or the structure is invalid.
    """
    fence_re = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
    m = fence_re.search(llm_response)
    code = m.group(1) if m else llm_response

    try:
        parsed = ast.literal_eval(code.strip())
    except Exception as e:
        print(llm_response)
        raise ValueError(f"Could not literal_eval scene graph: {e}")

    if not isinstance(parsed, list):
        print(llm_response)
        raise ValueError(f"Expected a list, got {type(parsed).__name__}")
    for idx, item in enumerate(parsed):
        if not (isinstance(item, tuple) and len(item) == 3):
            print(llm_response)
            raise ValueError(f"Item {idx} is not a 3-tuple: {item!r}")
        if not all(isinstance(elem, str) for elem in item):
            print(llm_response)
            raise ValueError(f"Tuple {idx} contains non-str elements: {item!r}")
    return parsed


def encode_image(image_path):
    """Encodes image at image_path using base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_scene_graph(image_path: str, image_name:str, objects: list[str], use_broad: bool):
    """
    Returns:
        A list of relationships, e.g., [["object1", "relation", "object2"], ["object1", "attribute", "value"]],
        or None if an error occurs.
    """

    base64_image = encode_image(image_path)
    img_type = "image/jpeg" if image_path.lower().endswith("jpg") else "image/png"
    prompt_func = get_broad_scene_graph_prompt if use_broad else get_narrow_scene_graph_prompt
    prompt = prompt_func(objects)

    msgs = [
        {
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{base64_image}"}},
                {"type": "text", "text": prompt}
        ]}
    ]

    chat_response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            temperature=0.1
    )

    response_text = chat_response.choices[0].message.content
    parsed_response = parse_llm_output(response_text)
    prompt_tokens = chat_response.usage.prompt_tokens
    completion_tokens = chat_response.usage.completion_tokens
    return parsed_response, prompt_tokens, completion_tokens, image_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use", choices=["small", "large"])
    args = parser.parse_args()
    ds = args.dataset

    image_names = os.listdir(f"./{ds}/0_imgs")
    aux_save_dir = "Aux_small" if ds == "small" else "Aux_large"
    os.makedirs(aux_save_dir, exist_ok=True)

    use_broad = False

    total_p_tks, total_c_tks = 0, 0 
    for rot in [0, 90, 180, 270]:
        L_out = []
        zip_img_paths = [(name, f"./{ds}/{rot}_imgs/{name}") for name in image_names]

        with open(f"./{aux_save_dir}/subjects/rot{rot}/GPT4o.json", "r") as f: 
            objects_json = json.load(f)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(generate_scene_graph, fp, name, objects_json[name], False): (name, fp) for (name, fp) in zip_img_paths}

            for fut in tqdm(as_completed(futures), total=len(futures)):
                L_out.append(fut.result())

        output_json = {}
        for parsed_response, p_tks, c_tks, image_name in L_out: 
            output_json[image_name] = parsed_response
            total_p_tks += p_tks
            total_c_tks += c_tks

        os.makedirs(f"./{aux_save_dir}/scene_graphs/rot{rot}", exist_ok=True)
        save_fp = f"./{aux_save_dir}/scene_graphs/rot{rot}/GPT4o_broad.json" if use_broad else f"./{aux_save_dir}/scene_graphs/rot{rot}/GPT4o_narrow.json"

        with open(save_fp, "w") as f: 
            json.dump(output_json, f, indent=4)
        
    print(f"{total_p_tks=}")
    print(f"{total_c_tks=}")