import os, json, argparse, shutil, base64, random

from tqdm import tqdm
from openai import OpenAI
from google import genai
from google.genai import types
from common import encode_image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential

from dotenv import load_dotenv
load_dotenv()


ROTATIONS = [0, 90, 180, 270]
CHOICE_LETTERS = ['A', 'B', 'C', 'D']
TEMPERATURE = 0.3

SYS_PROMPT = "You are an intelligent AI assistant that specializes in identifying rotation in images. You will be given an image and a multiple choice question. Each choice corresponds to the number of degrees the image has been rotated. A 90° rotation is a quarter-turn counter-clockwise; 270° is a quarter-turn clockwise. A 0° rotation indicates the image is right-side up; a 180° rotation indicates the image is up-side down."


def new_choice_mapping() -> tuple[dict[int, str], str]:
    """
    Return a fresh degree ➜ letter mapping and the formatted answer lines.
    Mapping: {90: 'A', 0: 'B', 180: 'C', 270: 'D'}
    prompt_lines:
        'A. 90
         B. 0
         C. 180
         D. 270
        '
    """
    shuffled = ROTATIONS.copy()
    random.shuffle(shuffled)

    mapping: dict[int, str] = {}
    prompt_lines: list[str] = []
    for idx, deg in enumerate(shuffled):
        letter = CHOICE_LETTERS[idx]
        mapping[deg] = letter
        prompt_lines.append(f"{letter}. {deg}")
    return mapping, "\n".join(prompt_lines)


def build_prompt(
        prompt_lines: str,
        cap_str: str | None,
        bb_str: str | None, 
        sg_str: str | None,
        include_depth_map: bool,
        include_seg_map: bool,
        include_img_grid: bool,
        use_cot: bool
        ) -> str:
    """Compose the final prompt shown to the vision model."""

    prompt = "Identify whether the image has been rotated."

    if cap_str or bb_str or sg_str or include_depth_map or include_seg_map:
        prompt += " In addition, you have been provided some extra information about this image below."
    
    if cap_str: 
        prompt += f"\n\nThe image is given the following caption:\n{cap_str}"

    if bb_str:
        prompt += f"\n\nBelow is the normalized bounding box of objects in the image. Each object is bounded by four floats [xmin, ymin, xmax, ymax] (each float has been normalized between 0 and 1).\n{bb_str}"

    if sg_str:
        prompt += f"\n\nBelow is a scene graph representing objects within the image and the relationship between them.\n{sg_str}"

    if include_depth_map:
        prompt += f"\n\nAttached is also an estimated depth map of the image. The brighter the pixel, the further it is."

    if include_seg_map:
        prompt += f"\n\nAttached is also an segmentation map of the image. Each object has been highlighted a different color."

    if include_img_grid:
        prompt += f"\n\n Attached is a grid showing the image rotated counter-clockwise in different orientations. The top-left image is the original image shown prior, the top-right image has been rotated 270° counter-clockwise, the bottom-right 180° counter-clockwise, and the bottom-left 270° counter-clockwise."

        if use_cot: 
            prompt += " Use this three step procedure: (1) Carefully examine all four images shown in the grid. (2) Identify the image you are most familiar with, or which image most resembles your training data, as an anchor point. (3) Starting from that image, algebraically determine the rotation of the original image. Using these other images as aid, what is the rotation of the original image? Let's think step-by-step"
        else: 
            prompt += " Using these other images as aid, what is the rotation of the original image? Let's think step-by-step"
        
        return prompt

    if use_cot: 
        prompt += "\n\nWhat is the rotation of this image? Let's think step-by-step."
    else: 
        prompt += "\n\nResponse with a SINGLE LETTER, either A, B, C, or D, representing the correct rotation. You must select one of these choices even if you are uncertain. DO NOT INCLUDE ANYTHING ELSE IN YOUR RESPONSE."
        prompt += f"\n\nThe rotation of the image is:\n{prompt_lines}\n\nAnswer: "

    return prompt


# ────────────────────────────────────────────────────────────────────────────────
# Model Calls
# ────────────────────────────────────────────────────────────────────────────────
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def vllm_answer(
        model_name: str, 
        img_path: str, 
        client: OpenAI, 
        prompt: str,
        dm_filepath: str | None,
        sm_filepath : str | None,
        grid_filepath: str | None) -> tuple[str, int, int]:

    msg_content = []

    img_type = "image/jpeg" if img_path.lower().endswith("jpg") else "image/png"
    encoded_img = encode_image(img_path)
    msg_content.append({"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{encoded_img}"}})

    if dm_filepath:
        dm_type = "image/jpeg" if dm_filepath.lower().endswith("jpg") else "image/png"
        encoded_dm = encode_image(dm_filepath)
        msg_content.append({"type": "image_url", "image_url": {"url": f"data:{dm_type};base64,{encoded_dm}"}})
        # print("DEPTH MAP SHOWN")

    if sm_filepath:
        sm_type = "image/jpeg" if sm_filepath.lower().endswith("jpg") else "image/png"
        encoded_sm = encode_image(sm_filepath)
        msg_content.append({"type": "image_url", "image_url": {"url": f"data:{sm_type};base64,{encoded_sm}"}})
        # print("SEG MAP SHOWN")

    if grid_filepath:
        grid_type = "image/jpeg" if grid_filepath.lower().endswith("jpg") else "image/png"
        encoded_grid = encode_image(grid_filepath)
        msg_content.append({"type": "image_url", "image_url": {"url": f"data:{grid_type};base64,{encoded_grid}"}})
        # print("GRID SHOWN")

    msg_content.append({"type": "text", "text": prompt})
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": msg_content}
    ]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=TEMPERATURE
    )

    response_text = chat_response.choices[0].message.content
    prompt_tokens = chat_response.usage.prompt_tokens
    completion_tokens = chat_response.usage.completion_tokens
    return response_text, prompt_tokens, completion_tokens


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_answer(
        model_name: str, 
        img_path: str, 
        client: OpenAI, 
        prompt: str,
        dm_filepath: str | None,
        sm_filepath : str | None,
        grid_filepath: str | None) -> tuple[str, int, int]:


    msg_content = []
    img_type = "image/jpeg" if img_path.lower().endswith("jpg") else "image/png"
    encoded_img = encode_image(img_path)
    msg_content.append({"type": "input_image", "image_url": f"data:{img_type};base64,{encoded_img}"})

    
    if dm_filepath:
        dm_type = "image/jpeg" if dm_filepath.lower().endswith("jpg") else "image/png"
        encoded_dm = encode_image(dm_filepath)
        msg_content.append({"type": "input_image", "image_url": f"data:{dm_type};base64,{encoded_dm}"})
        # print("DEPTH MAP SHOWN")

    if sm_filepath:
        sm_type = "image/jpeg" if sm_filepath.lower().endswith("jpg") else "image/png"
        encoded_sm = encode_image(sm_filepath)
        msg_content.append({"type": "input_image", "image_url": f"data:{sm_type};base64,{encoded_sm}"})
        # print("SEG MAP SHOWN")

    if grid_filepath:
        grid_type = "image/jpeg" if grid_filepath.lower().endswith("jpg") else "image/png"
        encoded_grid = encode_image(grid_filepath)
        msg_content.append({"type": "input_image", "image_url": f"data:{grid_type};base64,{encoded_grid}"})
        # print("GRID SHOWN")

    msg_content.append({"type": "input_text", "text": prompt})
    if model_name == "o3":
        chat_response = client.responses.create(
            model=model_name, 
            instructions = SYS_PROMPT,
            input = [{"role": "user", "content": msg_content}],
        )
    else: 
        chat_response = client.responses.create(
            model=model_name, 
            instructions = SYS_PROMPT,
            input = [{"role": "user", "content": msg_content}],
            temperature=TEMPERATURE
        )
    response_text = chat_response.output_text
    prompt_tokens = chat_response.usage.input_tokens
    completion_tokens = chat_response.usage.output_tokens
    return response_text, prompt_tokens, completion_tokens


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gemini_answer(
        model_name: str, 
        img_path: str, 
        client: genai.Client,
        prompt: str,
        dm_filepath: str | None,
        sm_filepath: str | None,
        grid_filepath: str | None) -> tuple[str, int, int]:
    

    contents = []

    img_type = "image/jpeg" if img_path.lower().endswith("jpg") else "image/png"
    with open(img_path, "rb") as f:
        image_bytes = f.read()
    contents.append(types.Part.from_bytes(data=image_bytes, mime_type=img_type))

    if dm_filepath:
        dm_type = "image/jpeg" if dm_filepath.lower().endswith("jpg") else "image/png"
        with open(dm_filepath, "rb") as f:
            dm_bytes = f.read()
        contents.append(types.Part.from_bytes(data=dm_bytes, mime_type=dm_type))

    if sm_filepath:
        sm_type = "image/jpeg" if sm_filepath.lower().endswith("jpg") else "image/png"
        with open(sm_filepath, "rb") as f:
            sm_bytes = f.read()
        contents.append(types.Part.from_bytes(data=sm_bytes, mime_type=sm_type))

    if grid_filepath:
        grid_type = "image/jpeg" if grid_filepath.lower().endswith("jpg") else "image/png"
        with open(grid_filepath, "rb") as f:
            grid_bytes = f.read()
        contents.append(types.Part.from_bytes(data=grid_bytes, mime_type=grid_type))
    
    contents.append(f"{SYS_PROMPT}\n\n{prompt}")

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=8192, 
            temperature=TEMPERATURE
            )
    )
    response_text = response.text
    if not response_text:
        print(response)
        return "", response.usage_metadata.prompt_token_count, 0
    prompt_tokens = response.usage_metadata.prompt_token_count
    completion_tokens = response.usage_metadata.candidates_token_count
    # print(f"##### {response.text} {prompt_tokens} {completion_tokens}")
    return response_text, prompt_tokens, completion_tokens


# ────────────────────────────────────────────────────────────────────────────────
# Image processing worker (runs in threads)
# ────────────────────────────────────────────────────────────────────────────────

def process_image(
        img_name: str,
        img_path: str, 
        rot_gt: int, 
        client, 
        model_name: str,
        cap_dict: dict | None,
        bb_dict: dict | None,
        sg_dict: dict | None,
        use_dm: bool,
        use_sm: bool,
        use_grid: bool,
        use_cot: bool):
    """End-to-end processing of a single image. Returns (img_name, answer, mapping, p_tk, c_tk)."""

    mapping, prompt_lines = new_choice_mapping()
    use_gemini = "gemini" in model_name.lower()
    
    cap_str = str(cap_dict[img_name]) if cap_dict else None

    # Find corresponding bb_json
    if bb_dict:
        bb_str = str(bb_dict[img_name])
    else: 
        bb_str = None

    if sg_dict:
        sg_str = str(sg_dict[img_name]) 
    else: 
        sg_str = None

    if "small" in img_path: 
        aux_save_dir = "Aux_small"
    else: 
        aux_save_dir = "Aux_large"


    dm_filepath = f"./{aux_save_dir}/depth_maps/rot{rot_gt}/{img_name}" if use_dm else None
    sm_filepath = f"./{aux_save_dir}/seg_maps/rot{rot_gt}/{img_name}" if use_sm else None
    grid_filepath = f"./{aux_save_dir}/img_grid/rot{rot_gt}/{img_name}" if use_grid else None

    prompt = build_prompt(prompt_lines, cap_str, bb_str, sg_str, use_dm, use_sm, use_grid,use_cot)
    # print(prompt)

    if use_gemini:
        answer_func = gemini_answer
    elif "gpt" in model_name or model_name == "o3": 
        answer_func = openai_answer
    else: 
        answer_func = vllm_answer

    ans, p_tk, c_tk = answer_func(model_name, img_path, client, prompt, dm_filepath, sm_filepath, grid_filepath)
    # print(ans)

    return img_name, ans, mapping, prompt, p_tk, c_tk


# ────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-nick", "--model_nickname", type=str, required=True, help="Model nickname")
    parser.add_argument("-workers", "--max_workers", type=int, default=3, help="Thread pool size")
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use", choices=["small", "large"])
    parser.add_argument("--run", type=int, required=True, help="Number of current run, also used for seeding")

    parser.add_argument("-bb", "--bounding_box", action="store_true", help="Use bounding boxes")
    parser.add_argument("-cap", "--caption", action="store_true", help="Use captions")
    parser.add_argument("-dm", "--depth_map", action="store_true", help="Use depth maps")
    parser.add_argument("-sm", "--seg_map", action="store_true", help="Use segmentation maps")
    parser.add_argument("-sg", "--scene_graph", action="store_true", help="Use scene graphs")
    parser.add_argument("-gr", "--image_grid", action="store_true", help="Use image grid")   
    parser.add_argument("-cot", "--use_cot", action="store_true", help="Use CoT prompting.")
    args = parser.parse_args()

    # CLI flags
    model_nickname = args.model_nickname
    max_workers = args.max_workers
    dataset = args.dataset
    use_cap = args.caption
    use_bb = args.bounding_box 
    use_sg = args.scene_graph
    use_dm = args.depth_map
    use_sm = args.seg_map
    use_gr = args.image_grid
    use_cot = args.use_cot
    random.seed(args.run)


    # API configuration info
    nick_to_name_port = {
        "Llama11": ("meta-llama/Llama-3.2-11B-Vision-Instruct", 7473),
        "Qwen7": ("Qwen/Qwen2.5-VL-7B-Instruct", 7472),
        "Qwen32": ("Qwen/Qwen2.5-VL-32B-Instruct", 7473),
        "GPT4o": ("gpt-4o", 0),
        "Gemini2": ("gemini-2.0-flash", 0),
        "GPT41": ("gpt-4.1", 0),
        "o3": ("o3", 0),
        "Gemini25": ("gemini-2.5-flash", 0),
        "Gemini25pro": ("gemini-2.5-pro", 0)
    }

    model_name, port = nick_to_name_port[model_nickname]
    # Initialize API clients
    if "gemini" in model_name:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    elif "gpt" in model_name.lower() or "o3" in model_name.lower():
        base_url = "https://api.openai.com/v1"
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)  
    else:
        base_url = f"http://localhost:{port}/v1"
        client = OpenAI(api_key="EMPTY", base_url=base_url)

    for rot_deg in ROTATIONS:
        if dataset == "large":
            data_dir = f"./large/{rot_deg}_imgs"
            aux_save_dir = "Aux_large"
        else:
            assert dataset == "small"
            data_dir = f"./small/{rot_deg}_imgs"
            aux_save_dir = "Aux_small"

        img_names = os.listdir(data_dir)
        print(f"Currently on rotation {rot_deg}, {data_dir}")


        # Load captions
        cap_dict = None
        if use_cap: 
            with open(f"./{aux_save_dir}/captions/rot{rot_deg}/GPT4o.json", "r") as f: 
                cap_dict = json.load(f)

        # Load bounding boxes
        bb_dict = None
        if use_bb:
            with open(f"./{aux_save_dir}/bounding_boxes/rot{rot_deg}/GPT4o.json", "r") as f:
                bb_dict = json.load(f)

        # Load scene graphs
        sg_dict = None
        if use_sg:
            with open(f"./{aux_save_dir}/scene_graphs/rot{rot_deg}/GPT4o_narrow.json", "r") as f:
                sg_dict = json.load(f)
        
        L_out = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_image,
                    img_name,
                    os.path.join(data_dir, img_name),
                    rot_deg,
                    client,
                    model_name,
                    cap_dict,
                    bb_dict,
                    sg_dict,
                    use_dm,
                    use_sm,
                    use_gr,
                    use_cot
                ): img_name
                for img_name in img_names
            }

            for fut in tqdm(as_completed(futures), total=len(futures)):
                L_out.append(fut.result())

        # Write to json 
        output_json = []
        total_p_tk, total_c_tk = 0, 0
        for (img_name, ans, mapping, prompt, p_tk, c_tk) in L_out: 
            total_p_tk += p_tk
            total_c_tk += c_tk
            output_json.append({
                "image_name": img_name,
                "model_response": ans,
                "mapping": mapping
            })
            # print(f"Model response: {ans}")

        if dataset == "large": 
            output_dir = f"./Results_large/{model_nickname}"
        else:
            assert dataset == "small"
            output_dir = f"./Results_small/{model_nickname}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        save_filename = f"{output_dir}/rot{rot_deg}_run{args.run}"
        save_filename += "_cap" if use_cap else ""
        save_filename += "_bb" if use_bb else ""
        save_filename += "_sg" if use_sg else ""
        save_filename += "_dm" if use_dm else ""
        save_filename += "_sm" if use_sm else ""
        save_filename += "_gr" if use_gr else ""
        save_filename += "_cot" if use_cot else ""
        save_filename += ".json"
        with open(save_filename, "w", encoding="utf-8") as f: 
            json.dump(output_json, f, indent=4, ensure_ascii=False) 

        print(f"{rot_deg} degrees used {total_p_tk=} {total_c_tk=}")