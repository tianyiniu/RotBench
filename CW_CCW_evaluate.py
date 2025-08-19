import os, json, argparse, shutil, random, re
import numpy as np 

from tqdm import tqdm
from openai import OpenAI
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=f"http://localhost:7471/v1")

SYS_PROMPT = "You are an intelligent AI assistant that specializes mapping student responses to a fixed set of answers. You MUST reply with either 'CW' or 'CCW'."


def compute_and_print_confusion_matrix(y_true, y_pred, labels=[-1, 90, 270], normalize=False):
    """
    Args:
        y_true & y_pred (list[int])
        normalize (bool): If True, normalize rows to show percentages.
    """
    label_to_index = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)
    cm = np.zeros((num_labels, num_labels), dtype=int)

    for true, pred in zip(y_true, y_pred):
        i = label_to_index[true]
        j = label_to_index[pred]
        cm[i][j] += 1

    # Normalize if requested
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    # Print matrix
    print(" " * 9 + "Predicted")
    print(" " * 8 + " ".join(f"{l:>6}" for l in labels))
    print("        +" + "------" * len(labels))

    for i, label in enumerate(labels):
        row_label = f"True {label:>3}"
        row_values = [
            f"{cm[i][j]:6.2f}" if normalize else f"{int(cm[i][j]):6d}"
            for j in range(len(labels))
        ]
        print(f"{row_label} |" + "".join(row_values))


def compute_mean(values): 
    arr = np.array(values, dtype=np.float64)
    return round(np.mean(arr), 3)


def qwen_answer_match(model_answer: str) -> int:
    chat_response = CLIENT.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"You are an assistant that maps student responses to the available multiple-choice letters. The student's task is to identify whether a given image has been rotated 90 degrees counterclockwise (CCW) or clockwise (CW). Based on the student's response, determine if the student answered CW or CCW. You MUST reply with either 'CW' or 'CCW'\n\nStudent response: {model_answer}\n\nStudent selected orientation: "}
            ],
        temperature=0.3, 
        extra_body={"chat_template_kwargs":{"enable_thinking": False}}
    )
    response_text = chat_response.choices[0].message.content.strip()
    assert response_text.upper() in ["CCW", "CW"]
    p_tks = chat_response.usage.prompt_tokens
    c_tks = chat_response.usage.completion_tokens
    return model_answer, response_text, p_tks, c_tks


def template_match(image_name, model_response: str, is_cot: bool): 
    if is_cot:
        model_response = model_response.split(".")[-2]
        print(model_response)
        if "counter-clockwise" in model_response: 
            angle =  90
        elif "clockwise" in model_response: 
            angle =  270
        else: 
            angle = -1
    else: 
        if model_response == "cw": 
            angle = 270
        elif model_response == "ccw": 
            angle = 90
        else: 
            angle = -1
    return image_name, model_response, angle, 0, 0


# Main entry point
if __name__ == "__main__":

    nick, max_workers = "GPT5", 3

    all_gts, all_preds = [], []
    accs = []
    for rot in [90, 270]:
        results_fp = f"./CW_CCW/{nick}/rot{rot}.json"

        with open(results_fp, "r") as f: 
            results = json.load(f)
            model_answers = [o["model_response"] for o in results]

        L_out = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    qwen_answer_match,
                    res["model_response"],
                ): res for res in results
            }

            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    L_out.append(fut.result())
                except Exception as e:
                    print(f"Error processing {futures[fut]}: {e}") 


        for (model_answer, model_response, p_tks, c_tks) in L_out:

            gt_rot = rot
            angle = 90 if model_response == "CCW" else 270
            all_preds.append(angle)
            all_gts.append(rot)
            acc = int(angle == gt_rot)
            accs.append(acc)

    run_acc = round(sum(accs)/len(accs), 3)
    print(f"Acc across runs: {run_acc}")
    compute_and_print_confusion_matrix(all_gts, all_preds)