import json, os, argparse, re
from tqdm import tqdm
import numpy as np 
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=f"http://localhost:7471/v1")


FALLBACK_SYS_PROMPT = """
You are a helpful AI assistant that specializes in a homework problem that involves idenfiyting the rotation of an image. More specifically, each student is given a rotated image. The student must identify whether the image has been rotated 0, 90, 180, or 270 degrees counter-clockwise. Your job is to map student responses (free-text) to their identified angle of rotation (0/90/180/270). Given the a few examples and the student's response, you are output the angle that the student selected. RESPONSE ONLY WITH A SINGLE NUMBER, either 0/90/180/270.

A few notes: 
    - Some student may hint their answer using key phrases, such as 'the answer is ...', 'the rotation is ...', 'the correct rotation is ...'
    - Likewise, some students use specific formatting to indicate their answer, such as '*** 90 ***" or "box [90]". 
    - Some students answer in CLOCKWISE instead of COUNTER-CLOCKWISE rotations. Keep a look out for that and ensure you translate any clockwise rotations into counter clockwise rotations. The following are some alternative phrasing for each angle: 
        - 0: right-side up, upright, no rotation
        -- 90: 90 degrees counter-clockwise, 270 degrees clockwise, rotated 90 degrees
        -- 180: up-side down, rotated 180 degrees, 
        -- 270: 270 degrees counter-clockwise, 90 degrees clockwise, rotated 270 degrees
"""

FALLBACK_USER_PROMPT = """
Here are a few examples of the task: 

Sample student response 1: Step-by-step reasoning:\n1. Clouds and bright sky appear at the bottom of the frame, while darker ground, trees, and rock cliff appear at the top.\n2. Under normal orientation, sky should be above the ground; seeing it below means the image is upside-down.\n3. An upside-down image corresponds to a 180 rotation.\n\nAnswer: 180
Sample selected angle 1: 180

Sample student response 2: The image is upside-down; it has been rotated 180.
Sample selected angle 2: 180

Sample student response 3: *   **Step 1: Analyze the image content and identify key orientation cues.** The image is a high-angle, top-down shot of a narrow alley. The most reliable indicators of orientation are the people within the scene. We can see two individuals. One person, wearing a red shirt, is in the lower right portion of the frame. Another person's legs and feet are visible at the top of the frame.\n\n*   **Step 2: Determine the natural orientation based on these cues.** People stand upright due to gravity. In a top-down photograph like this, we are looking straight down at them. The person in the red shirt is walking, and their body is aligned vertically within the frame, which is consistent with a person standing or walking on the ground as seen from above. The same applies to the legs visible at the top.\n\n*   **Step 3: Evaluate the given rotation options.**\n    *   **0 (upright):** In this orientation, the scene appears as it would to an observer looking straight down. The people are standing on the ground below. This matches the visual evidence.\n    *   **90 (rotated counter-clockwise):** If the image were rotated 90, the people would appear to be lying on their sides, defying gravity. This is an unnatural orientation.\n    *   **180 (upside-down):** If the image were rotated 180, the people would appear upside-down relative to their current position. For example, the head of the person in red would be pointing towards the top of the frame. This is not the case.\n    *   **270 (rotated clockwise):** Similar to the 90 rotation, this would make the people appear to be lying on their sides.\n\n*   **Step 4: Conclude the correct rotation.** The current orientation of the image is the most logical and natural one, depicting a scene as viewed from directly above. Therefore, the image is not rotated.\n\n**What is the rotation of this image?**\n*   0\n*   90\n*   180\n*   270\n\nThe correct answer is **0**.
Sample selected angle 3: 0
"""


def compute_and_print_confusion_matrix(y_true, y_pred, labels=[0, 90, 180, 270], normalize=False):
    """
    Computes and prints a 4x4 confusion matrix for classification over 4 rotation classes.

    Args:
        y_true (list[int]): Ground truth labels (0, 90, 180, 270).
        y_pred (list[int]): Predicted labels (same as above).
        labels (list[int]): The class labels, must match the possible values in y_true/y_pred.
        normalize (bool): If True, normalize rows to show percentages.
    """
    label_to_index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((4, 4), dtype=int)

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


def compute_mean_std(values): 
    arr = np.array(values, dtype=np.float64)
    return round(np.mean(arr), 2), round(np.std(arr), 2)


def qwen_answer_match(model_answer: str, mapping_dict: dict) -> int:
    """Fallback: ask Qwen to map free-text → angle (int), using the same prompt."""
    inverse_mapping = {letter: rot_deg for rot_deg,letter in mapping_dict.items()}
    chat_response = CLIENT.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[
            {"role": "system", "content": FALLBACK_SYS_PROMPT},
            {"role": "user", "content": FALLBACK_USER_PROMPT + f"\n\nStudent response: {model_answer}\n\nStudent selected angle: "}
            ],
        temperature=0.3,
        extra_body={"chat_template_kwargs":{"enable_thinking": False}}
    )
    response_text = chat_response.choices[0].message.content.strip()
    predicted_angle = response_text.strip()
    try:
        assert int(predicted_angle) in [0, 90, 180, 270]
    except AssertionError:
        print(f"ASSERTION ERROR: {predicted_angle=}, {model_answer=}")
        predicted_angle = "0"
    return int(predicted_angle)


def main(old_res):
    res = old_res.copy()
    mapping = res["mapping"]
    inverse_mapping = {v: k for k, v in mapping.items()} # letter to angle

    ans = res["model_response"]
    if not ans: 
        print("ERROR - model did not provide a response")
        res["predicted_angle"] = 0
        return res

    if len(ans) == 1 and ans[0].upper() in ['A', 'B', 'C', 'D']:
        extracted_letter = ans
        predicted_angle = int(inverse_mapping[extracted_letter])
    else:
        predicted_angle = qwen_answer_match(ans, mapping)

    res["predicted_angle"] = predicted_angle
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nick", "--model_nickname", required=True, type=str)
    parser.add_argument("-ds", "--dataset", required=True, type=str, choices=["large", "small"])
    parser.add_argument(
        '--runs',
        nargs='+',
        type=int,         
        required=True,    
        help='list of runs, e.g. --runs 0 1'
    )
    parser.add_argument("-bb", "--bounding_box", action="store_true", help="Use bounding boxes")
    parser.add_argument("-cap", "--caption", action="store_true", help="Use captions")
    parser.add_argument("-dm", "--depth_map", action="store_true", help="Use depth maps")
    parser.add_argument("-sm", "--seg_map", action="store_true", help="Use segmentation maps")
    parser.add_argument("-sg", "--scene_graph", action="store_true", help="Use scene graphs")
    parser.add_argument("-gr", "--img_grid", action="store_true", help="Use image grid")

    parser.add_argument("-cot", "--use_cot", action="store_true", help="Use CoT prompting.")

    args = parser.parse_args()
    runs = args.runs
    model_nickname = args.model_nickname
    dataset = args.dataset
    
    all_preds, all_gts = [], [] # For calculating confusion matrix
    for rot in [0, 90, 180, 270]:

        acc_across_runs = []
        soft_acc_across_runs = []

        for run in runs:
            if dataset == "large": 
                results_fp = f"./Results_large/{model_nickname}/rot{rot}_run{run}"       
            else: 
                assert dataset == "small"
                results_fp = f"./Results_small/{model_nickname}/rot{rot}_run{run}"
            os.makedirs(results_fp, exist_ok=True)

            results_fp += "_cap" if args.caption else ""
            results_fp += "_bb" if args.bounding_box else ""
            results_fp += "_sg" if args.scene_graph else ""
            results_fp += "_dm" if args.depth_map else ""
            results_fp += "_sm" if args.seg_map else ""
            results_fp += "_gr" if args.img_grid else ""
            results_fp += "_cot" if args.use_cot else "" 
            results_fp += ".json"

            with open(results_fp, "r", encoding="utf-8") as f: 
                results = json.load(f)

            processed_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(main, res): res for res in results
                }
            for fut in tqdm(as_completed(futures), total=len(futures)):
                processed_results.append(fut.result())


            accs, soft_accs = [], []
            for res in processed_results:
                predicted_angle = res["predicted_angle"]
                all_preds.append(predicted_angle)
                all_gts.append(rot)
                acc = int(predicted_angle == rot)
                accs.append(acc)
                if rot in [0, 180]:
                    soft_acc = int(predicted_angle in [0, 180])
                else: 
                    soft_acc = int(predicted_angle in [90, 270])
                soft_accs.append(soft_acc)

                # Write back to file
                res["is_correct"] = bool(acc)
                res["is_correct_soft"] = bool(soft_acc)

            run_acc = round(sum(accs)/len(accs), 2)
            acc_across_runs.append(run_acc)
            run_soft_acc = round(sum(soft_accs)/len(soft_accs), 2)
            soft_acc_across_runs.append(run_soft_acc)


            assert len(results) == len(processed_results)
            with open(results_fp, "w", encoding="utf-8") as f: 
                json.dump(processed_results, f, indent=4, ensure_ascii=False)


        acc_mean, acc_std = compute_mean_std(acc_across_runs)
        soft_acc_mean, soft_acc_std = compute_mean_std(soft_acc_across_runs)
        print(f"ROT {rot} - Acc across runs: {acc_mean} ± {acc_std} | soft {soft_acc_mean} ± {soft_acc_std}")

    compute_and_print_confusion_matrix(all_gts, all_preds)
