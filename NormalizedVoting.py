import json, argparse, os, random
from collections import Counter

def get_predicted_angle(mapping, model_response): 
    """
        Mapping: angle: str -> letter choiec: str
        model_response: single letter: str
    """
    inverse_mapping = {v:k for k,v in mapping.items()}
    return int(inverse_mapping[model_response])


def load_results(model_nick): 
    """Results dictionary: 
        res[ground_truth_rotation][image_name] = predicted_angle
    """
    all_results = {}
    for rot in [0, 90, 180, 270]:
        with open(f"./Results_small/{model_nick}/rot{rot}_run0.json", "r", encoding="utf-8") as f: 
            results = json.load(f)
            rotation_dict = {}
            for res in results: 
                img_name = res["image_name"]
                model_response = res["model_response"]
                if len(model_response) != 1 or model_response not in ["A", "B", "C", "D"]:
                    print(model_response)
                    model_response = input("What did the model predict?: ")
                prediction = get_predicted_angle(res["mapping"], model_response)
                rotation_dict[img_name] = prediction
        all_results[rot] = rotation_dict
    return all_results

def get_angle_order(starting_angle: int):
    cur_angle = starting_angle
    order = [cur_angle]
    for _ in range(3): 
        cur_angle = (cur_angle+90) % 360
        order.append(cur_angle)
    return order


def conduct_vote(img_name: str, gt_rot: int, all_results): 
    angle_order = get_angle_order(gt_rot)
    predictions = [all_results[a][img_name] for a in angle_order]
    normalized_angles = []
    for i, predicted_angle in enumerate(predictions):
        normalized_angles.append((predicted_angle - i*90) % 360)

    count = Counter(normalized_angles)
    max_freq = max(count.values())
    most_frequent_items = [item for item, freq in count.items() if freq == max_freq]
    majority_vote_angle = random.choice(most_frequent_items)
    return majority_vote_angle


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-nick", "--model_nickname", type=str, required=True, help="Model nickname")
    args = parser.parse_args()

    # API configuration info
    nick_to_name_port = {
        "Llama11": ("meta-llama/Llama-3.2-11B-Vision-Instruct", 7473),
        "Qwen7": ("Qwen/Qwen2.5-VL-7B-Instruct", 7472),
        "Qwen32": ("Qwen/Qwen2.5-VL-32B-Instruct", 7473),
        "GPT4o": ("gpt-4o", 0),
        "Gemini2": ("gemini-2.0-flash", 0),
        "GPT41": ("gpt-4.1", 0),
        "GPT5": ("gpt-5", 0),
        "o3": ("o3", 0),
        "Gemini25": ("gemini-2.5-flash", 0),
        "Gemini25pro": ("gemini-2.5-pro", 0)
    }
    img_names = os.listdir("./small/0_imgs")

    model_nick = args.model_nickname
    model_name, _ = nick_to_name_port[model_nick]

    all_results = load_results(model_nick)

    for rot in [0, 90, 180, 270]: 
        rot_accs = []
        for img_name in img_names: 
            voted_rotation = conduct_vote(img_name, rot, all_results)
            acc = voted_rotation == rot
            rot_accs.append(acc)
        average_accs = sum(rot_accs)/len(rot_accs)
        print(f"Rotation {rot}: {round(average_accs, 2)}")

