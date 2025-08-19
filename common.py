import base64
import random

# Rotation setup
ROTATIONS = [0, 90, 180, 270]
CHOICE_LETTERS = ['A', 'B', 'C', 'D']


def new_choice_mapping() -> tuple[dict[int, str], str]:
    """Shuffle ROTATIONS into A–D and build the prompt lines."""
    shuffled = ROTATIONS.copy()
    random.shuffle(shuffled)
    mapping = {}
    lines = []
    for idx, deg in enumerate(shuffled):
        letter = CHOICE_LETTERS[idx]
        mapping[deg] = letter
        lines.append(f"{letter}. {deg}")
    return mapping, "\n".join(lines)

def encode_image(image_path: str) -> str:
    """Base-64-encode an image so it can be embedded in the prompt."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_img_mime(path: str) -> str:
    return "image/jpeg" if path.lower().endswith(".jpg") else "image/png"
