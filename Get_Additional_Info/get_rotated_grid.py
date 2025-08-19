import os, shutil, argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Path to a scalable TrueType/OpenType font on your system:
# - Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# - macOS: "/Library/Fonts/Arial.ttf"
# - Windows: "C:\\Windows\\Fonts\\arial.ttf"
DEFAULT_TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def flush_directory(dir_path):
    """
    Deletes all contents of a directory if it exists.
    The directory itself is not removed.
    """
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def compose_grid(
    img_path,
    font_path=None,
    font_size=80,
    padding=50,
    spacing=50,
    border_width=4,
    outer_padding=60
):
    """
    Creates a 2x2 grid of rotated versions of a single image with captions.

    Layout:
        [0°]   [270°]
        [90°]  [180°]

    :param img_path:        File path to the original image
    :param font_path:       Optional path to a .ttf/.otf font file
    :param font_size:       Font size for captions
    :param padding:         Space between image and caption
    :param spacing:         Space between grid cells
    :param border_width:    Width of border around each image
    :param outer_padding:   Padding around the entire grid
    :return:                PIL.Image object of the final grid
    """
    # Load base image
    base_img = Image.open(img_path)
    img_w, img_h = base_img.size

    # Generate rotated versions (CCW)
    rotations = [0, 90, 180, 270]
    rotated = [base_img.rotate(a, expand=False) for a in rotations]

    # Load a scalable font, fall back to default bitmap if that fails
    ttf = font_path or DEFAULT_TTF
    try:
        font = ImageFont.truetype(ttf, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure caption height
    try:
        bbox = font.getbbox("Sample")
        caption_h = bbox[3] - bbox[1]
    except AttributeError:
        try:
            _, caption_h = font.getsize("Sample")
        except AttributeError:
            caption_h = font_size

    # Calculate canvas size
    cell_h = img_h + padding + caption_h
    total_w = 2 * img_w + spacing + 2 * outer_padding
    total_h = 2 * cell_h + spacing + 2 * outer_padding

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    captions = [
        "Original Image",
        "Rotated 90°",
        "Rotated 180°",
        "Rotated 270°",
    ]
    # Place in [0°, 270°, 90°, 180°] order: TL, TR, BL, BR
    order     = [0, 3, 1, 2]
    positions = [
        (outer_padding, outer_padding),
        (outer_padding + img_w + spacing, outer_padding),
        (outer_padding, outer_padding + cell_h + spacing),
        (outer_padding + img_w + spacing, outer_padding + cell_h + spacing),
    ]

    for idx, (x, y) in zip(order, positions):
        img = rotated[idx]
        cap = captions[idx]

        # Border
        rect = [
            x - border_width,
            y - border_width,
            x + img_w + border_width - 1,
            y + img_h + border_width - 1,
        ]
        draw.rectangle(rect, outline="black", width=border_width)

        # Paste image
        canvas.paste(img, (x, y))

        # Centered caption
        try:
            tb = draw.textbbox((0, 0), cap, font=font)
            tw = tb[2] - tb[0]
        except AttributeError:
            try:
                tw, _ = draw.textsize(cap, font=font)
            except AttributeError:
                tw = len(cap) * font_size // 2

        tx = x + (img_w - tw) // 2
        ty = y + img_h + padding
        draw.text((tx, ty), cap, fill="black", font=font)

    return canvas


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, required=True, help="Dataset to use", choices=["small", "large"])
    args = parser.parse_args()
    ds = args.dataset

    image_names = os.listdir(f"./{ds}")
    aux_save_dir = "Aux_small" if ds == "small" else "Aux_large"
    os.makedirs(aux_save_dir, exist_ok=True)


    for rot in [0, 90, 180, 270]:
        src_dir = f"./{ds}/{rot}_imgs"
        out_dir = f"./{aux_save_dir}/img_grid/rot{rot}"

        if os.path.exists(out_dir):
            flush_directory(out_dir)
        else:
            os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(src_dir):
            print(f"Warning: {src_dir} does not exist, skipping.")
            continue

        imgs = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"))
        ]
        if not imgs:
            print(f"No images in {src_dir}")
            continue

        print(f"Processing {len(imgs)} images in {src_dir}...")
        for name in tqdm(imgs, desc=f"rot{rot}"):
            try:
                path = os.path.join(src_dir, name)
                grid = compose_grid(
                    path,
                    font_path=DEFAULT_TTF,   
                    font_size=30,           
                    padding=50,
                    spacing=50,
                    border_width=4,
                    outer_padding=30
                )
                grid.save(os.path.join(out_dir, name), dpi=(300, 300))
            except Exception as e:
                print(f"Error on {name}: {e}")
                continue