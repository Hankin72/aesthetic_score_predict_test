# nima_mobilenet.py
import argparse
import numpy as np
import time 
from pathlib import Path
from typing import List, Tuple

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.utils import load_img, img_to_array
# from tensorflow.keras import Model
# from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf

from utils.score_utils import mean_score, std_score

from math import ceil, sqrt
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.load_default()          # 也可换成 truetype
LABEL_BG = (0, 0, 0, 128)                # 半透明黑
LABEL_FG = (255, 255, 255, 255)          # 白

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
WEIGHTS_PATH = Path(__file__).with_suffix("").parent / "weights" / "mobilenet_weights.h5"
TARGET_SIZE = (224, 224)


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def gather_images(folder: Path) -> List[Path]:
    """Recursively collect image files under *folder* with supported extensions."""
    return [p for p in folder.rglob("*") if p.suffix.lower() in EXTS]


def build_model() -> Model:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Weights not found → {WEIGHTS_PATH}\n"
            "Please place mobilenet_weights.h5 there."
        )

    inputs = tf.keras.Input(shape=(*TARGET_SIZE, 3))
    # base = MobileNet(include_top=False, pooling="avg", weights=None, input_tensor=inputs)
    
    base = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    
    x = Dropout(0.75)(base.output)
    output = Dense(10, activation="softmax")(x)
    
    model = Model(base.input, output)
    model.load_weights(WEIGHTS_PATH)
    
    return model


def score_image(model: Model, img_path: Path, resize: bool = False) -> Tuple[float, float, float]:
    """Return *(mean, std)* score for a single image."""
    pil_img = load_img(img_path, target_size=TARGET_SIZE if resize else None)
    arr = img_to_array(pil_img)
    
    arr = np.expand_dims(arr, axis=0)
    
    arr = preprocess_input(arr)  # MobileNet: RGB→BGR & scale to [-1,1]
    
    t0 = time.perf_counter()
    scores = model.predict(arr, batch_size=1, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    return mean_score(scores), std_score(scores), latency_ms


def make_grid(paths: List[Path], scores: List[float], cell_size=(224, 224)) -> Image.Image:
    """Stitch images into a near‑square grid with score labels."""
    n = len(paths)
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)
    W, H = cell_size
    grid = Image.new("RGB", (cols * W, rows * H), (30, 30, 30))

    for idx, (p, sc) in enumerate(zip(paths, scores)):
        r, c = divmod(idx, cols)
        im = Image.open(p).convert("RGB").resize(cell_size, Image.LANCZOS)
        draw = ImageDraw.Draw(im, "RGBA")
        text = f"{p.name} - {sc:.2f}"
        if hasattr(draw, "textbbox"):
            x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=FONT)
            tw, th = x1 - x0, y1 - y0
        else:  # Pillow <8 fallback
            tw, th = draw.textsize(text, font=FONT)
        draw.rectangle([(0, 0), (tw + 6, th + 4)], LABEL_BG)
        draw.text((3, 2), text, font=FONT, fill=LABEL_FG)
        grid.paste(im, (c * W, r * H))
    return grid

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NIMA MobileNet evaluator (refactored)")
    p.add_argument("-dir", type=str, default=None,
        help="Folder to scan images recursively (jpg/jpeg/png/bmp/tiff)",
    )
    p.add_argument("-img", type=str, nargs="+", default=[None],
        help="One or more individual image paths (space separated)",
    )
    p.add_argument("-resize", type=str, default="false",
        help="Resize input to 224×224 before scoring (string boolean, default false)",
    )
    p.add_argument("-rank", type=str, default="true",
        help="After scoring multiple images, display ranking (string boolean, default true)",
    )
    return p.parse_args()

    
def main():
    args = parse_args()
    resize = args.resize.lower() in {"true", "t", "yes", "1"}
    rank = args.rank.lower() in {"true", "t", "yes", "1"}
    
    if args.dir:
        img_paths = gather_images(Path(args.dir))
        if not img_paths:
            print("No images found in directory.")
            return
    elif args.img[0] is not None:
        img_paths = [Path(p) for p in args.img]
    else:
        raise RuntimeError("Either -dir or -img must be provided")
    
    model = build_model()
    
    scored: List[Tuple[str, float, float, float]] = []  # (name, mean, std, ms)
    scores_only: List[float] = []
    
    print("\nEvaluating images\n----------------------------------")
    for p in img_paths:
        mean, std, ms = score_image(model, p, resize)
        scored.append((p.name, mean, ms))
        scores_only.append(mean)
        print(f"{p.name:<30} | {mean:6.3f} | +-{std:5.3f} | {ms:6.1f} ms")
    
    if rank and len(scored) > 1:
        print("*" * 40, "Ranking Images", "*" * 40)
        for idx, (name, mean, _) in enumerate(sorted(scored, key=lambda x: x[1], reverse=True), 1):
            print(f"{idx:2d}. {name:<30} : {mean:6.3f}")
            
    grid = make_grid(img_paths, scores_only)
    out_path = Path(f"grid_with_scores-{WEIGHTS_PATH.name}.jpg")
    grid.save(out_path, quality=90)
    print(f"\nComposite saved → {out_path}")
        
if __name__ == "__main__":
    main()