import numpy as np
import argparse
import time 
from pathlib import Path
from typing import List, Tuple

from keras.models import Model
from keras.layers import Dense, Dropout
# from keras.preprocessing.image import load_img, img_to_array
from keras.utils import load_img, img_to_array
import tensorflow as tf

from tensorflow.keras.applications import NASNetMobile, nasnet
from utils.score_utils import mean_score, std_score
from nima_mobilenet import gather_images, make_grid


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
WEIGHTS_PATH_NASNET = Path(__file__).with_suffix("").parent / "weights" / "nasnet_weights.h5"
TARGET_SIZE = (224, 224)


def build_model() -> Model:
    if not WEIGHTS_PATH_NASNET.exists():
        raise FileNotFoundError(
            f"Weights not found → {WEIGHTS_PATH_NASNET}\n"
            "Please place nasnet_weights.h5 there."
        )

    base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights(WEIGHTS_PATH_NASNET)
    
    return model

def predict_image(model: Model, img_path: Path) -> Tuple[float, float, float]:
    """Return *(mean, std)* score for a single image."""
    pil_img = load_img(img_path, target_size=TARGET_SIZE)
    
    x = img_to_array(pil_img)
    x = np.expand_dims(x, axis=0)

    x = nasnet.preprocess_input(x)

    t0 = time.perf_counter()
    scores = model.predict(x, batch_size=1, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return mean_score(scores), std_score(scores), latency_ms



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
    parser.add_argument('-dir', type=str, default=None,
                        help='Pass a directory to evaluate the images in it')

    parser.add_argument('-img', type=str, default=[None], nargs='+',
                        help='Pass one or more image paths to evaluate them')

    parser.add_argument('-rank', type=str, default='true',
                        help='Whether to tank the images after they have been scored')
    return parser.parse_args()


def main():
    args = parse_args()
    rank = args.rank.lower() in ("true", "yes", "t", "1")

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
        mean, std, ms = predict_image(model, p)
        scored.append((p.name, mean, ms))
        scores_only.append(mean)
        print(f"{p.name:<30} | {mean:6.3f} | +-{std:5.3f} | {ms:6.1f} ms")
    
    if rank and len(scored) > 1:
        print("*" * 40, "Ranking Images", "*" * 40)
        for idx, (name, mean, _) in enumerate(sorted(scored, key=lambda x: x[1], reverse=True), 1):
            print(f"{idx:2d}. {name:<30} : {mean:6.3f}")

    grid = make_grid(img_paths, scores_only)
    out_path = Path(f"grid_with_scores-{WEIGHTS_PATH_NASNET.name}.jpg")
    grid.save(out_path, quality=90)
    print(f"\nComposite saved → {out_path}")

if __name__ == "__main__":
    main()

