import argparse
import numpy as np
import time 
from pathlib import Path
from typing import List, Tuple

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_incep
from keras.utils import load_img, img_to_array
import tensorflow as tf

from utils.score_utils import mean_score, std_score
from nima_mobilenet import gather_images, make_grid

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
WEIGHTS_PATH_INCEPTION = Path(__file__).with_suffix("").parent / "weights" / "inception_resnet_weights.h5"
TARGET_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
    parser.add_argument('-dir', type=str, default=None,
                        help='Pass a directory to evaluate the images in it')

    parser.add_argument('-img', type=str, default=[None], nargs='+',
                        help='Pass one or more image paths to evaluate them')

    parser.add_argument('-resize', type=str, default='false',
                        help='Resize images to 224x224 before scoring')

    parser.add_argument('-rank', type=str, default='true',
                        help='Whether to tank the images after they have been scored')
    return parser.parse_args()
    
    
def build_model() -> Model:
    if not WEIGHTS_PATH_INCEPTION.exists():
        raise FileNotFoundError(
            f"Weights not found → {WEIGHTS_PATH_INCEPTION}\n"
            "Please place inception_resnet_weights.h5 there."
        )

    base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights(WEIGHTS_PATH_INCEPTION)
    
    return model

def score_image(model: Model, img_path: Path, resize: bool = False) -> Tuple[float, float, float]:
    """Return *(mean, std)* score for a single image."""
    pil_img = load_img(img_path, target_size=TARGET_SIZE if resize else None)
    arr = img_to_array(pil_img)
    
    arr = np.expand_dims(arr, axis=0)
    
    arr = preprocess_input_incep(arr)  # MobileNet: RGB→BGR & scale to [-1,1]
    
    t0 = time.perf_counter()
    scores = model.predict(arr, batch_size=1, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    return mean_score(scores), std_score(scores), latency_ms

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
        # print(f"Evaluating : {p}\nNIMA Score : {mean:.3f} ± ({std:.3f})\n")
    
    if rank and len(scored) > 1:
        print("*" * 40, "Ranking Images", "*" * 40)
        for idx, (name, mean, _) in enumerate(sorted(scored, key=lambda x: x[1], reverse=True), 1):
            print(f"{idx:2d}. {name:<30} : {mean:6.3f}")

    grid = make_grid(img_paths, scores_only)
    out_path = Path(f"grid_with_scores-{WEIGHTS_PATH_INCEPTION.name}.jpg")
    grid.save(out_path, quality=90)
    print(f"\nComposite saved → {out_path}")
        
if __name__ == "__main__":
    main()