from pathlib import Path
import time

import torch
# from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from local_siglip_v2_5 import convert_v2_5_from_siglip
from PIL import Image

SAMPLE_IMAGE_PATH = Path("imgs/images_aesthetic/aesthetic1.jpg")  # 替换成你的图片路径

# load model and preprocessor
model, preprocessor = convert_v2_5_from_siglip(
    predictor_name_or_path="models/aesthetic_predictor_v2_5.pth",
    encoder_model_name_or_path="models/models--google--siglip-so400m-patch14-384/snapshots/siglip-so400m-patch14-384",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# 不使用 bfloat16，不使用 cuda，直接使用默认 float32 CPU
# model = model.to(torch.bfloat16).cuda()  # 删除

# load image to evaluate
image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")

# preprocess image
pixel_values = preprocessor(images=image, return_tensors="pt").pixel_values
# .to(torch.bfloat16).cuda()  # 删除

# predict aesthetic score with time measurement
with torch.inference_mode():
    start_time = time.time()
    score = model(pixel_values).logits.squeeze().float().cpu().numpy()
end_time = time.time()

# print result
print(f"Aesthetics score: {score:.2f}")
print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
