import requests
import torch
from PIL import Image
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1
import time 
from pathlib import Path

#
# Load the aesthetics predictor
#
model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"

predictor = AestheticsPredictorV1.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

#
# Download sample image
#
# url = "https://github.com/shunk031/simple-aesthetics-predictor/blob/master/assets/a-photo-of-an-astronaut-riding-a-horse.png?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
# load image to evaluate

SAMPLE_IMAGE_PATH = Path("imgs/images_aesthetic/aesthetic5.jpg")  # 替换成你的图片路径
image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
#
# Preprocess the image
#
inputs = processor(images=image, return_tensors="pt")

#
# Move to GPU
#
device = "cpu"
predictor = predictor.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

#
# Inference for the image
#
with torch.no_grad(): # or `torch.inference_model` in torch 1.9+
    start_time = time.time()
    outputs = predictor(**inputs)
    end_time = time.time()
prediction = outputs.logits

print(f"Aesthetics score: {prediction}")
print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")