import torch
from torchvision import models
from PIL import Image
import os
import pandas as pd

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()
categories = weights.meta["categories"]
transform = weights.transforms()

image_folder = "images"

results = []
successful = []
uncertain = []

CONFIDENCE_THRESHOLD = 0.30
def get_status(confidence):
    if confidence >= 0.30:
        return "Удачное распознавание"
    return "Сомнительное распознавание"

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
