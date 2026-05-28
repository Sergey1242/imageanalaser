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
