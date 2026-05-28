import torch
from torchvision import models
from PIL import Image
import os
import pandas as pd

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()
