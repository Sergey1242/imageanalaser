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

    try:
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        print("\n" + "=" * 60)
        print(f"Изображение: {image_name}")
        print("=" * 60)

        best_class = categories[top5_catid[0]]
        best_conf = top5_prob[0].item()
        status = get_status(best_conf)

        row = {
            "Изображение": image_name,
            "Лучший класс": best_class,
            "Уверенность": round(best_conf, 4),
            "Статус": status
        }

        for i in range(5):
            class_name = categories[top5_catid[i]]
            confidence = top5_prob[i].item()

            print(
                f"{i+1}. "
                f"{class_name:<25}"
                f" | вероятность: "
                f"{confidence:.4f}"
            )

            row[f"Класс_{i+1}"] = class_name
            row[f"Вероятность_{i+1}"] = round(confidence, 4)

        print(f"\nСтатус: {status}")

        results.append(row)

        if best_conf >= CONFIDENCE_THRESHOLD:
            successful.append(row)
        else:
            uncertain.append(row)

    except Exception as e:
        print(f"Ошибка {image_name}: {e}")

pd.DataFrame(results).to_csv(
    "results.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(successful).to_csv(
    "successful_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(uncertain).to_csv(
    "uncertain_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("УДАЧНЫЕ РАСПОЗНАВАНИЯ")
print("=" * 60)

for item in successful:
    print(
        f"{item['Изображение']} → "
        f"{item['Лучший класс']} "
        f"(уверенность: {item['Уверенность']})"
    )

print("\n" + "=" * 60)
print("СОМНИТЕЛЬНЫЕ РАСПОЗНАВАНИЯ")
print("=" * 60)

for item in uncertain:
    print(
        f"{item['Изображение']} → "
        f"{item['Лучший класс']} "
        f"(уверенность: {item['Уверенность']})"
    )

print("\nГотово!")
print("Созданы файлы:")
print("results.csv")
print("successful_predictions.csv")
print("uncertain_predictions.csv")
