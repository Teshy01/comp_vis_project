import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

from Modules.dataset import FoodDataset, collate_fn
from Modules.config import VAL_ANNOTATION_FILE, VAL_IMAGES_FILE
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

def load_model(model_path, num_classes=499):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True)

    model.class_labels_classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, processor

def visualize_prediction(image, boxes, scores, labels, id2label, threshold=0.7):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)
        label_text = f"{id2label[label.item()]} ({score:.2f})"
        draw.text((x0, y0), label_text, fill="white")

    plt.imshow(image)
    plt.axis("off")
    plt.show()

def run_predictions(model_path, show_samples=5, threshold=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load_model(model_path)
    model.to(device)

    val_coco = COCO(VAL_ANNOTATION_FILE)
    val_dataset = FoodDataset(val_coco, VAL_IMAGES_FILE, processor)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    id2label = {i: f"class_{i}" for i in range(498)}
    id2label[498] = "no object"

    count = 0
    for images, _ in val_loader:
        pixel_values = [transforms.ToTensor()(img) for img in images]
        pixel_values = torch.stack(pixel_values).to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        results = processor.post_process_object_detection(outputs, target_sizes=[img.size[::-1] for img in images], threshold=threshold)[0]
        image = images[0]

        visualize_prediction(image, results['boxes'], results['scores'], results['labels'], id2label, threshold)

        count += 1
        print("Predicted boxes:", results['boxes'])
        print("Scores:", results['scores'])
        print("Labels:", results['labels'])
        if count >= show_samples:
            break
