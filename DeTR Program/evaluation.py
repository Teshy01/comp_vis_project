import torch
import os
from pycocotools.coco import COCO
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import DataLoader
from Modules.dataset import FoodDataset, collate_fn
from Modules.metrics import evaluate_coco, save_predictions_to_coco_format
from tqdm import tqdm
from torchvision import transforms

# ---------------------------
# Setup paths and parameters
# ---------------------------
ANNOTATION_FILE = "archive/raw_data/public_validation_set_2.0/annotations.json"
IMAGE_DIR = "archive/raw_data/public_validation_set_2.0/images"
MODEL_PATH = "detr_resnet_50_10_epochs.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
TRANSFORM = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

# ---------------------------
# Load model and processor
# ---------------------------
print("Loading model...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True)
model.class_labels_classifier = torch.nn.Linear(model.config.hidden_size, 499)
model.config.num_labels = 499
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------------------
# Load dataset
# ---------------------------
print("Loading dataset...")
coco = COCO(ANNOTATION_FILE)
dataset = FoodDataset(coco, IMAGE_DIR, processor, transform=TRANSFORM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Run prediction & save
# ---------------------------
print("Generating predictions...")
all_predictions = []
image_ids = []

with torch.no_grad():
    for images, targets in tqdm(dataloader):
        pixel_values = torch.stack([TRANSFORM(img) for img in images]).to(DEVICE)
        outputs = model(pixel_values=pixel_values)

        for i, output in enumerate(outputs.logits):
            id = targets[i]['image_id'].item()
            image_ids.append(id)
            boxes = outputs.pred_boxes[i].cpu()
            scores = outputs.logits[i].softmax(-1).cpu()

            pred_scores, pred_labels = torch.max(scores, dim=-1)

            for box, label, score in zip(boxes, pred_labels, pred_scores):
                if label == 498:  # skip 'no object'
                    continue
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                prediction = {
                    "image_id": id,
                    "category_id": coco.getCatIds()[label],
                    "bbox": [x_min.item(), y_min.item(), width.item(), height.item()],
                    "score": score.item()
                }
                all_predictions.append(prediction)

# ---------------------------
# Save predictions to JSON
# ---------------------------
prediction_file = "outputs/coco_predictions.json"
save_predictions_to_coco_format(all_predictions, prediction_file)
# save_predictions_to_coco_format(all_predictions,dataloader, processor, DEVICE, prediction_file)


# ---------------------------
# Evaluate with COCO metrics
# ---------------------------
evaluate_coco(ANNOTATION_FILE, prediction_file)
