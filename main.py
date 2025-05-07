import torch
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader

from Modules.config import *
from Modules.preprocess_data import load_coco_annotations, build_dataframe, get_class_to_idx, compute_class_weights, get_class_weights_tensor
from Modules.dataset import FoodDataset, collate_fn
from Modules.model_architecture import load_detr_model
from Modules.train import train

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Annotation Dataset
train_coco, train_json = load_coco_annotations(TRAIN_ANNOTATION_FILE)
val_coco, val_json = load_coco_annotations(VAL_ANNOTATION_FILE)

# Preprocess Data
train_df = build_dataframe(train_json)
class_to_idx = get_class_to_idx(train_json)
weights = compute_class_weights(train_df)
weight_tensor = get_class_weights_tensor(weights, class_to_idx)

# Load Processor for DETR Model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Load Dataset
train_dataset = FoodDataset(train_coco, TRAIN_IMAGES_FILE, processor)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Load DETR Model
model = load_detr_model(num_classes=len(weight_tensor), device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Train Model
train(model, train_dataloader, optimizer, device)
