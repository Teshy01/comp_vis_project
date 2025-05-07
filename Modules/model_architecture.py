import torch
from transformers import DetrForObjectDetection

def load_detr_model(num_classes, device):
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True)
    model.class_labels_classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    model.config.num_labels = num_classes
    model.config.id2label = {i: f"class_{i}" for i in range(num_classes - 1)}
    model.config.label2id = {f"class_{i}": i for i in range(num_classes - 1)}
    return model.to(device)
