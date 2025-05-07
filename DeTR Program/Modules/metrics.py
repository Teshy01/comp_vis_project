import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


def log_loss_per_epoch(model, dataloader, processor, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def compute_confusion_matrix(model, dataloader, processor, device, id2label):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits.cpu()

            probs = F.softmax(logits, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)

            for true, pred in zip(targets, pred_classes):
                if len(true["class_labels"]) == 0:
                    continue
                y_true.append(true["class_labels"][0].item())
                y_pred.append(pred[0].item())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2label.values()))
    disp.plot(xticks_rotation="vertical")
    plt.tight_layout()
    plt.show()

# In metrics.py
def save_predictions_to_coco_format(predictions, output_path):
    with open(output_path, "w") as f:
        json.dump(predictions, f)

def evaluate_coco(ann_json_path, pred_json_path):
    # Load ground truth COCO annotations
    coco_gt = COCO(ann_json_path)
    
    # Load predictions (already in COCO format, but NOT a COCO object)
    with open(pred_json_path, "r") as f:
        predictions = json.load(f)

    # Load predictions into a COCO object (as a result dataset)
    coco_dt = coco_gt.loadRes(predictions)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
