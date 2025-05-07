import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, coco, img_dir, processor, transform=None):
        self.coco = coco
        self.img_ids = list(coco.imgs.keys())
        self.img_dir = img_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(path).convert("RGB")

        boxes = []
        labels = []
        for an in anns:
            x, y, w, h = an['bbox']
            boxes.append([x, y, x + w, y + h])
            category_id = an['category_id']
            mapped_id = self.coco.getCatIds().index(category_id)
            labels.append(mapped_id)

        return image, {
            "image_id": torch.tensor([img_id]),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_labels": torch.tensor(labels, dtype=torch.int64),
        }

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


