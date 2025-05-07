import json
import pandas as pd
import numpy as np
import torch
from pycocotools.coco import COCO

def load_coco_annotations(annotation_file):
    coco = COCO(annotation_file)
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return coco, data

def get_class_to_idx(json_data):
    if isinstance(json_data, str):
        with open(json_data, 'r') as f:
            json_data = json.load(f)

    categories = sorted(json_data['categories'], key=lambda x: x['name'])
    return {cat['name']: idx for idx, cat in enumerate(categories)}

def build_dataframe(data):
    images_df = pd.DataFrame(data['images'])[['id', 'file_name']].rename(columns={'id': 'image_id'})
    categories_df = pd.DataFrame(data['categories'])[['id', 'name']].rename(columns={'id': 'category_id'})
    annotations_df = pd.DataFrame(data['annotations'])[['image_id', 'category_id', 'bbox']]
    return annotations_df.merge(categories_df, on='category_id').merge(images_df, on='image_id')

def compute_class_weights(df):
    classes = df['name'].unique()
    total = df.shape[0]
    freqs = df['name'].value_counts(normalize=True)
    median_freq = np.median(freqs.values)
    weights = {cls: median_freq / freqs[cls] for cls in classes}
    return weights

def get_class_weights_tensor(weights, class_to_idx, no_object_weight=0.1):
    ordered_weights = [weights[c] for c in sorted(class_to_idx, key=class_to_idx.get)]
    ordered_weights.append(no_object_weight)
    return torch.tensor(ordered_weights, dtype=torch.float32)
