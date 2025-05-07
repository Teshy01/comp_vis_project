import torch
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Load the pre-trained YOLOv8 model (e.g., yolov8n.pt - Nano version)
    model = YOLO('yolov8n.pt')  # Ensure this is the correct model path and name
    
    # Train
    model.train(
        data=str('datasets/UECFOOD100_YOLO/data.yaml'),  # Path to data.yaml
        epochs=100,                       # Number of training epochs
        imgsz=640,                        # Image size (YOLO default is 640x640)
        batch=32,                         # Batch size
        name='UECFOOD100_YOLOV11',        # Experiment name
        device=0,                         # GPU id (or 'cpu')
        degrees=10,
        translate=0.2,
        lr0=0.001,                        # Initial learning rate
        lrf=0.1,                          # Learning rate final value (multiplier)
        weight_decay=0.0005,              # Weight decay (L2 regularization)
        patience=10,                      # Early stopping patience
        scale = 0.5,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.55
)

if __name__ == '__main__':
    main()


