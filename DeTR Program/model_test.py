from Modules.predict import run_predictions

run_predictions("detr_resnet_50_10_epochs.pth", show_samples=10, threshold=0.1)
