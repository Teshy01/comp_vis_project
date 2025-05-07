import torch
from tqdm import tqdm
from torchvision import transforms

def train(model, dataloader, optimizer, device, epochs=10, save_path_prefix="model_epoch"):
    transform = transforms.Compose([transforms.Resize((800, 800)), transforms.ToTensor()])

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0

        for images, targets in tqdm(dataloader, desc="Training"):
            pixel_vals = torch.stack([transform(img) for img in images]).to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(pixel_values=pixel_vals, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Avg Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"{save_path_prefix}_{epoch + 1}.pth")
