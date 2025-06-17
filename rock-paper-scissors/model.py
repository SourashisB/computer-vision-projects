import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import joblib

# Configuration
IMAGES_DIR = "../rock-paper-scissors/train/train/"
CSV_PATH = "train/train/_annotations.csv"
MODEL_PATH = "rps_model.pt"
CLASS_MAP_PATH = "class_map.pkl"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, class_to_idx, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[row['class']]
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # Load data
    df = pd.read_csv(CSV_PATH)
    classes = sorted(df['class'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    print("Classes:", class_to_idx)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(df, IMAGES_DIR, class_to_idx, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Model
    backbone = models.resnet18(pretrained=True)
    for param in backbone.parameters():
        param.requires_grad = False  # Freeze backbone
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone = backbone.to(DEVICE)

    classifier = nn.Linear(num_features, len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    # Training loop
    backbone.eval()
    classifier.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        correct = 0
        total = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                features = backbone(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={epoch_loss/total:.4f}, Acc={acc:.4f}")

    # Save model and class map
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'num_features': num_features
    }, MODEL_PATH)
    joblib.dump(idx_to_class, CLASS_MAP_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Class mapping saved to {CLASS_MAP_PATH}")

if __name__ == '__main__':
    main()