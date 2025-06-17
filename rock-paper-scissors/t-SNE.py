import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.manifold import TSNE

# ---- Configuration ----
IMAGES_DIR = "../rock-paper-scissors/train/train/"  # Change if images are elsewhere
CSV_PATH = "train/train/_annotations.csv"
BATCH_SIZE = 32
EMBEDDING_DIM = 512  # For ResNet18

class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['class']
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(CSV_PATH)
    print("Data sample:\n", df.head())
    print("Class distribution:\n", df['class'].value_counts())

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(df, IMAGES_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(DEVICE)
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(DEVICE)
            features = model(images).cpu().numpy()
            all_embeddings.append(features)
            all_labels.extend(labels)

    embeddings = np.concatenate(all_embeddings, axis=0)
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}
    labels_numeric = np.array([label_to_index[label] for label in all_labels])
    class_names = sorted(set(all_labels))

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("Set1", n_colors=len(class_names))
    sns.scatterplot(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
        hue=[class_names[i] for i in labels_numeric],
        palette=palette,
        legend='full'
    )
    plt.title('t-SNE of Image Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig("tsne_embeddings.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()