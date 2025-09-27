import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FaceNetResNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load test data
test_dataset = datasets.ImageFolder("data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes

# Load model
model = FaceNetResNet().to(device)
model.load_state_dict(torch.load("arcface_backbone.pth", map_location=device))
model.eval()

# Get embeddings and labels
embeddings_list = []
labels_list = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        embeddings = model(imgs)
        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels)

embeddings = torch.cat(embeddings_list)
labels = torch.cat(labels_list)

# Compute centroids
centroids = []
unique_labels = torch.unique(labels)
for ul in unique_labels:
    centroid = embeddings[labels == ul].mean(dim=0)
    centroids.append(centroid)
centroids = torch.stack(centroids)

# Predictions
preds = []
for emb in embeddings:
    sims = F.cosine_similarity(emb.unsqueeze(0), centroids)
    pred = torch.argmax(sims).item()
    preds.append(pred)

# Confusion matrix
true_indices = [unique_labels.tolist().index(l.item()) for l in labels]
cm = confusion_matrix(true_indices, preds)

# Slice top 20 for better readability
top_n = 20
cm_subset = cm[:top_n, :top_n]
label_subset = [str(i) for i in range(top_n)]

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(cm_subset, annot=True, fmt="d", cmap="Blues", xticklabels=label_subset, yticklabels=label_subset)
plt.title("Confusion Matrix (Top 20 Classes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

# Save
os.makedirs("eval_results", exist_ok=True)
plt.savefig("eval_results/confusion_matrix_top20.png", dpi=300)
print("✅ Confusion matrix (Top 20) saved to eval_results/confusion_matrix_top20.png")
