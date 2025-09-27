import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FaceNetResNet
from arc_margin_product import ArcMarginProduct
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# === Output paths ===
os.makedirs("eval_results", exist_ok=True)
REPORT_PATH = "eval_results/classification_report.txt"
CM_PLOT_PATH = "eval_results/confusion_matrix.png"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Load test data
test_dataset = datasets.ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
model = FaceNetResNet().to(device)
model.load_state_dict(torch.load("arcface_backbone.pth", map_location=device))
model.eval()

all_preds = []
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

# Calculate centroids
centroids = []
unique_labels = torch.unique(labels)

for ul in unique_labels:
    centroid = embeddings[labels == ul].mean(dim=0)
    centroids.append(centroid)

centroids = torch.stack(centroids)

# Predict using cosine similarity
for emb in embeddings:
    sims = F.cosine_similarity(emb.unsqueeze(0), centroids)
    pred = torch.argmax(sims).item()
    all_preds.append(pred)

all_labels = [unique_labels.tolist().index(l.item()) for l in labels]

# Accuracy and metrics
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds)

# Save classification report
with open(REPORT_PATH, "w") as f:
    f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Report saved to: {REPORT_PATH}")

# Save confusion matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(CM_PLOT_PATH)
print(f"Confusion matrix image saved to: {CM_PLOT_PATH}")
