import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import FaceNetResNet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import combinations

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceNetResNet().to(device)
model.load_state_dict(torch.load("arcface_backbone.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
        emb = F.normalize(emb)
    return emb

# Load image paths
base_path = "data/test"
persons = os.listdir(base_path)

pairs = []
labels = []

# Create all possible pairs
for i, person1 in enumerate(persons):
    person1_imgs = [os.path.join(base_path, person1, f) for f in os.listdir(os.path.join(base_path, person1)) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Same person pairs
    for img1, img2 in combinations(person1_imgs, 2):
        pairs.append((img1, img2))
        labels.append(1)

    # Different person pairs
    for person2 in persons[i+1:]:
        person2_imgs = [os.path.join(base_path, person2, f) for f in os.listdir(os.path.join(base_path, person2)) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for img1 in person1_imgs:
            for img2 in person2_imgs:
                pairs.append((img1, img2))
                labels.append(0)

# Compare and collect predictions
predictions = []
for img1, img2 in pairs:
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    sim = F.cosine_similarity(emb1, emb2).item()
    pred = 1 if sim > 0.5 else 0
    predictions.append(pred)

# Plot confusion matrix
cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(cm, display_labels=["Different", "Same"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
