import cv2
import torch
import numpy as np
from PIL import Image
from collections import deque, Counter, defaultdict
from model import FaceNetResNet # Assuming model.py exists and contains FaceNetResNet
from torchvision import transforms
import os
import pandas as pd
import logging # Import logging

# Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # Used for os.path.splitext correctly

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Load ArcFace model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
try:
    model = FaceNetResNet().to(device)
    model.load_state_dict(torch.load("arcface_backbone.pth", map_location=device))
    model.eval()
    logger.info("✅ ArcFace model loaded successfully.")
except FileNotFoundError:
    logger.error("❌ Error: 'arcface_backbone.pth' not found. Make sure the model weights are in the correct directory.")
    exit()
except Exception as e:
    logger.error(f"❌ Error loading ArcFace model: {e}")
    exit()

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_embedding(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(image)
        emb = emb / emb.norm(dim=1, keepdim=True) # Normalize the embedding
    return emb.cpu().numpy()

# === Load Known Embeddings ===
known_embeddings = []
known_names = []
train_dir = "data/train"

if not os.path.exists(train_dir):
    logger.error(f"❌ Error: Training directory '{train_dir}' not found. Please create it and add face images.")
    exit()

logger.info(f"Loading known embeddings from '{train_dir}'...")
for person_name in os.listdir(train_dir):
    person_folder = os.path.join(train_dir, person_name)
    if not os.path.isdir(person_folder):
        continue
    
    # Try to find at least one image in the person's folder
    img_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if not img_files:
        logger.warning(f"⚠️ No images found for {person_name} in {person_folder}. Skipping.")
        continue

    person_embeddings = []
    for img_file in img_files:
        img_path = os.path.join(person_folder, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            emb = get_embedding(img)
            person_embeddings.append(emb)
        except Exception as e:
            logger.error(f"❌ Error processing image {img_path} for {person_name}: {e}")
            
    if person_embeddings:
        avg_emb = np.mean(np.vstack(person_embeddings), axis=0)
        known_embeddings.append(avg_emb)
        known_names.append(person_name)

if not known_embeddings:
    logger.warning("⚠️ No known embeddings loaded. Recognition will not be possible.")
    known_embeddings = np.array([])
else:
    known_embeddings = np.vstack(known_embeddings)
    logger.info(f"✅ Loaded {len(known_names)} known embeddings.")


# === Similarity Function ===
def predict_name(embedding, threshold=0.7):
    if known_embeddings.size == 0:
        return "Unknown", 0.0

    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    sims = np.dot(known_embeddings, embedding.T).flatten()
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    if best_score >= threshold:
        return known_names[best_idx], best_score
    else:
        return "Unknown", best_score

# === Video Input ===
video_path = "videos/anne.mp4" # Make sure this path is correct
if not os.path.exists(video_path):
    logger.error(f"❌ Error: Video file '{video_path}' not found.")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    logger.error(f"❌ Error: Could not open video file {video_path}")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    logger.error("❌ Error: Could not load Haar cascade classifier. Check opencv-python installation or file path.")
    exit()

# === Video Writer Setup ===
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Corrected from CAP_PROP_HEIGHT
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
output_video_path = "recognized_videos/anne- to check resutls of video file.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
if not out.isOpened():
    logger.error(f"❌ Error: Could not create video writer for {output_video_path}. Check codec or path.")
    out = None
else:
    logger.info(f"🎥 Output video will be saved to: {output_video_path}")

# === Buffers and Configs ===
recent_predictions = deque(maxlen=15)
person_scores = defaultdict(list)
min_consistency = 5

logger.info("Starting video processing. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_name = "Detecting..."

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(60, 60))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size == 0 or w == 0 or h == 0:
            logger.warning("⚠️ Detected face region is empty or has zero dimensions. Skipping embedding for this frame.")
            recent_predictions.append("Unknown")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Unreadable", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            emb = get_embedding(face_pil)
            name, score = predict_name(emb, threshold=0.7)

            if name != "Unknown":
                recent_predictions.append(name)
                person_scores[name].append(score)
            else:
                recent_predictions.append("Unknown")

            if recent_predictions:
                counted = Counter(recent_predictions)
                most_common_pred, count = counted.most_common(1)[0]
                
                if most_common_pred != "Unknown" and count >= min_consistency:
                    display_name = most_common_pred
                elif most_common_pred == "Unknown" and count >= min_consistency:
                    display_name = "Unknown"
                else:
                    display_name = "Verifying..."

            color = (0, 255, 0) if display_name in known_names else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        recent_predictions.append("No Face")
        display_name = "No Face"
        cv2.putText(frame, display_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if out:
        out.write(frame)
    cv2.imshow("Video Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        logger.info("User requested exit.")
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

# === Save Excel/CSV Results ===
output_csv = "all_video_results.csv"

all_results_for_csv = []
if person_scores:
    for name, scores_list in person_scores.items():
        if scores_list:
            avg_confidence = np.mean(scores_list)
            detections = len(scores_list)
            video_name = os.path.basename(video_path)
            all_results_for_csv.append({
                "Video Name": video_name,
                "Top Actor": name, # <--- CHANGED HERE: Now writes 'Top Actor'
                "Detections": detections,
                "Avg Confidence": round(avg_confidence, 4)
            })

    if all_results_for_csv:
        new_df = pd.DataFrame(all_results_for_csv)
        if os.path.exists(output_csv):
            df_existing = pd.read_csv(output_csv)
            
            # <--- CHANGED HERE: Now looks for 'Top Actor' in existing CSV
            existing_combinations = df_existing[['Video Name', 'Top Actor']].apply(tuple, axis=1)
            new_combinations = new_df[['Video Name', 'Top Actor']].apply(tuple, axis=1) # <--- Also changed here for new_df

            rows_to_add = new_df[~new_combinations.isin(existing_combinations)]
            
            if not rows_to_add.empty:
                df_existing = pd.concat([df_existing, rows_to_add], ignore_index=True)
                df_existing.to_csv(output_csv, index=False)
                logger.info(f"✅ New recognition results appended to: {output_csv}")
            else:
                logger.info(f"ℹ️ No new recognition results to append to {output_csv} for this video.")
        else:
            new_df.to_csv(output_csv, index=False)
            logger.info(f"✅ Recognition results saved to new file: {output_csv}")
        
        logger.info(f"🎥 Annotated video saved to: {output_video_path}")
        logger.info("\n--- Summary of Detections ---")
        for row in all_results_for_csv:
            logger.info(f"Video: {row['Video Name']} | Name: {row['Top Actor']} | Detections: {row['Detections']} | Avg Confidence: {row['Avg Confidence']:.4f}")
    else:
        logger.warning("⚠️ No recognized faces to log in CSV.")
else:
    logger.warning("⚠️ No faces detected or no matches found during video processing.")


# === Plotting Functions ===
def generate_and_save_plots(person_scores: dict, video_path: str, graphs_dir: str = "graphs", threshold: float = 0.7):
    """
    Generates and saves relevant plots based on recognition results.
    """
    os.makedirs(graphs_dir, exist_ok=True)
    
    video_name_stem = Path(video_path).stem # Use pathlib for robust stem extraction

    logger.info(f"📊 Generating plots for video '{video_name_stem}' analysis...")

    all_confidences = [score for scores_list in person_scores.values() for score in scores_list]

    if not all_confidences:
        logger.warning("No confidence scores available for plotting.")
        return

    # --- Plot 1: Overall Recognition Confidence Distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(all_confidences, bins=30, kde=True, color='skyblue')
    plt.title(f'this one {video_name_stem}')
    plt.xlabel('Confidence Score (Cosine Similarity)')
    plt.ylabel('Frequency')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plot_filename = os.path.join(graphs_dir, f"{video_name_stem}_confidence_distribution this one.png")
    plt.savefig(plot_filename)
    plt.close()
    logger.info(f"Saved: {plot_filename}")

    plot_data = []
    for name, scores_list in person_scores.items():
        if scores_list:
            plot_data.append({
                "Top Actor": name, # <--- CHANGED HERE: Plots use 'Top Actor'
                "Total Detections": len(scores_list),
                "Average Confidence": np.mean(scores_list)
            })
    
    if not plot_data:
        logger.warning("No data for per-actor plots.")
        return

    df_plots = pd.DataFrame(plot_data).sort_values(by="Total Detections", ascending=False)

    # --- Plot 2: Total Detections per Top Actor ---
    plt.figure(figsize=(12, max(6, len(df_plots) * 0.5)))
    sns.barplot(x='Total Detections', y='Top Actor', data=df_plots, palette='viridis') # <--- CHANGED HERE: 'Top Actor'
    plt.title(f'Total Detections per Top Actor in {video_name_stem}')
    plt.xlabel('Number of Detections (Frames)')
    plt.ylabel('Top Actor') # <--- CHANGED HERE: 'Top Actor'
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    plot_filename = os.path.join(graphs_dir, f"{video_name_stem}_total_detections_per_actor.png")
    plt.savefig(plot_filename)
    plt.close()
    logger.info(f"Saved: {plot_filename}")

    # --- Plot 3: Average Confidence per Top Actor ---
    df_plots_sorted_confidence = df_plots.sort_values(by="Average Confidence", ascending=False)
    plt.figure(figsize=(12, max(6, len(df_plots_sorted_confidence) * 0.5)))
    sns.barplot(x='Average Confidence', y='Top Actor', data=df_plots_sorted_confidence, palette='magma') # <--- CHANGED HERE: 'Top Actor'
    plt.title(f'Average Confidence per Top Actor in {video_name_stem}')
    plt.xlabel('Average Confidence Score')
    plt.ylabel('Top Actor') # <--- CHANGED HERE: 'Top Actor'
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    plot_filename = os.path.join(graphs_dir, f"{video_name_stem}_average_confidence_per_actor.png")
    plt.savefig(plot_filename)
    plt.close()
    logger.info(f"Saved: {plot_filename}")
        
    logger.info("✅ All plots generated and saved.")


# --- Call plotting function after video processing and CSV save ---
recognition_threshold = 0.7 
generate_and_save_plots(person_scores, video_path, graphs_dir="graphs", threshold=recognition_threshold)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Confusion Matrix ===
true_identity = "Anne"  # Set this to the actual person in the video
all_preds = list(recent_predictions)

# Convert predictions to binary: 1 = correct, 0 = wrong
y_true = []
y_pred = []

for pred in all_preds:
    if pred == "No Face":
        continue  # Skip frames with no face
    y_true.append(1 if true_identity != "Unknown" else 0)
    y_pred.append(1 if pred == true_identity else 0)

if y_true and y_pred:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Correct", "Wrong"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix for '{true_identity}'")
    plt.show()
else:
    logger.warning("⚠️ Not enough prediction data to generate confusion matrix.")
