# 🎬 Name That Celeb – ArcFace  

A PyTorch-based project that detects and recognizes **celebrities in videos** using **ArcFace embeddings** with a ResNet backbone.  
The system extracts face embeddings, compares them against known identities, and overlays predictions directly on video frames.  

---

## 🚀 Features
- Face embeddings generated with **ArcFace + ResNet18**  
- Celebrity recognition from both **images and videos**  
- Evaluation metrics: accuracy (80%), classification report, and confusion matrix  
- Sample **recognized videos** included in this repo  
- Easy to extend to new datasets (just drop your images in `data/train` and `data/test`)  

---

## For Images 
python evaluate.py

## Running On Videos
python video.py

## 📌 Results

- Achieved 80% accuracy on the test dataset for videos.
- ArcFace embeddings give strong separation between celebrity identities
- Confusion matrix highlights which celebs are most easily confused
- Works in real-time on GPU for video recognition
