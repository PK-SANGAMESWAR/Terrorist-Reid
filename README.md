# Closed-Set Suspect Re-Identification System

A real-time deep learning pipeline that identifies registered suspects in video streams using whole-body appearance features. Built on top of **YOLOv8** for real-time person detection and **OSNet-x1.0** for deep person re-identification (Re-ID).

---

## 📌 Architecture Overview

This system operates in three distinct phases:

```
[Offline Phase]
Curated Suspect Images ──> OSNet-x1.0 Embedding ──> Quality-Weighted Averaging ──> Database (CSV)

[Online Phase]
Video Stream ──> YOLOv8n (Person Class) ──> Bounding Box Crops ──> OSNet-x1.0 Inference
                                                                       │
                                                                       ▼
[Matching Phase]
Video Frame Output <── Bounding Box Annotation <── L2 Distance Match (Threshold = 17.0)
```

1. **Suspect Registration (Offline)**: Processes raw reference crops of a known target, filters outliers, extracts 512-dimensional feature representations using OSNet-x1.0, and computes a centroid vector saved as a lightweight CSV file.
2. **Detection & Cropping (Real-Time)**: Runs YOLOv8n on incoming frames to detect pedestrians, extract bounding boxes, and crop them.
3. **Re-ID & Matching (Real-Time)**: Converts each crop into a 512-dimensional embedding, calculates Euclidean (L2) distance against the registered suspects, and flags matches falling below a calibrated threshold of 17.0.

---

## 🛠️ Technology Stack

- **Core Framework**: PyTorch
- **Object Detection**: Ultralytics YOLOv8 (Nano variant for high FPS)
- **Feature Extraction**: OSNet-x1.0 (Pre-trained on MSMT17 dataset)
- **Image Processing**: OpenCV, Pillow, Torchvision
- **Data & Analysis**: Pandas, NumPy

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/PK-SANGAMESWAR/Terrorist-Reid.git
cd Terrorist-Reid
pip install -r requirements.txt
```

### 2. Gallery Setup
Place crop images of your target suspect into a directory (e.g., `ajmal_khasab/` or `osama_bin_laden/`). Then run the embedding generator to generate the CSV vector:
```python
# Run the notebook or script to generate average embeddings
python get_person_crop_embedding.py
```

### 3. Run Inference
To execute the matching pipeline on a test video:
```python
# Execute the real-time inference driver
python realtime_person_bbox_extract.py
```

---

## 📖 Technical Deep Dive & Interview Q&A

For a comprehensive breakdown of the design choices, edge cases, failure modes, and interview preparation questions, see the [project.md](./project.md) file.
