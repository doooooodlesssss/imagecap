# A Frequency-Domain Dual Attention and Spectral Fusion Framework for Robust Image Captioning with Instruction-Tuned VLM Caption Refinement

---

## 🏗️ Overview

This repository presents a **novel image captioning architecture** that integrates **frequency-domain spectral attention**, **dual-branch visual reasoning**, and **instruction-tuned visual language model (VLM) refinement** for generating accurate, detailed, and grounded image captions.

The framework fuses **YOLOv8-based object detection** and **Xception-based scene classification** using **Discrete Cosine Transform (DCT)** in the frequency domain. It then applies a **Spectral Attention Module** using FFT to capture long-range dependencies efficiently, followed by a **GRU-based decoder** and an **LLM-guided refinement stage** for factual correctness.

---

## 🔍 Key Contributions

* **Dual-Stream Encoder:** Object-level features from YOLOv8 + scene-level features from Xception.
* **Dual Bahdanau Attention:** Independent attention over object and scene embeddings.
* **Frequency-Domain Fusion (DCT):** Low-frequency spectral merging for compact and meaningful representations.
* **Spectral Attention Module (FFT-based):** Captures global feature interactions with reduced complexity.
* **GRU Caption Decoder:** Generates draft captions.
* **VLM Refinement:** Multimodal LLM (e.g., MiniGPT-4/LLaVA) improves caption factuality and grounding.
* **Improved Performance on MSCOCO-2014** compared to classical CNN+RNN architectures.

---

## 🧩 Architecture Summary

1. Input Image
2. YOLOv8 + Xception Feature Extraction
3. Dual Bahdanau Attention
4. DCT-based Frequency Fusion
5. FFT-based Spectral Attention
6. GRU Decoder
7. Instruction-Tuned VLM Caption Refinement

---

## 📦 Installation

### Requirements

```
Python >= 3.10
TensorFlow >= 2.12
torch >= 2.0
opencv-python==4.6.0.66
numpy < 1.27
matplotlib
tqdm
pycocotools
pandas
```

### Setup

```bash
git clone https://github.com/<your-username>/Spectral-Captioning-VLM.git
cd Spectral-Captioning-VLM
pip install -r requirements.txt
```

To install YOLOv8:

```bash
pip install ultralytics
```

---

## 🧠 Training Pipeline

### 1. Prepare the MSCOCO-2014 dataset

```
/datasets/coco2014/
 ├── train2014/
 ├── val2014/
 └── annotations/
```

### 2. Extract Features

```bash
python extract_features.py
```

### 3. Train the Model

```bash
python train_spectral_captioning.py
```

### 4. Evaluate Metrics

```bash
python evaluate_coco_metrics.py
```

---

## 📊 Quantitative Results

| Model                    | BLEU-1    | BLEU-4    | METEOR    | ROUGE-L   | CIDEr     | SPICE     |
| ------------------------ | --------- | --------- | --------- | --------- | --------- | --------- |
| Baseline (Xception+GRU)  | 0.399     | 0.066     | 0.142     | 0.324     | 0.248     | 0.081     |
| + YOLO Object Stream     | 0.431     | 0.079     | 0.151     | 0.339     | 0.284     | 0.088     |
| + Dual Attention Fusion  | 0.456     | 0.089     | 0.158     | 0.351     | 0.309     | 0.093     |
| + DCT Frequency Fusion   | 0.478     | 0.098     | 0.163     | 0.362     | 0.334     | 0.099     |
| + Spectral Attention     | 0.495     | 0.106     | 0.168     | 0.374     | 0.359     | 0.104     |
| + VLM Refinement (Final) | **0.512** | **0.114** | **0.173** | **0.386** | **0.382** | **0.110** |

---

## 🎯 Qualitative Examples

| Image | Baseline Caption     | Proposed Model Output                                                                |
| ----- | -------------------- | ------------------------------------------------------------------------------------ |
| 🏂    | “People skiing.”     | “A group of skiers descend a snow-covered mountain under cloudy skies.”              |
| 🐕    | “A dog playing.”     | “A brown dog runs across the grass chasing a yellow tennis ball in a sunny park.”    |
| 🚆    | “A train on tracks.” | “A modern train passes through an urban station surrounded by tall glass buildings.” |

---

## 🧩 Future Work

* Integrating Vision Transformers for encoder replacement
* RLHF-based caption diversity enhancement
* Cross-domain generalization (Flickr30k, VizWiz)
* Audio-visual captioning


## 💬 Contact

📧 [diya.bangera.ug23@nsut.ac.in](mailto:diya.bangera.ug23@nsut.ac.in)

