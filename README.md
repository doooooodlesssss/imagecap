# 🧠 Spectral-Guided Multimodal Image Captioning with LLM Refinement  
### *Integrating Frequency-Domain Fusion and Spectral Attention for Enhanced Visual-Linguistic Reasoning*

---

## 📘 Overview
This repository presents **Spectral Captioning**, a novel deep learning framework for **image caption generation** that unifies spatial reasoning, frequency-domain fusion, and large language model (LLM) refinement.

The model integrates:

- **YOLOv8** → object-level semantics (what is present, where it is)
- **Xception** → scene-level features (global layout, context)
- **Dual Attention** → independent attention for each modality  
- **DCT Frequency Fusion** → noise-robust global structural fusion  
- **Spectral Attention** → FFT-based global dependency modeling  
- **GRU Decoder** → caption generation  
- **LLM Refinement** → final grammar, richness, factual grounding  

This architecture is designed for **stronger grounding**, **better compositionality**, and **efficient long-range feature interaction**.

---

## 🧩 Architecture Overview

### **1️⃣ Dual Visual Feature Extraction**
- **YOLOv8**: detects objects, bounding boxes, class labels, and confidence scores  
- **Xception**: extracts high-level semantic context  

These together encode both **local details** and **global scene information**.

---

### **2️⃣ Dual Attention Refinement**
Each feature stream is passed through a **Bahdanau-style attention** module separately.

This prevents feature interference and allows:
- object attention → focuses on key entities  
- scene attention → focuses on high-level context  

---

### **3️⃣ DCT Frequency-Domain Fusion**
Steps:
1. Apply **2D DCT** on both attended feature maps  
2. Retain **low-frequency components** (rich in structure)  
3. **Elementwise multiply** the frequency maps  
4. Apply **inverse DCT** to reconstruct a fused, global-aware feature  

This improves robustness against noise and lighting variations.

---

### **4️⃣ Spectral Attention (Core Novelty)**
Instead of self-attention (O(n²)), we use:

- **FFT → learnable complex spectral weights → inverse FFT**  

This yields:
- global receptive field  
- cheaper computation (O(n log n))  
- smoother, more coherent feature propagation  

---

### **5️⃣ GRU Caption Decoder**
A lightweight GRU sequentially generates a caption from the fused spectral features.

Trained with:
- Cross Entropy Loss  
- CIDEr optimization  
- Teacher forcing + scheduled sampling  

---

### **6️⃣ LLM-Guided Caption Refinement**
A multimodal LLM (MiniGPT-4 / LLaVA-style) receives:
- detected objects  
- raw GRU caption  
- image embeddings  

It refines the caption by:
- adding missing attributes  
- removing hallucinations  
- improving grammar and coherence  

This produces **human-like, grounded captions**.
