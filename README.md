**"A Frequency-Domain Dual Attention and Spectral Fusion Framework for Robust Image Captioning with Instruction-Tuned VLM Caption Refinement."**

---

## Overview

This repository presents a **novel image captioning architecture** that integrates **frequency-domain spectral attention**, **dual-branch visual reasoning**, and **instruction-tuned visual language model (VLM) refinement** for generating accurate, detailed, and grounded image captions.

The framework fuses **YOLOv8-based object detection** and **Xception-based scene classification** using **Discrete Cosine Transform (DCT)** in the frequency domain. It then applies a **Spectral Attention Module** using FFT to capture long-range dependencies efficiently, followed by a **GRU-based decoder** and an **LLM-guided refinement stage** for factual correctness.

---

## Key Features & Architecture
This framework improves upon classical CNN+RNN architectures, resulting in improved performance on the MSCOCO-2014 dataset.

### Key Contributions
* **Dual-Stream Encoder:** Object-level features from YOLOv8 + scene-level features from Xception.
* **Dual Bahdanau Attention:** Independent attention over object and scene embeddings.
* **Frequency-Domain Fusion (DCT):** Low-frequency spectral merging for compact and meaningful representations.
* **Spectral Attention Module (FFT-based):** Captures global feature interactions with reduced complexity.
* **GRU Caption Decoder:** Generates draft captions.
* **VLM Refinement:** Multimodal LLM (e.g., MiniGPT-4/LLaVA) improves caption factuality and grounding.
### Architecture Flow
1.  Input Image
2.  YOLOv8 + Xception Feature Extraction
3.  Dual Bahdanau Attention
4.  DCT-based Frequency Fusion
5.  FFT-based Spectral Attention
6.  GRU Decoder
7.  Instruction-Tuned VLM Caption Refinement

---

## Base Repository
This work builds upon and extends the baseline architecture from the following open-source repository:
[abdelhadie-amalla/image_captioning](https://github.com/abdelhadie-almalla/image_captioning)
Our framework enhances the original CNN–GRU pipeline with dual attention, frequency-domain fusion, spectral attention, and instruction-tuned VLM caption refinement.

---

## Quick Start
1.  Create a Python environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  Place your original script (the file you uploaded) as:
    ```bash
    src/original_imgcap.py
    ```
    *(Note: Do not rename the internal code — keep it unchanged)*
3.  Edit `configs/config.yaml` to point to your local dataset and weight locations.

---

## Usage
Run the main project stages using the provided scripts:
1.  **Run feature extraction:**
    ```bash
    bash scripts/extract_features.sh
    ```
2.  **Train the model:**
    ```bash
    bash scripts/train_model.sh
    ```
3.  **Evaluate the model:**
    ```bash
    bash scripts/evaluate_model.sh
    ```
4.  **Produce refined captions:**
    *(The VLM refinement wrapper uses the rule-based fallback by default)*
    ```bash
    bash scripts/refine_captions.sh
    ```

---

## Project Structure

SpectralCaptioning-VLM/ 
├── README.md 
├── requirements.txt 
├── configs/config.yaml 
├── src/original_imgcap.py 
├── src/improv_imgcap.py 
├── src/vlm_refiner_wrapper.py 
├── src/helpers.py 
├── scripts/*.sh 

---

## Contact
📧 [diya.bangera.ug23@nsut.ac.in](mailto:diya.bangera.ug23@nsut.ac.in)
