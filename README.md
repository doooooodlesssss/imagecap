# A Frequency-Domain Dual Attention and Spectral Fusion Framework for Robust Image Captioning with Instruction-Tuned VLM Caption Refinement

---

## Overview

This repository presents a **novel image captioning architecture** that integrates **frequency-domain spectral attention**, **dual-branch visual reasoning**, and **instruction-tuned visual language model (VLM) refinement** for generating accurate, detailed, and grounded image captions.

The framework fuses **YOLOv8-based object detection** and **Xception-based scene classification** using **Discrete Cosine Transform (DCT)** in the frequency domain. It then applies a **Spectral Attention Module** using FFT to capture long-range dependencies efficiently, followed by a **GRU-based decoder** and an **LLM-guided refinement stage** for factual correctness.

---

## Key Contributions

* **Dual-Stream Encoder:** Object-level features from YOLOv8 + scene-level features from Xception.
* **Dual Bahdanau Attention:** Independent attention over object and scene embeddings.
* **Frequency-Domain Fusion (DCT):** Low-frequency spectral merging for compact and meaningful representations.
* **Spectral Attention Module (FFT-based):** Captures global feature interactions with reduced complexity.
* **GRU Caption Decoder:** Generates draft captions.
* **VLM Refinement:** Multimodal LLM (e.g., MiniGPT-4/LLaVA) improves caption factuality and grounding.
* **Improved Performance on MSCOCO-2014** compared to classical CNN+RNN architectures.

---

## Architecture Summary

1. Input Image
2. YOLOv8 + Xception Feature Extraction
3. Dual Bahdanau Attention
4. DCT-based Frequency Fusion
5. FFT-based Spectral Attention
6. GRU Decoder
7. Instruction-Tuned VLM Caption Refinement

---

Base Paper Repository

This work builds upon and extends the baseline architecture from the following open-source repository:

[abdelhadie-amalla/image_captioning
](https://github.com/abdelhadie-almalla/image_captioning)

Our framework enhances the original CNN–GRU pipeline with dual attention, frequency-domain fusion, spectral attention, and instruction-tuned VLM caption refinement.

---

## Quantitative Results

| Model                              | BLEU-1    | BLEU-2    | BLEU-3    | BLEU-4    | METEOR    | ROUGE-L   | CIDEr     | SPICE     |
| ---------------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| **Baseline (Xception + GRU)**      | 0.399     | 0.224     | 0.123     | 0.066     | 0.142     | 0.324     | 0.248     | 0.081     |
| **+ YOLO Object Features**         | 0.395     | 0.221     | 0.125     | 0.079     | 0.151     | **0.320** | 0.284     | 0.088     |
| **+ Dual Attention Fusion**        | 0.398     | 0.226     | 0.125     | **0.079** | 0.155     | 0.340     | 0.305     | **0.087** |
| **+ DCT Frequency Fusion**         | 0.405     | 0.230     | 0.122     | **0.075** | 0.160     | 0.355     | 0.330     | 0.092     |
| **+ Spectral Attention Module**    | 0.470     | 0.285     | 0.180     | 0.102     | **0.161** | 0.360     | 0.350     | 0.095     |
| **+ LLM Refinement (Final Model)** | **0.512** | **0.324** | **0.201** | **0.114** | **0.173** | **0.386** | **0.382** | **0.110** |

---

If you'd like, I can **insert this into your full README**, keeping formatting consistent and placing it exactly where it fits best.

---

## Qualitative Examples

| Image | Baseline Caption     | Proposed Model Output                                                                |
| ----- | -------------------- | ------------------------------------------------------------------------------------ |
| 🏂    | “People skiing on the snow.”     | "A group of skiers skiing a snow-covered mountain under clouds.”              |
| 🐕    | “A dog playing with balls.”     | “A brown dog runs across the grass chasing a yellow tennis ball in a sunny park.”    |
| 🚆    | “A train on tracks.” | “A modern train passes through an urban station surrounded by tall glass buildings.” |

---

## Future Work

* Integrating Vision Transformers for encoder replacement
* RLHF-based caption diversity enhancement
* Cross-domain generalization (Flickr30k, VizWiz)
* Audio-visual captioning


## 💬 Contact

📧 [diya.bangera.ug23@nsut.ac.in](mailto:diya.bangera.ug23@nsut.ac.in)

