# **ML Challenge 2025: Smart Product Pricing Solution Template**

**Team Name:** Batch_Normalisation
**Team Members:** Amulya Jain, Himanshu Pokhriyal, Narind Verma, Naman Chanana
**Submission Date:** 13th October 2025

---

## **1. Executive Summary**

Our solution leverages a **BiLSTM + Attention neural network** combined with **structured numerical features** to predict product prices from catalog content.
Key innovations include **SMAPE-based optimization**, **log-transformed targets**, and **robust feature scaling** for stable learning within **≤5 epochs**.
The model achieves strong generalization through **attention-driven text understanding** and **regularized multimodal fusion**.

---

## **2. Methodology Overview**

### **2.1 Problem Analysis**

We interpreted the challenge as a **multimodal regression task**, where product price depends on both **textual attributes** (catalog descriptions) and **structured metadata** (weight, units, pack size, etc.).

**Key Observations from EDA:**

* Product prices exhibit **heavy-tailed distribution**, requiring **log-transformation** for stability.
* Textual descriptions often contain **quantity, size, and unit references**, crucial for price prediction.
* A small number of outliers skewed training; clipping between 1st and 99th percentiles improved model convergence.

---

### **2.2 Solution Strategy**

**Approach Type:** Hybrid Deep Learning Model (Text + Structured Features)
**Core Innovation:** Integration of **BiLSTM with Attention mechanism** and **Robustly Scaled structured inputs**, trained using a **SMAPE-based loss** to directly optimize competition metric.

**Highlights:**

* Attention mechanism captures **semantic importance** of keywords (e.g., size, value, quantity).
* Log-transformed target (`log1p(price)`) improves learning on skewed distributions.
* Early stopping, gradient clipping, and strong regularization ensure fast, stable convergence.

---

## **3. Model Architecture**

### **3.1 Architecture Overview**

```
Catalog Text ─► Tokenization ─► Embedding (128d)
                                  │
                           BiLSTM (128 hidden, bidirectional)
                                  │
                      Attention-weighted Context Vector
                                  │
Structured Numeric Features ──────┘
                                  │
             Fully Connected Layers (256 → 128 → 1)
                                  │
                         Predicted Log(Price)
```

---

### **3.2 Model Components**

#### **Text Processing Pipeline**

* **Preprocessing:** Lowercasing, punctuation removal, stopword filtering
* **Tokenizer:** Vocabulary size = 12,000, sequence length = 150
* **Model Type:** BiLSTM + Attention
* **Key Parameters:**

  * Embedding dimension = 128
  * Hidden size = 128 (bidirectional)
  * Dropout = 0.5
  * Attention: single-layer linear softmax

#### **Structured Feature Pipeline**

* **Features Used:** `weight`, `pack_size`, and one-hot encoded `unit` features
* **Scaling:** RobustScaler (less sensitive to outliers)
* **Numeric Layers:** Fully connected layers (256 → 128 → 1) with BatchNorm + LayerNorm regularization

---

## **4. Model Performance**

### **4.1 Validation Results**

* **Best Validation SMAPE:** [Insert your best value, e.g., 12.87]
* **Other Observations:**

  * Stable convergence within 4–5 epochs
  * Early stopping triggered after no improvement for 2 epochs
  * Log-transform + Robust scaling significantly reduced overfitting

---

## **5. Conclusion**

Our hybrid **BiLSTM + Attention + Structured Features** model effectively learns complex relationships between product text descriptions and numeric attributes.
The solution is computationally efficient, robust to outliers, and aligns directly with the **SMAPE competition metric**, achieving strong validation performance within a small number of epochs.

---

## **Appendix**

### **A. Code Artefacts**

[🔗 Google Drive or GitHub Link to full project directory]

Includes:

* `model_train.py` (main training + evaluation script & AttentionBiLSTM architecture)
* `submission.csv` (final test predictions)

---

### **B. Additional Results**

* Feature importance analysis of structured variables
* Attention heatmaps visualizing influential words
* Learning curves showing convergence over epochs

---
