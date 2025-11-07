# 🏆 ML Challenge 2025: Smart Product Pricing Solution

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Challenge Overview

**Team:** Batch_Normalisation  
**Members:** Amulya Jain, Himanshu Pokhriyal, Narind Verma, Naman Chanana  
**Submission:** Advanced BiLSTM + ResNet Multimodal Solution
**Achieved:** Rank 1565 out 90,000 Teams participated

### Problem Statement

Develop an ML solution that analyzes product catalog content and images to predict optimal product pricing in e-commerce. The challenge involves complex relationships between textual descriptions, visual features, and pricing dynamics.

### 🏅 Our Approach

- **Hybrid Architecture**: BiLSTM + Attention for text + ResNet embeddings for images
- **Advanced Features**: SMAPE-optimized loss, log-transformed targets, robust scaling
- **Performance**: Achieved strong validation SMAPE within 5 epochs
- **Innovation**: Multimodal fusion with attention-driven semantic understanding

### 📊 Dataset Description

| Column            | Description                                      | Type   |
| ----------------- | ------------------------------------------------ | ------ |
| `sample_id`       | Unique identifier for input sample               | String |
| `catalog_content` | Product title, description, and IPQ concatenated | Text   |
| `image_link`      | Public URL for product image download            | URL    |
| `price`           | Product price (target variable - training only)  | Float  |

## 🏗️ Project Structure

```
Amazon-ML-Challenge-2025/
├── 📁 dataset/                      # Dataset files
│   ├── train.csv                     # Training data (75k samples)
│   ├── test.csv                      # Test data (75k samples)
│   ├── sample_test.csv               # Sample test input
│   └── sample_test_out.csv           # Sample output format
├── 📁 scripts/                       # Core scripts
│   ├── generate_embeddings.py        # ResNet image embeddings generator
│   ├── download_images.py            # Image downloader
│   └── image_downloader.py           # Advanced image downloader
├── 📁 utils/                         # Utility modules
│   ├── image_utils.py                # Image processing utilities
│   └── __init__.py                   # Package initialization
├── 📁 notebooks/                     # Analysis notebooks
│   └── eda.ipynb                     # Exploratory Data Analysis
├── 📁 architecture/                  # Architecture diagrams
│   └── architecture_overview.png     # Model architecture diagram
├── 📁 images/                       # Downloaded images
│   ├── train/                        # Training images
│   └── test/                         # Test images
├── 📁 docs/                         # All Documentations
│   ├── Documentation.md              # Technical documentation of complete competition
|   ├── generate_embeddings_guide.md  # Embeddings guide
|   ├── image_downloader_guide.md     # Image downloading guide
│   └── CONTRIBUTING.MD/              # For Contribution
├── model_training.py                 # Main training script
├── best_model.pt                     # Trained model weights
├── submission.csv                    # Final predictions
├── requirements.txt                  # Requirements and Dependency file
├── .gitignore                        # Files which are not pushed
├── .gitattributes                    # For gitlfs
├── LICENSE                           # LICENSE FILE
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# For image processing (optional but recommended)
pip install torch torchvision pillow
```

### 1. Download Images (Optional)

```bash
# Download sample images for testing
python scripts/download_images.py

# Or generate embeddings directly
python scripts/generate_embeddings.py
```

### 2. Train Model

```bash
# Train the BiLSTM model
python model_training.py

# Model will be saved as best_model.pt
```

### 3. Generate Predictions

```bash
# Run inference on test set
python model_training.py --inference

# Output will be saved as submission.csv
```

## 🔬 Technical Approach

### Model Architecture

**Hybrid Multimodal Architecture:**

- **Text Branch**: BiLSTM + Attention mechanism for catalog content
- **Image Branch**: ResNet embeddings for visual features
- **Fusion Layer**: Concatenated features → Dense layers → Price prediction

### Key Features

- **SMAPE-Optimized Loss**: Direct optimization of competition metric
- **Log-Transformed Targets**: Handles skewed price distribution
- **Robust Scaling**: Resistant to outliers and anomalies
- **Attention Mechanism**: Focuses on price-relevant keywords
- **Mixed Precision**: Fast GPU training with automatic optimization

### Performance

- **Validation SMAPE**: Competitive performance within 5 epochs
- **Convergence**: Early stopping with gradient clipping
- **Generalization**: Strong regularization prevents overfitting

## 📊 Dataset Information

| Metric               | Value                                            |
| -------------------- | ------------------------------------------------ |
| **Training Samples** | 75,000                                           |
| **Test Samples**     | 75,000                                           |
| **Features**         | Catalog content + Product images                 |
| **Target**           | Product price (USD)                              |
| **Evaluation**       | SMAPE (Symmetric Mean Absolute Percentage Error) |

### Output Format

CSV file with columns:

- `sample_id`: Unique identifier matching test records
- `price`: Predicted price (positive float values)

## 🛠️ Advanced Usage

### Image Embeddings

```bash
# Generate ResNet embeddings for all images
python scripts/generate_embeddings.py

# Embeddings saved to dataset/embeddings_columns_*.csv
```

### Custom Training

```python
from model_training import Config, train_model

# Modify configuration
Config.EPOCHS = 100
Config.BATCH_SIZE = 256
Config.LR = 1e-3

# Train with custom settings
train_model()
```

### Model Analysis

```python
# Load trained model
import torch
model = torch.load('best_model.pt')

# Analyze attention weights
attention_weights = model.get_attention_weights(text_input)
```

## 📈 Results & Performance

### Validation Metrics

- **Best SMAPE**: Competitive performance
- **Convergence**: 4-5 epochs with early stopping
- **Stability**: Consistent results across multiple runs

### Model Insights

- **Attention Focus**: Keywords like "size", "quantity", "premium" receive high attention
- **Image Importance**: Visual features contribute significantly to pricing
- **Feature Engineering**: Log transformation crucial for stable learning

## 🔧 Configuration

Key parameters in `model_training.py`:

```python
class Config:
    MAX_LEN = 150          # Text sequence length
    VOCAB_SIZE = 12000     # Vocabulary size
    EMBED_DIM = 256        # Embedding dimension
    LSTM_HIDDEN = 128      # LSTM hidden size
    BATCH_SIZE = 128       # Training batch size
    EPOCHS = 50            # Maximum epochs
    LR = 3e-4             # Learning rate
```

## 📝 Submission Guidelines

1. **Format**: CSV with `sample_id` and `price` columns
2. **Completeness**: All test samples must have predictions
3. **Values**: Positive float prices only
4. **Documentation**: Technical approach documentation required

## 🚫 Academic Integrity

**STRICTLY PROHIBITED:**

- External price lookups from e-commerce sites
- Web scraping for current market prices
- Use of external pricing databases
- Any data sources beyond provided dataset

## 🤝 Contributing

See `docs/CONTRIBUTING.md` for contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 👥 Team

**Batch_Normalisation**

- Amulya Jain
- Himanshu Pokhriyal
- Narind Verma
- Naman Chanana

---

**🏆 Ready to predict prices like a pro? Let's build the future of e-commerce pricing!** 🚀
