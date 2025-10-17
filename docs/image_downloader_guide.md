# Image Downloader for Amazon ML Dataset

This repository contains scripts to download images from the Amazon ML product dataset into organized folders for training and testing.

## Project Structure

```
Amazon-ML-Challenge-2025/
├── 📁 dataset/                    # Original dataset files
│   ├── train.csv                  # Training data with image_link column
│   └── test.csv                   # Test data with image_link column
├── 📁 scripts/                    # Image downloading scripts
│   ├── image_downloader.py        # Main downloader class
│   └── download_images.py         # Interactive downloader script
├── 📁 utils/                      # Utility modules
│   └── image_utils.py            # Image processing utilities
├── 📁 images/                     # Downloaded images (created after running scripts)
│   ├── train/                    # Training images
│   └── test/                     # Test images
└── 📁 notebooks/                  # Analysis notebooks
    └── eda.ipynb                 # Exploratory analysis
```

## Files Overview

- `scripts/image_downloader.py` - Main image downloader class with full functionality
- `scripts/download_images.py` - Simple interactive script for easy usage
- `utils/image_utils.py` - Utility functions for image downloading
- `notebooks/eda.ipynb` - Exploratory data analysis with usage examples## Quick Start

### Option 1: From Root Directory (Recommended)

```bash
# Run the simple interactive script
python scripts/download_images.py
```

This will show you a menu to:

1. Check dataset information
2. Download sample images (100 train + 50 test)
3. Download full dataset (use with caution)

### Option 2: Direct Script Usage

```bash
# From root directory
python scripts/image_downloader.py --train-sample 100 --test-sample 50

# From scripts directory
cd scripts
python image_downloader.py --train-only --train-sample 200

# Download only test images
python scripts/image_downloader.py --test-only --test-sample 100

# Download full dataset (warning: large!)
python scripts/image_downloader.py
```

## Folder Structure After Download

```
images/
├── train/                          # Training images
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── test/                           # Test images
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── train_image_metadata.csv        # Mapping: sample_id -> image filename
├── test_image_metadata.csv         # Mapping: sample_id -> image filename
└── download_summary.json           # Download statistics and info
```

## Features

### 🚀 **Smart Downloading**

- Uses multiprocessing for fast parallel downloads
- Skips already downloaded images
- Handles download errors gracefully
- Progress tracking with tqdm

### 📁 **Organized Structure**

- Separate folders for train/test images
- Metadata files linking sample_ids to images
- JSON summary with download statistics

### 🛡️ **Error Handling**

- Comprehensive logging
- Validates image URLs
- Continues on individual download failures
- Detailed error reporting

### ⚙️ **Flexible Options**

- Sample subset of images for testing
- Download only train or test images
- Custom output folders
- Command line interface

## Command Line Arguments

```bash
python scripts/image_downloader.py [OPTIONS]

Options:
  --dataset-folder PATH     Path to dataset folder (default: ../dataset)
  --output-folder PATH      Output folder for images (default: images)
  --train-sample N         Sample size for training images
  --test-sample N          Sample size for test images
  --train-only            Download only training images
  --test-only             Download only test images
  -h, --help              Show help message
```

## Usage Examples

### Download Small Sample for Testing

```python
# Add the scripts directory to Python path
import sys
import os
sys.path.append('scripts')

from image_downloader import ImageDownloader

downloader = ImageDownloader()
summary = downloader.run_download(train_sample_size=50, test_sample_size=25)
print(f"Downloaded {summary['dataset_info']['train_images_downloaded']} training images")
```

### Download Specific Amount

```bash
# Download 500 training images and 200 test images (from root directory)
python scripts/image_downloader.py --train-sample 500 --test-sample 200

# Or from scripts directory
cd scripts
python image_downloader.py --train-sample 500 --test-sample 200
```

### Check What You Downloaded

```python
import pandas as pd

# Load metadata to see what was downloaded
train_meta = pd.read_csv('images/train_image_metadata.csv')
test_meta = pd.read_csv('images/test_image_metadata.csv')

print(f"Training images: {len(train_meta)}")
print(f"Test images: {len(test_meta)}")
```

## Integration with Model Training

The downloaded images can be easily integrated with your ML pipeline:

```python
import pandas as pd
from PIL import Image
import os

# Load image metadata
train_meta = pd.read_csv('images/train_image_metadata.csv')
train_df = pd.read_csv('dataset/train.csv')

# Merge to get both text and image data
merged_df = train_df.merge(train_meta, on='sample_id', how='inner')

# Now you have both catalog_content (text) and local_path (images)
for _, row in merged_df.iterrows():
    text = row['catalog_content']
    price = row['price']
    image_path = row['local_path']

    # Load image
    if os.path.exists(image_path):
        image = Image.open(image_path)
        # Process image + text for your model
```

## Integration with ResNet Embeddings

After downloading images, generate ResNet embeddings:

```bash
# Generate embeddings for downloaded images
python scripts/generate_embeddings.py

# This will create:
# - dataset/embeddings_columns_train_new.csv
# - dataset/embeddings_columns_test_new.csv
```

The embeddings can then be used in your model training pipeline.

## Logs and Monitoring

- Logs are saved in `logs/` folder with timestamps
- Monitor progress in real-time
- Check `download_summary.json` for statistics
- Use metadata CSV files to track sample_id -> image mapping

## Tips for Large Downloads

1. **Start Small**: Use sample sizes first to test
2. **Check Space**: Ensure you have enough disk space
3. **Monitor Progress**: Watch logs for any issues
4. **Resume Downloads**: Script skips existing images automatically
5. **Network**: Stable internet connection recommended

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're running from the correct directory or add scripts to Python path
2. **No Images Downloaded**: Check if dataset has `image_link` column in `dataset/train.csv` and `dataset/test.csv`
3. **Permission Error**: Ensure write permissions in output folder
4. **Path Error**: Use correct relative paths - scripts expect to find dataset folder at `../dataset/` when run from scripts directory
5. **Network Timeout**: Some images may fail - this is normal and expected

### Check Download Status

```bash
# See what was downloaded
ls -la images/train/ | wc -l   # Count training images
ls -la images/test/ | wc -l    # Count test images

# Check logs
tail -f logs/image_download_*.log
```

## Requirements

- pandas
- numpy
- tqdm
- requests
- urllib
- pathlib
- multiprocessing

Install with: `pip install pandas numpy tqdm requests`
