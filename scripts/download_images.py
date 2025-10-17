#!/usr/bin/env python3
"""
Simple Image Downloader Script
Downloads a sample of images from the Amazon ML dataset for quick testing
"""

import os
import sys

# Add parent directory to path to import from scripts folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from image_downloader import ImageDownloader

def download_sample_images():
    """Download a small sample of images for testing"""
    print("🚀 Starting image download process...")
    print("=" * 50)
    
    # Initialize downloader
    downloader = ImageDownloader(
        dataset_folder='../dataset',
        output_folder='images'
    )
    
    # Download with smaller sample sizes for testing
    # Adjust these numbers based on your needs
    TRAIN_SAMPLE_SIZE = 100  # Download 100 training images
    TEST_SAMPLE_SIZE = 50    # Download 50 test images
    
    try:
        summary = downloader.run_download(
            train_sample_size=TRAIN_SAMPLE_SIZE,
            test_sample_size=TEST_SAMPLE_SIZE
        )
        
        print("\n" + "=" * 50)
        print("✅ Download completed successfully!")
        print(f"📁 Images saved in: {downloader.output_folder}")
        print(f"🎯 Training images: {summary['dataset_info']['train_images_downloaded']}")
        print(f"🧪 Test images: {summary['dataset_info']['test_images_downloaded']}")
        print(f"📊 Check download_summary.json for details")
        
        # Show folder structure
        print("\n📂 Folder Structure:")
        print(f"   {downloader.output_folder}/")
        print(f"   ├── train/              ({summary['dataset_info']['train_images_downloaded']} images)")
        print(f"   ├── test/               ({summary['dataset_info']['test_images_downloaded']} images)")
        print(f"   ├── train_image_metadata.csv")
        print(f"   ├── test_image_metadata.csv")
        print(f"   └── download_summary.json")
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False
        
    return True

def download_full_dataset():
    """Download the complete dataset (use with caution - large download)"""
    print("⚠️  WARNING: This will download ALL images from the dataset!")
    print("This could be thousands of images and take a very long time.")
    
    response = input("Are you sure you want to continue? (yes/NO): ").lower()
    if response != 'yes':
        print("Download cancelled.")
        return
        
    downloader = ImageDownloader()
    summary = downloader.run_download()
    
    print(f"✅ Full dataset download completed!")
    print(f"Training images: {summary['dataset_info']['train_images_downloaded']}")
    print(f"Test images: {summary['dataset_info']['test_images_downloaded']}")

def check_dataset_info():
    """Check basic information about the dataset"""
    print("📊 Checking dataset information...")
    
    try:
        import pandas as pd
        
        # Load datasets
        train_df = pd.read_csv('../dataset/train.csv')
        test_df = pd.read_csv('../dataset/test.csv')
        
        print(f"📈 Training dataset: {train_df.shape[0]:,} samples")
        print(f"🧪 Test dataset: {test_df.shape[0]:,} samples")
        
        # Check for image_link column
        train_has_images = 'image_link' in train_df.columns
        test_has_images = 'image_link' in test_df.columns
        
        print(f"🖼️  Training images available: {'✅' if train_has_images else '❌'}")
        print(f"🖼️  Test images available: {'✅' if test_has_images else '❌'}")
        
        if train_has_images:
            valid_train_links = train_df['image_link'].dropna().nunique()
            print(f"   Valid training image links: {valid_train_links:,}")
            
        if test_has_images:
            valid_test_links = test_df['image_link'].dropna().nunique()
            print(f"   Valid test image links: {valid_test_links:,}")
            
    except Exception as e:
        print(f"❌ Error checking dataset: {e}")

def main():
    print("🎯 Amazon ML Image Downloader")
    print("=" * 40)
    
    while True:
        print("\nSelect an option:")
        print("1. 📊 Check dataset information")
        print("2. 🔽 Download sample images (recommended)")
        print("3. ⚠️  Download full dataset (caution)")
        print("4. ❌ Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            check_dataset_info()
        elif choice == '2':
            if download_sample_images():
                print("\n✨ Sample download completed! You can now use these images for training.")
        elif choice == '3':
            download_full_dataset()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()