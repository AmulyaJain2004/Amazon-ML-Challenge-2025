#!/usr/bin/env python3
"""
Image Downloader Script for Amazon ML Dataset
Downloads images from train.csv and test.csv into separate folders for training and testing.
Uses the existing utils.py download functionality with enhanced error handling and progress tracking.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json

# Import from utils package (parent directory)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import download_images

class ImageDownloader:
    def __init__(self, dataset_folder='../dataset', output_folder='images'):
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.train_folder = os.path.join(output_folder, 'train')
        self.test_folder = os.path.join(output_folder, 'test')
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_folder = 'logs'
        os.makedirs(log_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_folder, f'image_download_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_directories(self):
        """Create necessary directories for image storage"""
        directories = [self.output_folder, self.train_folder, self.test_folder]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
            
    def load_datasets(self):
        """Load train and test datasets"""
        try:
            train_path = os.path.join(self.dataset_folder, 'train.csv')
            test_path = os.path.join(self.dataset_folder, 'test.csv')
            
            self.logger.info(f"Loading training dataset from: {train_path}")
            self.train_df = pd.read_csv(train_path)
            
            self.logger.info(f"Loading test dataset from: {test_path}")
            self.test_df = pd.read_csv(test_path)
            
            self.logger.info(f"Train dataset shape: {self.train_df.shape}")
            self.logger.info(f"Test dataset shape: {self.test_df.shape}")
            
            # Check if image_link column exists
            if 'image_link' not in self.train_df.columns:
                self.logger.warning("No 'image_link' column found in train dataset")
                self.train_df['image_link'] = None
                
            if 'image_link' not in self.test_df.columns:
                self.logger.warning("No 'image_link' column found in test dataset")
                self.test_df['image_link'] = None
                
        except Exception as e:
            self.logger.error(f"Error loading datasets: {e}")
            raise
            
    def check_existing_images(self, folder):
        """Check how many images already exist in the folder"""
        if os.path.exists(folder):
            existing_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
            return len(existing_files)
        return 0
        
    def download_train_images(self, sample_size=None):
        """Download training images"""
        self.logger.info("Starting training images download...")
        
        train_data = self.train_df.copy()
        if sample_size:
            train_data = train_data.sample(n=min(sample_size, len(train_data)), random_state=42)
            self.logger.info(f"Sampling {len(train_data)} training images")
            
        # Filter out rows with valid image links
        valid_links = train_data['image_link'].dropna()
        valid_links = valid_links[valid_links.str.startswith(('http://', 'https://'))].unique()
        valid_links_list = valid_links.tolist()  # Convert to list
        
        self.logger.info(f"Found {len(valid_links_list)} valid training image links")
        
        existing_count = self.check_existing_images(self.train_folder)
        self.logger.info(f"Existing training images: {existing_count}")
        
        if len(valid_links_list) > 0:
            download_images(valid_links_list, self.train_folder)
            
        final_count = self.check_existing_images(self.train_folder)
        self.logger.info(f"Training images after download: {final_count}")
        
        return final_count
        
    def download_test_images(self, sample_size=None):
        """Download test images"""
        self.logger.info("Starting test images download...")
        
        test_data = self.test_df.copy()
        if sample_size:
            test_data = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
            self.logger.info(f"Sampling {len(test_data)} test images")
            
        # Filter out rows with valid image links
        valid_links = test_data['image_link'].dropna()
        valid_links = valid_links[valid_links.str.startswith(('http://', 'https://'))].unique()
        valid_links_list = valid_links.tolist()  # Convert to list
        
        self.logger.info(f"Found {len(valid_links_list)} valid test image links")
        
        existing_count = self.check_existing_images(self.test_folder)
        self.logger.info(f"Existing test images: {existing_count}")
        
        if len(valid_links_list) > 0:
            download_images(valid_links_list, self.test_folder)
            
        final_count = self.check_existing_images(self.test_folder)
        self.logger.info(f"Test images after download: {final_count}")
        
        return final_count
        
    def create_image_metadata(self):
        """Create metadata files mapping sample_ids to image filenames"""
        train_metadata = []
        test_metadata = []
        
        # Process training data
        for _, row in self.train_df.iterrows():
            if pd.notna(row.get('image_link', None)):
                filename = Path(row['image_link']).name
                image_path = os.path.join(self.train_folder, filename)
                if os.path.exists(image_path):
                    train_metadata.append({
                        'sample_id': row['sample_id'],
                        'image_link': row['image_link'],
                        'local_path': image_path,
                        'filename': filename
                    })
                    
        # Process test data
        for _, row in self.test_df.iterrows():
            if pd.notna(row.get('image_link', None)):
                filename = Path(row['image_link']).name
                image_path = os.path.join(self.test_folder, filename)
                if os.path.exists(image_path):
                    test_metadata.append({
                        'sample_id': row['sample_id'],
                        'image_link': row['image_link'],
                        'local_path': image_path,
                        'filename': filename
                    })
                    
        # Save metadata
        train_meta_df = pd.DataFrame(train_metadata)
        test_meta_df = pd.DataFrame(test_metadata)
        
        train_meta_path = os.path.join(self.output_folder, 'train_image_metadata.csv')
        test_meta_path = os.path.join(self.output_folder, 'test_image_metadata.csv')
        
        train_meta_df.to_csv(train_meta_path, index=False)
        test_meta_df.to_csv(test_meta_path, index=False)
        
        self.logger.info(f"Saved training metadata: {train_meta_path} ({len(train_meta_df)} images)")
        self.logger.info(f"Saved test metadata: {test_meta_path} ({len(test_meta_df)} images)")
        
        return train_meta_df, test_meta_df
        
    def create_summary_report(self, train_count, test_count):
        """Create a summary report of the download process"""
        summary = {
            'download_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'train_samples': len(self.train_df),
                'test_samples': len(self.test_df),
                'train_images_downloaded': train_count,
                'test_images_downloaded': test_count
            },
            'folder_structure': {
                'output_folder': self.output_folder,
                'train_folder': self.train_folder,
                'test_folder': self.test_folder
            }
        }
        
        summary_path = os.path.join(self.output_folder, 'download_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Created summary report: {summary_path}")
        return summary
        
    def run_download(self, train_sample_size=None, test_sample_size=None):
        """Run the complete download process"""
        self.logger.info("Starting image download process...")
        
        # Setup
        self.create_directories()
        self.load_datasets()
        
        # Download images
        train_count = self.download_train_images(train_sample_size)
        test_count = self.download_test_images(test_sample_size)
        
        # Create metadata and summary
        self.create_image_metadata()
        summary = self.create_summary_report(train_count, test_count)
        
        self.logger.info("Image download process completed!")
        self.logger.info(f"Training images: {train_count}")
        self.logger.info(f"Test images: {test_count}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Download images for Amazon ML dataset')
    parser.add_argument('--dataset-folder', default='../dataset', help='Path to dataset folder')
    parser.add_argument('--output-folder', default='images', help='Output folder for images')
    parser.add_argument('--train-sample', type=int, help='Sample size for training images')
    parser.add_argument('--test-sample', type=int, help='Sample size for test images')
    parser.add_argument('--train-only', action='store_true', help='Download only training images')
    parser.add_argument('--test-only', action='store_true', help='Download only test images')
    
    args = parser.parse_args()
    
    downloader = ImageDownloader(args.dataset_folder, args.output_folder)
    
    if args.train_only:
        downloader.create_directories()
        downloader.load_datasets()
        train_count = downloader.download_train_images(args.train_sample)
        print(f"Downloaded {train_count} training images")
    elif args.test_only:
        downloader.create_directories()
        downloader.load_datasets()
        test_count = downloader.download_test_images(args.test_sample)
        print(f"Downloaded {test_count} test images")
    else:
        summary = downloader.run_download(args.train_sample, args.test_sample)
        print("Download Summary:")
        print(f"Training images: {summary['dataset_info']['train_images_downloaded']}")
        print(f"Test images: {summary['dataset_info']['test_images_downloaded']}")

if __name__ == "__main__":
    main()