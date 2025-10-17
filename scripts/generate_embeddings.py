#!/usr/bin/env python3
"""
ResNet Image Embeddings Generator
Generates embeddings for training and test images using ResNet deep learning model
Creates CSV files in the dataset folder for use in ML pipeline
Optimized for both CPU and GPU processing with automatic hardware detection
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import warnings
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductImageDataset(Dataset):
    """Dataset for loading product images with optimized preprocessing"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load image with optimized settings
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
                
            # Extract image ID from filename (without extension)
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            return image, image_id, image_path
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a default black image if loading fails
            default_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                default_image = self.transform(default_image)
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            return default_image, image_id, image_path

class EmbeddingGenerator:
    """Generalized embedding generator using ResNet models with automatic optimization"""
    
    def __init__(self, model_name='resnet101', device=None, use_mixed_precision=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Auto-detect mixed precision based on GPU availability
        if use_mixed_precision is None:
            self.use_mixed_precision = torch.cuda.is_available()
        else:
            self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        
        # Initialize model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Enable optimizations for GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Try to compile model for better performance (PyTorch 2.0+)
            try:
                self.model = torch.compile(self.model, mode='default')
                logger.info("Model compiled with torch.compile for optimal performance")
            except:
                logger.info("torch.compile not available, using standard model")
        
        # Define optimized image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"EmbeddingGenerator initialized with {model_name} on {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        
    def _load_model(self):
        """Load and modify ResNet model for feature extraction"""
        if self.model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2')
            feature_dim = 2048
        elif self.model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif self.model_name == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif self.model_name == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V2')
            feature_dim = 2048
        elif self.model_name == 'resnet152':
            model = models.resnet152(weights='IMAGENET1K_V2')
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Remove the final classification layer to get features
        model = nn.Sequential(*list(model.children())[:-1])
        return model
    
    def generate_embeddings_for_folder(self, image_folder, batch_size=32, max_images=None):
        """Generate embeddings for all images in a folder"""
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        if not os.path.exists(image_folder):
            logger.error(f"Image folder not found: {image_folder}")
            return pd.DataFrame()
        
        for filename in os.listdir(image_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(image_folder, filename))
        
        if not image_paths:
            logger.warning(f"No images found in {image_folder}")
            return pd.DataFrame()
        
        # Limit number of images if specified
        if max_images:
            image_paths = image_paths[:max_images]
        
        logger.info(f"Found {len(image_paths)} images in {image_folder}")
        
        # Create optimized dataset and dataloader
        dataset = ProductImageDataset(image_paths, transform=self.transform)
        
        # Optimize dataloader based on device
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        embeddings_list = []
        image_ids = []
        failed_images = []
        start_time = time.time()
        
        # Warm up GPU if available
        if torch.cuda.is_available() and len(dataset) > 0:
            logger.info("Warming up GPU...")
            dummy_input = torch.randn(min(batch_size, len(dataset)), 3, 224, 224).to(self.device)
            with torch.no_grad():
                if self.use_mixed_precision:
                    from torch.cuda.amp import autocast
                    with autocast():
                        _ = self.model(dummy_input)
                else:
                    _ = self.model(dummy_input)
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_idx, (batch_images, batch_ids, batch_paths) in enumerate(tqdm(dataloader, desc="Generating embeddings")):
                try:
                    # Move to device efficiently
                    batch_images = batch_images.to(self.device, non_blocking=True)
                    
                    # Generate embeddings with optional mixed precision
                    if self.use_mixed_precision:
                        from torch.cuda.amp import autocast
                        with autocast():
                            features = self.model(batch_images)
                    else:
                        features = self.model(batch_images)
                    
                    features = features.view(features.size(0), -1)  # Flatten
                    
                    # Move to CPU and convert to numpy
                    features_np = features.cpu().numpy()
                    
                    embeddings_list.append(features_np)
                    image_ids.extend(batch_ids)
                    
                    # Progress logging for large datasets
                    if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
                        elapsed = time.time() - start_time
                        images_processed = (batch_idx + 1) * batch_size
                        rate = images_processed / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {images_processed} images at {rate:.1f} images/sec")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    failed_images.extend(batch_paths)
                    continue
        
        if not embeddings_list:
            logger.error("No embeddings generated")
            return pd.DataFrame()
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings_list)
        
        # Create DataFrame
        embedding_columns = [f'embedding_{i}' for i in range(all_embeddings.shape[1])]
        embeddings_df = pd.DataFrame(all_embeddings, columns=embedding_columns)
        embeddings_df.insert(0, 'image_id', image_ids)
        
        # Performance statistics
        total_time = time.time() - start_time
        total_images = len(embeddings_df)
        rate = total_images / total_time if total_time > 0 else 0
        
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        logger.info(f"Processing time: {total_time:.2f}s")
        logger.info(f"Processing rate: {rate:.1f} images/sec")
        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images")
        
        return embeddings_df
    
    def generate_embeddings_from_csv(self, csv_file, image_folder, id_column='sample_id', batch_size=32):
        """Generate embeddings for images listed in a CSV file"""
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded CSV with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file}: {e}")
            return pd.DataFrame()
        
        if id_column not in df.columns:
            logger.error(f"Column '{id_column}' not found in CSV")
            return pd.DataFrame()
        
        # Get list of image IDs
        image_ids = df[id_column].astype(str).tolist()
        
        # Find corresponding image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_paths = []
        missing_images = []
        
        for img_id in image_ids:
            found = False
            for ext in image_extensions:
                img_path = os.path.join(image_folder, f"{img_id}{ext}")
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    found = True
                    break
            
            if not found:
                missing_images.append(img_id)
                # Add a placeholder for missing images
                image_paths.append(None)
        
        if missing_images:
            logger.warning(f"Missing {len(missing_images)} images")
        
        logger.info(f"Processing {len([p for p in image_paths if p])} images")
        
        # Filter out None paths and corresponding IDs
        valid_paths = []
        valid_ids = []
        for i, path in enumerate(image_paths):
            if path is not None:
                valid_paths.append(path)
                valid_ids.append(image_ids[i])
        
        if not valid_paths:
            logger.error("No valid image paths found")
            return pd.DataFrame()
        
        # Create optimized dataset and dataloader
        dataset = ProductImageDataset(valid_paths, transform=self.transform)
        
        # Optimize dataloader based on device
        num_workers = 4 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        embeddings_list = []
        processed_ids = []
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (batch_images, batch_ids, batch_paths) in enumerate(tqdm(dataloader, desc="Generating embeddings")):
                try:
                    # Move to device efficiently
                    batch_images = batch_images.to(self.device, non_blocking=True)
                    
                    # Generate embeddings with optional mixed precision
                    if self.use_mixed_precision:
                        from torch.cuda.amp import autocast
                        with autocast():
                            features = self.model(batch_images)
                    else:
                        features = self.model(batch_images)
                    
                    features = features.view(features.size(0), -1)  # Flatten
                    
                    # Move to CPU and convert to numpy
                    features_np = features.cpu().numpy()
                    
                    embeddings_list.append(features_np)
                    processed_ids.extend(batch_ids)
                    
                    # Progress logging for large datasets
                    if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
                        elapsed = time.time() - start_time
                        images_processed = (batch_idx + 1) * batch_size
                        rate = images_processed / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {images_processed} images at {rate:.1f} images/sec")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        if not embeddings_list:
            logger.error("No embeddings generated")
            return pd.DataFrame()
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings_list)
        
        # Create DataFrame
        embedding_columns = [f'embedding_{i}' for i in range(all_embeddings.shape[1])]
        embeddings_df = pd.DataFrame(all_embeddings, columns=embedding_columns)
        embeddings_df.insert(0, 'sample_id', processed_ids)
        
        # Performance statistics
        total_time = time.time() - start_time
        total_images = len(embeddings_df)
        rate = total_images / total_time if total_time > 0 else 0
        
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        logger.info(f"Processing time: {total_time:.2f}s")
        logger.info(f"Processing rate: {rate:.1f} images/sec")
        
        return embeddings_df

def main():
    """Main function to generate embeddings with automatic optimization"""
    
    # Auto-detect optimal configuration based on hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Optimal settings based on device
    if torch.cuda.is_available():
        # GPU configuration (optimized for powerful GPUs)
        batch_size = 128  # Can be increased to 256-512 for high-end GPUs
        model_name = 'resnet101'  # Best quality for GPU processing
        logger.info("GPU detected - using high-performance configuration")
    else:
        # CPU configuration
        batch_size = 16  # Smaller batch for CPU
        model_name = 'resnet50'  # Balanced model for CPU
        logger.info("CPU detected - using CPU-optimized configuration")
    
    config = {
        'model_name': model_name,
        'batch_size': batch_size,
        'device': device,
        'train_csv': '../dataset/train.csv',
        'test_csv': '../dataset/test.csv',
        'train_images_folder': '../images/train',
        'test_images_folder': '../images/test',
        'output_train_embeddings': '../dataset/embeddings_columns_train.csv',
        'output_test_embeddings': '../dataset/embeddings_columns_test.csv',
        'max_images_per_set': None  # Set to a number to limit images for testing
    }
    
    logger.info("Starting ResNet embedding generation...")
    logger.info(f"Using device: {config['device']}")
    logger.info(f"Model: {config['model_name']}")
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(
        model_name=config['model_name'],
        device=config['device']
    )
    
    # Create dataset directory if it doesn't exist (for output files)
    dataset_dir = os.path.dirname(config['output_train_embeddings'])
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Generate training embeddings
    logger.info("=" * 50)
    logger.info("Generating TRAINING embeddings...")
    
    if os.path.exists(config['train_csv']) and os.path.exists(config['train_images_folder']):
        train_embeddings = generator.generate_embeddings_from_csv(
            csv_file=config['train_csv'],
            image_folder=config['train_images_folder'],
            batch_size=config['batch_size']
        )
        
        if not train_embeddings.empty:
            train_embeddings.to_csv(config['output_train_embeddings'], index=False)
            logger.info(f"Training embeddings saved to: {config['output_train_embeddings']}")
            logger.info(f"Training embeddings shape: {train_embeddings.shape}")
        else:
            logger.error("Failed to generate training embeddings")
    else:
        logger.warning("Training CSV or images folder not found, skipping training embeddings")
    
    # Generate test embeddings
    logger.info("=" * 50)
    logger.info("Generating TEST embeddings...")
    
    if os.path.exists(config['test_csv']) and os.path.exists(config['test_images_folder']):
        test_embeddings = generator.generate_embeddings_from_csv(
            csv_file=config['test_csv'],
            image_folder=config['test_images_folder'],
            batch_size=config['batch_size']
        )
        
        if not test_embeddings.empty:
            test_embeddings.to_csv(config['output_test_embeddings'], index=False)
            logger.info(f"Test embeddings saved to: {config['output_test_embeddings']}")
            logger.info(f"Test embeddings shape: {test_embeddings.shape}")
        else:
            logger.error("Failed to generate test embeddings")
    else:
        logger.warning("Test CSV or images folder not found, skipping test embeddings")
    
    logger.info("=" * 50)
    logger.info("Embedding generation complete!")

if __name__ == "__main__":
    main()