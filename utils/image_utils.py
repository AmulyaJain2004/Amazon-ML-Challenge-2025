"""
Image Download Utilities
Enhanced version of the original utils.py with better error handling and features
"""

import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib.request
import logging
from typing import List, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_image(image_link: str, savefolder: str, timeout: int = 30) -> bool:
    """
    Download a single image from URL to specified folder
    
    Args:
        image_link: URL of the image to download
        savefolder: Directory to save the image
        timeout: Timeout in seconds for the download
        
    Returns:
        bool: True if download successful, False otherwise
    """
    if not isinstance(image_link, str) or not image_link.strip():
        return False
        
    try:
        # Extract filename from URL
        filename = Path(image_link).name
        if not filename or '.' not in filename:
            # Generate filename if not available
            filename = f"image_{hash(image_link) % 100000}.jpg"
            
        image_save_path = os.path.join(savefolder, filename)
        
        # Skip if file already exists
        if os.path.exists(image_save_path):
            return True
            
        # Create directory if it doesn't exist
        os.makedirs(savefolder, exist_ok=True)
        
        # Download with timeout
        urllib.request.urlretrieve(image_link, image_save_path)
        return True
        
    except Exception as ex:
        logger.warning(f'Failed to download {image_link}: {ex}')
        return False

def download_images(image_links: List[str], download_folder: str, max_workers: int = 50, 
                   timeout: int = 30) -> dict:
    """
    Download multiple images with multiprocessing
    
    Args:
        image_links: List of image URLs to download
        download_folder: Directory to save images
        max_workers: Maximum number of parallel workers
        timeout: Timeout per image download
        
    Returns:
        dict: Statistics about the download process
    """
    if not image_links:
        logger.warning("No image links provided")
        return {'total': 0, 'successful': 0, 'failed': 0}
    
    # Create download folder
    os.makedirs(download_folder, exist_ok=True)
    
    # Filter valid links
    valid_links = [link for link in image_links if isinstance(link, str) and link.strip()]
    logger.info(f"Starting download of {len(valid_links)} images to {download_folder}")
    
    # Setup partial function with parameters
    download_func = partial(download_image, savefolder=download_folder, timeout=timeout)
    
    # Use multiprocessing with progress bar
    successful_downloads = 0
    results = []
    
    with multiprocessing.Pool(min(max_workers, len(valid_links))) as pool:
        for result in tqdm(pool.imap(download_func, valid_links), total=len(valid_links), 
                          desc="Downloading images"):
            results.append(result)
            if result:
                successful_downloads += 1
                
        pool.close()
        pool.join()
    
    stats = {
        'total': len(valid_links),
        'successful': successful_downloads,
        'failed': len(valid_links) - successful_downloads,
        'success_rate': successful_downloads / len(valid_links) if valid_links else 0
    }
    
    logger.info(f"Download completed: {stats}")
    return stats

def validate_image_urls(urls: List[str]) -> List[str]:
    """
    Validate and filter image URLs
    
    Args:
        urls: List of URLs to validate
        
    Returns:
        List of valid URLs
    """
    valid_urls = []
    for url in urls:
        if isinstance(url, str) and url.strip():
            url = url.strip()
            if url.startswith(('http://', 'https://')):
                # Check if it looks like an image URL
                if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
                    valid_urls.append(url)
                elif 'image' in url.lower() or any(term in url.lower() for term in ['img', 'photo', 'pic']):
                    valid_urls.append(url)
    
    return valid_urls

def get_image_info(image_folder: str) -> dict:
    """
    Get information about downloaded images
    
    Args:
        image_folder: Path to the image folder
        
    Returns:
        dict: Information about images in the folder
    """
    if not os.path.exists(image_folder):
        return {'count': 0, 'total_size_mb': 0, 'extensions': {}}
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    files = os.listdir(image_folder)
    image_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
    
    total_size = 0
    extensions = {}
    
    for file in image_files:
        file_path = os.path.join(image_folder, file)
        size = os.path.getsize(file_path)
        total_size += size
        
        ext = Path(file).suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    return {
        'count': len(image_files),
        'total_size_mb': total_size / (1024 * 1024),
        'extensions': extensions,
        'avg_size_kb': (total_size / len(image_files) / 1024) if image_files else 0
    }