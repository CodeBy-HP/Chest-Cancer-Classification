import os
import zipfile
import gdown
import logging
import shutil
from pathlib import Path
from typing import Optional
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """Data ingestion component for downloading and extracting datasets"""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_file(self) -> Optional[Path]:
        """Download dataset from Google Drive"""
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            
            zip_download_dir.parent.mkdir(parents=True, exist_ok=True)
            
            if zip_download_dir.exists():
                file_size = get_size(zip_download_dir)
                self.logger.info(f"File already exists: {zip_download_dir} ({file_size})")
                return zip_download_dir
            
            self.logger.info(f"Downloading from: {dataset_url}")
            self.logger.info(f"Saving to: {zip_download_dir}")
            
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?export=download&id='
            
            gdown.download(
                url=prefix + file_id,
                output=str(zip_download_dir),
                quiet=False,
                fuzzy=True
            )
            
            if zip_download_dir.exists():
                file_size = get_size(zip_download_dir)
                self.logger.info(f"Download complete: {file_size}")
                return zip_download_dir
            else:
                raise FileNotFoundError(f"Downloaded file not found: {zip_download_dir}")
                
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            raise

    def extract_zip_file(self) -> Path:
        """Extract zip file to destination directory"""
        try:
            unzip_path = self.config.unzip_dir
            zip_file_path = self.config.local_data_file
            
            if not zip_file_path.exists():
                raise FileNotFoundError(f"Zip file not found: {zip_file_path}")
            
            expected_data_folder = unzip_path / "Chest-CT-Scan-data"
            if expected_data_folder.exists() and any(expected_data_folder.iterdir()):
                file_count = len(list(expected_data_folder.rglob('*')))
                self.logger.info(f"Data already extracted at: {expected_data_folder}")
                self.logger.info(f"Total files/folders: {file_count}")
                return unzip_path
            
            unzip_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Extracting: {zip_file_path}")
            self.logger.info(f"To: {unzip_path}")
            
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                bad_file = zip_ref.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(f"Corrupted zip file detected: {bad_file}")
                
                total_files = len(zip_ref.namelist())
                self.logger.info(f"Extracting {total_files} files...")
                zip_ref.extractall(unzip_path)
            
            extracted_files = list(unzip_path.rglob('*'))
            self.logger.info(f"Extraction complete: {len(extracted_files)} items extracted")
            
            return unzip_path
            
        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid zip file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise

    def split_data(self, test_size: float = 0.20, seed: int = 123) -> None:
        """Split data into train and test sets with proper separation"""
        import random
        
        try:
            source_dir = self.config.unzip_dir / "Chest-CT-Scan-data"
            train_dir = self.config.unzip_dir / "train"
            test_dir = self.config.unzip_dir / "test"
            
            # Skip if already split
            if train_dir.exists() and test_dir.exists():
                train_count = len(list(train_dir.rglob('*.png')) + list(train_dir.rglob('*.jpg')))
                test_count = len(list(test_dir.rglob('*.png')) + list(test_dir.rglob('*.jpg')))
                if train_count > 0 and test_count > 0:
                    self.logger.info(f"Data already split: train={train_count}, test={test_count}")
                    return
            
            if not source_dir.exists():
                raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
            random.seed(seed)
            
            # Get class folders
            class_folders = [f for f in source_dir.iterdir() if f.is_dir()]
            
            self.logger.info(f"Splitting data: {len(class_folders)} classes, test_size={test_size}")
            
            total_train = 0
            total_test = 0
            
            for class_folder in class_folders:
                class_name = class_folder.name
                
                # Get all image files
                image_files = list(class_folder.glob('*.png')) + list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.jpeg'))
                random.shuffle(image_files)
                
                # Calculate split point
                split_idx = int(len(image_files) * (1 - test_size))
                train_files = image_files[:split_idx]
                test_files = image_files[split_idx:]
                
                # Create class directories
                train_class_dir = train_dir / class_name
                test_class_dir = test_dir / class_name
                train_class_dir.mkdir(parents=True, exist_ok=True)
                test_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy files
                for img in train_files:
                    shutil.copy2(img, train_class_dir / img.name)
                for img in test_files:
                    shutil.copy2(img, test_class_dir / img.name)
                
                total_train += len(train_files)
                total_test += len(test_files)
                
                self.logger.info(f"{class_name}: train={len(train_files)}, test={len(test_files)}")
            
            self.logger.info(f"Data split complete: train={total_train}, test={total_test}")
            
        except Exception as e:
            self.logger.error(f"Data split failed: {e}")
            raise


