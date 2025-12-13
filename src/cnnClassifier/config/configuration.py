import os
import logging
from pathlib import Path
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)


class ConfigurationManager:
    """Configuration manager for loading and creating component configurations"""
    
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ):
        """Initialize configuration manager"""
        try:
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)
            
            create_directories([self.config.artifacts_root])
            logging.info("Configuration loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration"""
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        source_url = os.getenv('DATASET_URL', config.source_URL)
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=source_url,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        
        logging.info("Data ingestion config created")
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """Get base model preparation configuration"""
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        
        logging.info("Base model config created")
        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration"""
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        
        training_data = Path(self.config.data_ingestion.unzip_dir) / "Chest-CT-Scan-data"
        
        create_directories([Path(training.root_dir)])
        
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=training_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE
        )
        
        logging.info("Training config created")
        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration with environment variable overrides"""
        
        mlflow_uri = os.getenv(
            'MLFLOW_TRACKING_URI',
            f"https://dagshub.com/{os.getenv('DAGSHUB_REPO_OWNER', 'CodeBy-HP')}/"
            f"{os.getenv('DAGSHUB_REPO_NAME', 'chest-cancer-classification')}.mlflow"
        )
        
        eval_config = EvaluationConfig(
            path_of_model=Path("artifacts/training/model.keras"),
            training_data=Path("artifacts/data_ingestion/Chest-CT-Scan-data"),
            mlflow_uri=mlflow_uri,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        
        logging.info("Evaluation config created")
        logging.info(f"MLflow URI: {mlflow_uri}")
        
        return eval_config

      