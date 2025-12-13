import os
import tensorflow as tf
import logging
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
import mlflow
import mlflow.keras
import dagshub
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    """Model evaluation component with MLflow tracking"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.valid_generator = None
        self.score = None

    def _valid_generator(self) -> tf.data.Dataset:
        """Create validation dataset using tf.data API"""
        try:
            image_size = tuple(self.config.params_image_size[:-1])
            batch_size = self.config.params_batch_size
            
            self.logger.info(f"Loading validation data from: {self.config.training_data}")
            
            self.valid_generator = tf.keras.utils.image_dataset_from_directory(
                directory=str(self.config.training_data),
                validation_split=0.30,
                subset="validation",
                seed=123,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=False,
                label_mode='categorical'
            )
            
            class_names = self.valid_generator.class_names
            self.logger.info(f"Classes detected: {class_names}")
            
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            self.valid_generator = self.valid_generator.map(
                lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            self.valid_generator = self.valid_generator.cache().prefetch(
                buffer_size=tf.data.AUTOTUNE
            )
            
            self.logger.info("Validation data loaded and preprocessed")
            
            return self.valid_generator
            
        except Exception as e:
            self.logger.error(f"Failed to create validation generator: {e}")
            raise

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Load model from .keras file"""
        try:
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            
            model = tf.keras.models.load_model(path)
            logging.info(f"Model loaded from: {path}")
            
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def evaluation(self) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        try:
            self.logger.info("Starting model evaluation...")
            
            self.model = self.load_model(self.config.path_of_model)
            self._valid_generator()
            
            self.logger.info("Evaluating model on validation data...")
            results = self.model.evaluate(self.valid_generator, verbose=1)
            
            metric_names = self.model.metrics_names
            self.score = dict(zip(metric_names, results))
            
            self.logger.info("Evaluation completed")
            self.logger.info(f"Metrics: {self.score}")
            
            self.save_score()
            
            return self.score
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    def save_score(self) -> None:
        """Save evaluation scores to JSON file"""
        try:
            save_json(path=Path("scores.json"), data=self.score)
            self.logger.info("Scores saved to scores.json")
        except Exception as e:
            self.logger.error(f"Failed to save scores: {e}")
            raise

    def log_into_mlflow(self) -> None:
        """Log experiment to MLflow with DagHub integration"""
        try:
            dagshub_owner = os.getenv('DAGSHUB_REPO_OWNER', 'CodeBy-HP')
            dagshub_repo = os.getenv('DAGSHUB_REPO_NAME', 'chest-cancer-classification')
            
            dagshub.init(
                repo_owner=dagshub_owner,
                repo_name=dagshub_repo,
                mlflow=True
            )
            self.logger.info(f"DagHub initialized: {dagshub_owner}/{dagshub_repo}")
            
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            self.logger.info(f"MLflow tracking URI set: {self.config.mlflow_uri}")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run():
                self.logger.info("MLflow run started")
                
                self.logger.info("Logging parameters...")
                mlflow.log_params(self.config.all_params)
                
                self.logger.info("Logging metrics...")
                mlflow.log_metrics(self.score)
                
                self.logger.info("Logging model...")
                
                if tracking_url_type_store != "file":
                    self.logger.info("Remote tracking detected - registering model...")
                    
                    mlflow.keras.log_model(
                        self.model,
                        artifact_path="model",
                        registered_model_name="EfficientNetB0_ChestCancer"
                    )
                    
                    self.logger.info("Model registered in MLflow Model Registry")
                else:
                    self.logger.info("Local tracking detected - logging model...")
                    
                    mlflow.keras.log_model(
                        self.model,
                        artifact_path="model"
                    )
                    
                    self.logger.info("Model logged to MLflow (local)")
                
                run_id = mlflow.active_run().info.run_id
                self.logger.info(f"MLflow run completed: {run_id}")
                
        except Exception as e:
            self.logger.error(f"MLflow logging failed: {e}")
            self.logger.error("Check your DagHub credentials in .env file")
            raise
