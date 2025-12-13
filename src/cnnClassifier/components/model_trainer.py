import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from typing import Tuple
from datetime import datetime
from sklearn.utils import class_weight
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    """Model training component with modern TensorFlow best practices"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self) -> tf.keras.Model:
        """Load compiled model from .keras file"""
        try:
            model_path = self.config.updated_base_model_path
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.logger.info(f"Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            self.logger.info("Model loaded successfully")
            self.logger.info(f"Total parameters: {self.model.count_params():,}")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def train_valid_generator(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create training and validation datasets using tf.data API"""
        try:
            image_size = tuple(self.config.params_image_size[:-1])
            batch_size = self.config.params_batch_size
            
            self.logger.info(f"Loading dataset from: {self.config.training_data}")
            self.logger.info(f"Image size: {image_size}, Batch size: {batch_size}")
            
            self.valid_generator = tf.keras.utils.image_dataset_from_directory(
                directory=str(self.config.training_data),
                validation_split=0.20,
                subset="validation",
                seed=123,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=False,
                label_mode='categorical'
            )
            
            self.train_generator = tf.keras.utils.image_dataset_from_directory(
                directory=str(self.config.training_data),
                validation_split=0.20,
                subset="training",
                seed=123,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=True,
                label_mode='categorical'
            )
            
            class_names = self.train_generator.class_names
            self.logger.info(f"Classes detected: {class_names}")
            
            normalization_layer = tf.keras.layers.Rescaling(1./255)
            self.train_generator = self.train_generator.map(
                lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            self.valid_generator = self.valid_generator.map(
                lambda x, y: (normalization_layer(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            if self.config.params_is_augmentation:
                self.logger.info("Data augmentation enabled")
                
                data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                    tf.keras.layers.RandomRotation(0.2),
                    tf.keras.layers.RandomZoom(0.2),
                    tf.keras.layers.RandomTranslation(0.2, 0.2),
                    tf.keras.layers.RandomContrast(0.2),
                ], name='augmentation')
                
                self.train_generator = self.train_generator.map(
                    lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            AUTOTUNE = tf.data.AUTOTUNE
            self.train_generator = self.train_generator.cache().prefetch(buffer_size=AUTOTUNE)
            self.valid_generator = self.valid_generator.cache().prefetch(buffer_size=AUTOTUNE)
            
            train_batches = tf.data.experimental.cardinality(self.train_generator).numpy()
            valid_batches = tf.data.experimental.cardinality(self.valid_generator).numpy()
            
            self.logger.info(f"Training batches: {train_batches}")
            self.logger.info(f"Validation batches: {valid_batches}")
            
            return self.train_generator, self.valid_generator
            
        except Exception as e:
            self.logger.error(f"Failed to create data generators: {e}")
            raise

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """Save model in .keras format"""
        try:
            if not str(path).endswith('.keras'):
                path = Path(str(path).replace('.h5', '.keras'))
            
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(path, save_format='keras')
            
            file_size = path.stat().st_size / (1024 * 1024)
            logging.info(f"Model saved: {path} ({file_size:.2f} MB)")
            
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            raise

    def train(self) -> tf.keras.callbacks.History:
        """Train model with callbacks and class balancing"""
        try:
            self.logger.info("Calculating class weights for imbalanced dataset...")
            
            class_labels = np.concatenate([y for x, y in self.train_generator], axis=0)
            class_labels = np.argmax(class_labels, axis=1)
            
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(class_labels),
                y=class_labels
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            self.logger.info(f"Class weights: {class_weight_dict}")
            
            callbacks = self._create_callbacks()
            
            self.logger.info("Starting model training...")
            
            history = self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                validation_data=self.valid_generator,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
            
            self.logger.info("Training completed successfully")
            return history
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _create_callbacks(self) -> list:
        """Create training callbacks for monitoring"""
        callbacks = []
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        checkpoint_path = self.config.root_dir / "best_model_checkpoint.keras"
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        log_dir = self.config.root_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        self.logger.info(f"Configured {len(callbacks)} callbacks")
        self.logger.info(f"TensorBoard logs: {log_dir}")
        
        return callbacks

