import tensorflow as tf
import logging
from pathlib import Path
from typing import Optional
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """Base model preparation component for transfer learning"""
    
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.full_model = None

    def get_base_model(self) -> tf.keras.Model:
        """Load EfficientNetB0 pre-trained model"""
        try:
            self.logger.info("Loading EfficientNetB0 base model...")
            
            self.model = tf.keras.applications.EfficientNetB0(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
            
            self.logger.info(f"Base model loaded: {self.model.name}")
            self.logger.info("Base model loaded")
            
            self.save_model(path=self.config.base_model_path, model=self.model)
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool,
        freeze_till: Optional[int],
        learning_rate: float
    ) -> tf.keras.Model:
        """Prepare full model with custom classification head"""
        
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
            logging.info("Base model frozen")
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
            logging.info(f"Training last {freeze_till} layers")
        else:
            logging.info("All layers trainable")
        
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(model.output)
        x = tf.keras.layers.BatchNormalization(name='bn_head')(x)
        x = tf.keras.layers.Dropout(0.2, name='dropout_head')(x)
        
        predictions = tf.keras.layers.Dense(
            units=classes,
            activation='softmax',
            name='output',
            dtype='float32'
        )(x)
        
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=predictions,
            name='EfficientNetB0_ChestCancer'
        )
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        
        full_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        full_model.summary()
        
        trainable_params = sum([tf.size(w).numpy() for w in full_model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in full_model.non_trainable_weights])
        
        logging.info(f"Params — trainable: {trainable_params:,}, frozen: {non_trainable_params:,}")
        
        return full_model

    def update_base_model(self) -> tf.keras.Model:
        """Update base model with classification head"""
        try:
            self.logger.info("Building full model with classification head...")
            
            self.full_model = self._prepare_full_model(
                model=self.model,
                classes=self.config.params_classes,
                freeze_all=True,
                freeze_till=None,
                learning_rate=self.config.params_learning_rate
            )
            
            self.logger.info("Full model built successfully")
            
            self.save_model(
                path=self.config.updated_base_model_path,
                model=self.full_model
            )
            
            return self.full_model
            
        except Exception as e:
            self.logger.error(f"Failed to update base model: {e}")
            raise

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """Save model in .keras format"""
        try:
            if not str(path).endswith('.keras'):
                path = Path(str(path).replace('.h5', '.keras'))
                logging.warning(f"Changed extension to .keras: {path}")
            
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(path)  # Removed save_format to suppress warning
            
            file_size = path.stat().st_size / (1024 * 1024)
            logging.info(f"Model saved: {path} ({file_size:.2f} MB)")
            
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            raise


