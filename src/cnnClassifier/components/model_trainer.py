import os
import tensorflow as tf
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

# Ensure TensorFlow eager execution is enabled
tf.config.run_functions_eagerly(True)



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        # Load model without compiling to avoid optimizer state issues
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )
        # Create a fresh optimizer instance with the same learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    def train_valid_generator(self):
        # Using tf.keras.utils.image_dataset_from_directory (modern API)
        image_size = tuple(self.config.params_image_size[:-1])
        batch_size = self.config.params_batch_size
        
        # Validation dataset
        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Normalize validation data
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.valid_generator = self.valid_generator.map(lambda x, y: (normalization_layer(x), y))
        
        # Training dataset
        self.train_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Normalize training data
        self.train_generator = self.train_generator.map(lambda x, y: (normalization_layer(x), y))
        
        # Apply data augmentation if enabled
        if self.config.params_is_augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomTranslation(0.2, 0.2),
            ])
            self.train_generator = self.train_generator.map(
                lambda x, y: (data_augmentation(x, training=True), y)
            )
        
        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_generator = self.train_generator.prefetch(buffer_size=AUTOTUNE)
        self.valid_generator = self.valid_generator.prefetch(buffer_size=AUTOTUNE)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Use native Keras format to avoid pickle errors with eager execution
        keras_path = str(path).replace('.h5', '.keras')
        model.save(keras_path)



    
    def train(self):
        # Calculate class weights from directory structure to handle imbalanced dataset
        # This gives more importance to minority class during training
        from sklearn.utils import class_weight
        import numpy as np
        import os
        
        # Count samples per class from directory structure
        class_counts = []
        class_names = sorted(os.listdir(self.config.training_data))
        
        for class_name in class_names:
            class_dir = os.path.join(self.config.training_data, class_name)
            if os.path.isdir(class_dir):
                num_samples = len([f for f in os.listdir(class_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts.append(num_samples)
        
        # Calculate balanced class weights
        class_counts = np.array(class_counts)
        total_samples = np.sum(class_counts)
        class_weights = total_samples / (len(class_counts) * class_counts)
        # Convert to Python native types to avoid TensorFlow eager execution issues
        class_weight_dict = {int(k): float(v) for k, v in enumerate(class_weights)}
        
        print(f"Class distribution: {dict(zip(class_names, class_counts.tolist()))}")
        print(f"Class weights applied: {class_weight_dict}")
        print("This helps balance the dataset by giving more weight to minority classes")
        
        # Train with class weights for balanced learning
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            class_weight=class_weight_dict
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

