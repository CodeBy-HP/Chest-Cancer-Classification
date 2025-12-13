import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path


class PredictionPipeline:
    _model = None
    _model_path = Path("model/model.keras")

    @classmethod
    def get_model(cls):
        """Load model once and cache it"""
        if cls._model is None:
            cls._model = load_model(cls._model_path)
        return cls._model

    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        """
        Predict whether a chest CT scan shows Adenocarcinoma or Normal tissue.
        
        Returns:
            list: Dictionary containing the prediction result
        """
        # Get cached model
        model = self.get_model()

        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make prediction
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'
            
        return [{"image": prediction}]