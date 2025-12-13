import logging
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training


STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    """Pipeline for model training stage"""
    
    def __init__(self):
        pass

    def main(self):
        """Execute model training pipeline"""
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
        
