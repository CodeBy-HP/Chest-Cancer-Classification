import logging
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel


STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    """Pipeline for base model preparation stage"""
    
    def __init__(self):
        pass

    def main(self):
        """Execute base model preparation pipeline"""
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
