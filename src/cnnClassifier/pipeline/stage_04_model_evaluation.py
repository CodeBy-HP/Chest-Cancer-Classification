import logging
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import Evaluation


STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:
    """Pipeline for model evaluation and MLflow tracking stage"""
    
    def __init__(self):
        pass

    def main(self):
        """Execute model evaluation pipeline"""
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
            