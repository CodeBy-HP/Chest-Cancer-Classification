import logging
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")
except Exception as e:
    logging.exception(e)
    raise e


# Stage 2: Prepare Base Model
STAGE_NAME = "Prepare Base Model Stage"
try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    pipeline = PrepareBaseModelTrainingPipeline()
    pipeline.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")
except Exception as e:
    logging.exception(e)
    raise e


# Stage 3: Model Training
STAGE_NAME = "Model Training Stage"
try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    pipeline = ModelTrainingPipeline()
    pipeline.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")
except Exception as e:
    logging.exception(e)
    raise e


# Stage 4: Model Evaluation
STAGE_NAME = "Model Evaluation Stage"
try:
    logging.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    pipeline = EvaluationPipeline()
    pipeline.main()
    logging.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")
except Exception as e:
    logging.exception(e)
    raise e
