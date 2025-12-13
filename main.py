import logging
from dotenv import load_dotenv
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

# Load environment variables
load_dotenv()


# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion"
try:
    logging.info(f"Starting: {STAGE_NAME}")
    DataIngestionTrainingPipeline().main()
    logging.info(f"Completed: {STAGE_NAME}")
except Exception as e:
    logging.exception(e)
    raise e


# Stage 2: Prepare Base Model
STAGE_NAME = "Prepare Base Model"
try:
    logging.info(f"Starting: {STAGE_NAME}")
    PrepareBaseModelTrainingPipeline().main()
    logging.info(f"Completed: {STAGE_NAME}")
except Exception as e:
    logging.exception(e)
    raise e


# Stage 3: Model Training
STAGE_NAME = "Model Training"
try:
    logging.info(f"Starting: {STAGE_NAME}")
    ModelTrainingPipeline().main()
    logging.info(f"Completed: {STAGE_NAME}")
except Exception as e:
    logging.exception(e)
    raise e


# Stage 4: Model Evaluation
STAGE_NAME = "Model Evaluation"
try:
    logging.info(f"Starting: {STAGE_NAME}")
    EvaluationPipeline().main()
    logging.info(f"Completed: {STAGE_NAME}")
except Exception as e:
    logging.exception(e)
    raise e
