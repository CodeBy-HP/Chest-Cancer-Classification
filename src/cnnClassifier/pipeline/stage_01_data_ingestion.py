import logging
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion


STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    """Pipeline for data ingestion stage"""
    
    def __init__(self):
        pass

    def main(self):
        """Execute data ingestion pipeline"""
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        data_ingestion.split_data(test_size=0.20, seed=123)


if __name__ == '__main__':
    try:
        logging.info(f"Starting: {STAGE_NAME}")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f"Completed: {STAGE_NAME}")
    except Exception as e:
        logging.exception(e)
        raise e

