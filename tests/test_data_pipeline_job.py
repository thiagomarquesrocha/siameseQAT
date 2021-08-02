import pytest
import os
from src.jobs.data_pipeline import DataPipeline

class TestDataPipeline:

    def test_data_pipeline_output_expected_then_successful(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        PREPROCESSING = 'bert'
        pipeline = DataPipeline(dataset, domain, PREPROCESSING)
        pipeline.setup()
        expected_dir_output =  os.path.join("data", "processed", "eclipse_test", "bert")
        assert pipeline.DIR_OUTPUT == expected_dir_output

    def test_data_pipeline_workflow_then_successful(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        PREPROCESSING = 'fake'
        pipeline = DataPipeline(dataset, domain, PREPROCESSING)
        pipeline.run()
        assert True
        