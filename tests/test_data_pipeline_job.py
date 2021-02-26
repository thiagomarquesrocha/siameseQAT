import pytest
import os
from jobs.data_pipeline import DataPipeline

class TestDataPipeline:

    def test_data_pipeline_output_expected_then_successful(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        COLAB = ''
        PREPROCESSING = 'bert'
        pipeline = DataPipeline(dataset, domain, COLAB, PREPROCESSING)
        pipeline.setup()
        expected_dir_output =  os.path.join("data", "processed", "eclipse_test", "bert")
        assert pipeline.DIR_OUTPUT == expected_dir_output

    # TODO: How to mock bert pretrained
    # def test_data_pipeline_workflow_then_successful(self):
    #     dataset = 'eclipse_test'
    #     domain = 'eclipse_test'
    #     COLAB = ''
    #     PREPROCESSING = 'bert'
    #     pipeline = DataPipeline(dataset, domain, COLAB, PREPROCESSING)
    #     pipeline.run()
    #     assert True
        