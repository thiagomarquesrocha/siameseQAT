import pytest
import os
from jobs.preprocessor import Preprocessor

class TestPreprocessing:

    def test_data_transformation_path_expected(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        COLAB = ''
        PREPROCESSING = 'bert'
        preprocessor = Preprocessor(dataset, domain, COLAB, PREPROCESSING)
        preprocessor.setup()
        expected_dir_output =  os.path.join("data", "processed", "eclipse_test", "bert")
        assert preprocessor.DIR_OUTPUT == expected_dir_output

    def test_workflow(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        COLAB = ''
        PREPROCESSING = 'bert'
        preprocessor = Preprocessor(dataset, domain, COLAB, PREPROCESSING)
        preprocessor.run()
        assert True
        