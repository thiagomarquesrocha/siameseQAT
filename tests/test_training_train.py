import pytest
import os
from jobs.data_pipeline import DataPipeline
from deep_learning.training.train import Train

class TestTrainingTrain:

    @pytest.fixture
    def eclipse_test_dataset(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        COLAB = ''
        PREPROCESSING = 'fake'
        pipeline = DataPipeline(dataset, domain, COLAB, PREPROCESSING)
        pipeline.run()
        return pipeline

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_train_siameseTA_model_then_successful(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseTA'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        train = Train(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, EPOCHS=2, BATCH_SIZE=1, BATCH_SIZE_TEST=1)

        model = train.get_model()
        assert model != None