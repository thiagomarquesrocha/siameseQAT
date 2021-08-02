import pytest
import os
from src.jobs.data_pipeline import DataPipeline
from src.deep_learning.training.train_retrieval import TrainRetrieval

class TestTrainingTrain:

    @pytest.fixture(scope="class")
    def eclipse_test_dataset(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        PREPROCESSING = 'fake'
        pipeline = DataPipeline(dataset, domain, PREPROCESSING, VALIDATION_SPLIT=0.5)
        pipeline.run()
        return pipeline

    def test_train_prepare_dataset_then_successful(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseQA-A'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, EPOCHS=2, BATCH_SIZE=1, BATCH_SIZE_TEST=1)
        train.prepare_data()
        train.prepare_validation_data()
        sample = train.validation_sample
        expected_batch_size = 1 * 3 # ancho, pos and neg
        expected_categorical_size = (expected_batch_size, train.MAX_LENGTH_CATEGORICAL)
        expected_title_size = (expected_batch_size, train.MAX_SEQUENCE_LENGTH_T)
        expected_desc_size = (expected_batch_size, train.MAX_SEQUENCE_LENGTH_D)
        expected_batch_size = (expected_batch_size, )
        assert sample[0].shape == expected_categorical_size
        assert sample[1].shape == expected_desc_size
        assert sample[2].shape == expected_desc_size
        assert sample[3].shape == expected_title_size
        assert sample[4].shape == expected_title_size
        assert sample[5].shape == expected_batch_size

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_train_siameseQAT_A_model_then_successful(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseQAT-A'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    TOPIC_LENGTH=0, # TODO: Topic feature missing on pipeline
                    BERT_LAYERS=1, EPOCHS=2, BATCH_SIZE=2, BATCH_SIZE_TEST=2).run()

        model = train.get_model()
        assert model != None

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_train_siameseQAT_W_model_then_successful(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseQAT-W'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    TOPIC_LENGTH=0, # TODO: Topic feature missing on pipeline
                    BERT_LAYERS=1, EPOCHS=2, BATCH_SIZE=2, BATCH_SIZE_TEST=2).run()

        model = train.get_model()
        assert model != None

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_train_siameseQA_W_model_then_successful(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseQA-W'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, EPOCHS=2, BATCH_SIZE=2, BATCH_SIZE_TEST=2).run()

        model = train.get_model()
        assert model != None

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_train_siameseQA_A_model_then_successful(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseQA-A'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, EPOCHS=1, BATCH_SIZE=2, BATCH_SIZE_TEST=2).run()

        model = train.get_model()
        assert model != None

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_train_siameseTAT_model_then_successful(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseTAT'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        # TODO: Topic disabled because of missing topic feature on pipeline
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    TOPIC_LENGTH=0, BERT_LAYERS=1, 
                    EPOCHS=2, BATCH_SIZE=1, BATCH_SIZE_TEST=1).run()

        model = train.get_model()
        assert model != None

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_train_siameseTA_model_then_successful(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseTA'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, EPOCHS=2, BATCH_SIZE=1, BATCH_SIZE_TEST=1).run()

        model = train.get_model()
        assert model != None