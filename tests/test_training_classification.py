import pytest
import os
import keras
from jobs.data_pipeline import DataPipeline
from utils.keras_utils import KerasUtils
from deep_learning.training.train_config import TrainConfig
from deep_learning.training.train_retrieval import TrainRetrieval
from deep_learning.training.train_classification import TrainClassification

class TestTrainingClassification:

    @pytest.fixture(scope="class")
    def eclipse_test_dataset(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        COLAB = ''
        PREPROCESSING = 'fake'
        pipeline = DataPipeline(dataset, domain, COLAB, PREPROCESSING, VALIDATION_SPLIT=0.5)
        pipeline.run()
        return pipeline

    @pytest.fixture(scope="class")
    def retrieval_model(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseTA'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        EPOCHS = 1
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, EPOCHS=EPOCHS, BATCH_SIZE=1, BATCH_SIZE_TEST=1).run()

        model = train.get_model()
        TRAINED_OUTPUT = TrainConfig.MODEL_NAME.format(PREPROCESSING, MODEL_NAME, EPOCHS, DOMAIN)
        # Save model in .h5 format
        # KerasUtils.save_model(model, TRAINED_OUTPUT)
        KerasUtils.save_weights(model, TRAINED_OUTPUT)
        # Clear keras models
        keras.backend.clear_session()
        del model
        return EPOCHS

    def test_classification_model_preload_then_successful(self, eclipse_test_dataset, retrieval_model):
        # Retrieval
        EPOCHS_TRAINED = retrieval_model
        MODEL_NAME = 'SiameseTA'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        retrieval = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, EPOCHS=EPOCHS_TRAINED, BATCH_SIZE=1, BATCH_SIZE_TEST=1).build()

        retrieval_preload = retrieval.get_model()
        
        # Classification
        MODEL_NAME = 'SiameseTA'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        PRETRAINED_MODEL = os.path.join(TrainConfig.OUTPUT_MODELS, TrainConfig.MODEL_NAME.format(PREPROCESSING, MODEL_NAME, EPOCHS_TRAINED, DOMAIN))
        train = TrainClassification(retrieval_preload, MODEL_NAME, PRETRAINED_MODEL, 
                    DIR, DOMAIN, PREPROCESSING, EPOCHS=2, 
                    BATCH_SIZE=1, BATCH_SIZE_TEST=1)
        train.pre_load_model()
        expected_categorical_size = 9
        expected_title_size = 1
        expected_desc_size = 1
        expected_topic_size = train.INVALID_INPUT_SIZE
        assert train.TITLE_SIZE == expected_title_size
        assert train.DESC_SIZE == expected_desc_size
        assert train.CATEGORICAL_SIZE == expected_categorical_size
        assert train.TOPIC_SIZE == expected_topic_size