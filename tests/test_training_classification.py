import pytest
import os
from tensorflow import keras
from src.jobs.data_pipeline import DataPipeline
from src.utils.keras_utils import KerasUtils
from src.deep_learning.training.train_config import TrainConfig
from src.deep_learning.training.train_retrieval import TrainRetrieval
from src.deep_learning.training.train_classification import TrainClassification

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
    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
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

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_classification_prepare_data_then_successful(self, eclipse_test_dataset, retrieval_model):
        # Retrieval
        EPOCHS_TRAINED = retrieval_model
        MODEL_NAME = 'SiameseTA'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        retrieval = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, 
                    MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, 
                    EPOCHS=EPOCHS_TRAINED, 
                    BATCH_SIZE=1, 
                    BATCH_SIZE_TEST=1).build()

        retrieval_preload = retrieval.get_model()
        
        # Classification
        MODEL_NAME = 'SiameseTA'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        PRETRAINED_MODEL = os.path.join(TrainConfig.OUTPUT_MODELS, TrainConfig.MODEL_NAME.format(PREPROCESSING, MODEL_NAME, EPOCHS_TRAINED, DOMAIN))
        train = TrainClassification(retrieval_preload, 
                    MODEL_NAME, 
                    PRETRAINED_MODEL, 
                    DIR, DOMAIN, PREPROCESSING, 
                    EPOCHS=1, 
                    BATCH_SIZE=4, 
                    BATCH_SIZE_TEST=4)

        train.pre_load_model()
        train.prepare_data()
        train.prepare_validation_data()
        sample = train.validation_sample
        assert 'title_token_0' in sample
        assert 'title_token_1' in sample
        assert 'title_segment_0' in sample
        assert 'title_segment_1' in sample
        assert 'desc_token_0' in sample
        assert 'desc_token_1' in sample
        assert 'desc_segment_0' in sample
        assert 'desc_segment_1' in sample
        

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
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
        train.run()
        # Check preload
        expected_categorical_size = 9
        expected_title_size = 1
        expected_desc_size = 1
        expected_topic_size = train.INVALID_INPUT_SIZE
        assert train.TITLE_SIZE == expected_title_size
        assert train.DESC_SIZE == expected_desc_size
        assert train.CATEGORICAL_SIZE == expected_categorical_size
        assert train.TOPIC_SIZE == expected_topic_size
        # Check model
        model = train.get_model()
        assert model.get_layer('concatenated_bug_embed') != None
        assert model.get_layer('bugs') != None
        assert model.get_layer('softmax') != None