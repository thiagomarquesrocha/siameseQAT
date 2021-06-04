import numpy as np
import pytest
from src.jobs.data_pipeline import DataPipeline
from src.deep_learning.training.training_preparation import TrainingPreparation

class TestTrainingPreparationAndData:

    @pytest.fixture(scope="class")
    def eclipse_test_dataset(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        COLAB = ''
        PREPROCESSING = 'fake'
        pipeline = DataPipeline(dataset, domain, COLAB, PREPROCESSING)
        pipeline.run()
        return pipeline

    def test_training_batch_generation_then_successful(self, eclipse_test_dataset):
        batch_size_test = 1 # the final batch will be (3= anchor, pos and neg)
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        TOKEN_END = 102
        MAX_SEQUENCE_LENGTH_T = 10
        MAX_SEQUENCE_LENGTH_D = 100
        train_preparation = TrainingPreparation(DIR, DOMAIN, 
                                        PREPROCESSING,
                                        MAX_SEQUENCE_LENGTH_T, 
                                        MAX_SEQUENCE_LENGTH_D,
                                        TOKEN_END)
        train_preparation.run()
        data = train_preparation.get_data()
        categorical_size = data.categorical_size
        buckets = data.buckets
        issues_by_buckets = data.issues_by_buckets
        bug_ids = data.bug_test_ids
        set_data = data.test_data
        bug_set = data.bug_set
        # we want a constant validation group to have a frame of reference for model performance
        batch_triplets_valid, valid_input_sample, _, _, _, valid_sim = train_preparation.batch_iterator(
                                                                            bug_set, buckets, 
                                                                            set_data,
                                                                            bug_ids,
                                                                            batch_size_test,
                                                                            issues_by_buckets)
        validation_sample = [valid_input_sample['title']['token'], valid_input_sample['title']['segment'], 
                   valid_input_sample['description']['token'], valid_input_sample['description']['segment'],
                   valid_input_sample['info'], valid_input_sample['topics'], valid_sim]
        expected_validation = 7
        expected_batch_size = (3,)
        expected_title_lenght = (3, MAX_SEQUENCE_LENGTH_T)
        expected_desc_lenght = (3, MAX_SEQUENCE_LENGTH_D)
        expected_categorical_lenght = (3, categorical_size)
        assert len(validation_sample) == expected_validation
        assert expected_title_lenght == valid_input_sample['title']['token'].shape
        assert expected_desc_lenght == valid_input_sample['description']['token'].shape
        assert expected_categorical_lenght == valid_input_sample['info'].shape
        assert valid_sim.shape == expected_batch_size

    def test_training_batch_generation_anchor_pos_neg_then_successful(self, eclipse_test_dataset):
        batch_size_test = 3 # the final batch will be (anchor, pos and neg)
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        TOKEN_END = 102
        MAX_SEQUENCE_LENGTH_T = 10
        MAX_SEQUENCE_LENGTH_D = 100
        train_preparation = TrainingPreparation(DIR, DOMAIN, 
                                        PREPROCESSING,
                                        MAX_SEQUENCE_LENGTH_T, 
                                        MAX_SEQUENCE_LENGTH_D,
                                        TOKEN_END)
        train_preparation.run()
        data = train_preparation.get_data()
        categorical_size = data.categorical_size
        buckets = data.buckets
        issues_by_buckets = data.issues_by_buckets
        bug_ids = data.bug_test_ids
        set_data = data.test_data
        bug_set = data.bug_set
        # we want a constant validation group to have a frame of reference for model performance
        batch_triplets_valid, _, input_anchor, input_pos, input_neg, valid_sim = train_preparation.batch_iterator(
                                                                            bug_set, buckets, 
                                                                            set_data,
                                                                            bug_ids,
                                                                            batch_size_test,
                                                                            issues_by_buckets)
        
        expected_batch_size = (batch_size_test * 3,)
        expected_title_lenght = (batch_size_test, MAX_SEQUENCE_LENGTH_T)
        expected_desc_lenght = (batch_size_test, MAX_SEQUENCE_LENGTH_D)
        expected_categorical_lenght = (batch_size_test, categorical_size)

        # Anchor
        assert expected_title_lenght == input_anchor['title']['token'].shape
        assert expected_desc_lenght == input_anchor['description']['token'].shape
        assert expected_categorical_lenght == input_anchor['info'].shape
        # Pos
        assert expected_title_lenght == input_pos['title']['token'].shape
        assert expected_desc_lenght == input_pos['description']['token'].shape
        assert expected_categorical_lenght == input_pos['info'].shape
        # Neg
        assert expected_title_lenght == input_neg['title']['token'].shape
        assert expected_desc_lenght == input_neg['description']['token'].shape
        assert expected_categorical_lenght == input_neg['info'].shape
        # Label
        assert valid_sim.shape == expected_batch_size

    def test_prepare_training_data_then_successful(self, eclipse_test_dataset):
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        TOKEN_END = 102
        MAX_SEQUENCE_LENGTH_T = 10
        MAX_SEQUENCE_LENGTH_D = 100
        train_preparation = TrainingPreparation(DIR, DOMAIN, 
                                        PREPROCESSING,
                                        MAX_SEQUENCE_LENGTH_T, 
                                        MAX_SEQUENCE_LENGTH_D,
                                        TOKEN_END)
        train_preparation.run()
        data = train_preparation.get_data()
        bug_id = np.random.choice(data.bug_ids, 1)[0]
        assert bug_id in data.bug_set 
        bug = data.bug_set[bug_id]
        assert 'bug_severity' in bug
        assert 'bug_status' in bug
        assert 'component' in bug
        assert 'creation_ts' in bug
        assert 'delta_ts' in bug
        assert 'description' in bug
        assert 'description_token' in bug
        assert 'dup_id' in bug
        assert 'issue_id' in bug
        assert 'priority' in bug
        assert 'product' in bug
        assert 'resolution' in bug
        assert 'textual_token' in bug
        assert 'title' in bug
        assert 'title_token' in bug
        assert 'version' in bug

    def test_prepare_training_data_then_successful(self, eclipse_test_dataset):
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'bert'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        TOKEN_END = 102
        MAX_SEQUENCE_LENGTH_T = 10
        MAX_SEQUENCE_LENGTH_D = 100
        train_preparation = TrainingPreparation(DIR, DOMAIN, 
                                        PREPROCESSING,
                                        MAX_SEQUENCE_LENGTH_T, 
                                        MAX_SEQUENCE_LENGTH_D,
                                        TOKEN_END)
        train_preparation.run()
        data = train_preparation.get_data()
        bug_id = np.random.choice(data.bug_ids, 1)[0]
        assert bug_id in data.bug_set 
        bug = data.bug_set[bug_id]
        assert 'bug_severity' in bug
        assert 'bug_status' in bug
        assert 'component' in bug
        assert 'creation_ts' in bug
        assert 'delta_ts' in bug
        assert 'description' in bug
        assert 'description_token' in bug
        assert 'dup_id' in bug
        assert 'issue_id' in bug
        assert 'priority' in bug
        assert 'product' in bug
        assert 'resolution' in bug
        assert 'textual_token' in bug
        assert 'title' in bug
        assert 'title_token' in bug
        assert 'version' in bug