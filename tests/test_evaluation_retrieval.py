import pytest
import os
from jobs.data_pipeline import DataPipeline
from deep_learning.training.train_retrieval import TrainRetrieval
from deep_learning.model.fake_model import FakeModel
from utils.util import Util
from evaluation.retrieval import Retrieval
from evaluation.recall import Recall

class TestEvaluationRetrieval:

    @pytest.fixture
    def eclipse_test_dataset(self):
        dataset = 'eclipse_test'
        domain = 'eclipse_test'
        COLAB = ''
        PREPROCESSING = 'fake'
        pipeline = DataPipeline(dataset, domain, COLAB, PREPROCESSING, VALIDATION_SPLIT=0.5)
        pipeline.run()
        return pipeline

    @pytest.fixture
    def prepare_dataset(self, eclipse_test_dataset):
        MODEL_NAME = 'SiameseQAT-A'
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = 'eclipse_test'
        PREPROCESSING = 'fake'
        train = TrainRetrieval(MODEL_NAME, DIR, DOMAIN, PREPROCESSING, 
                    MAX_SEQUENCE_LENGTH_T=1, MAX_SEQUENCE_LENGTH_D=1,
                    BERT_LAYERS=1, EPOCHS=2, BATCH_SIZE=1, BATCH_SIZE_TEST=1)
        train.prepare_data()
        return train

    def test_retrieval_evaluation_then_successful(self, prepare_dataset):
        DOMAIN = prepare_dataset.DOMAIN
        data = prepare_dataset.train_preparation.get_data()
        # Categorical info
        info_dict = data.info_dict
        verbose = True
        retrieval = Retrieval(DOMAIN, info_dict, verbose)
        # Evaluation
        buckets = data.buckets
        test = data.test_data
        bug_set = data.bug_set
        issues_by_buckets = data.issues_by_buckets
        bug_ids = data.bug_test_ids
        model = FakeModel()
        method = 'bert'
        only_buckets = False # Include all dups
        
        recall_at_25, exported_rank, _ = retrieval.evaluate(buckets, 
                                                          test, 
                                                          bug_set, 
                                                          model, 
                                                          issues_by_buckets, 
                                                          bug_ids, 
                                                          method=method, 
                                                          only_buckets=only_buckets)
        assert len(exported_rank) > 0
        assert recall_at_25 > 0.0

    def test_save_retrieval_evaluation_then_successful(self, eclipse_test_dataset, prepare_dataset):
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = prepare_dataset.DOMAIN
        METHOD = 'fake_model'
        EXPORT_RANK_PATH = os.path.join(DIR, 'exported_rank_{}.txt'.format(METHOD))
        data = prepare_dataset.train_preparation.get_data()
        # Categorical info
        info_dict = data.info_dict
        verbose = False
        retrieval = Retrieval(DOMAIN, info_dict, verbose)
        # Evaluation
        buckets = data.buckets
        test = data.test_data
        bug_set = data.bug_set
        issues_by_buckets = data.issues_by_buckets
        bug_ids = data.bug_test_ids
        model = FakeModel()
        method = 'bert'
        only_buckets = False # Include all dups
        
        recall_at_25, exported_rank, _ = retrieval.evaluate(buckets, 
                                                          test, 
                                                          bug_set, 
                                                          model, 
                                                          issues_by_buckets, 
                                                          bug_ids, 
                                                          method=method, 
                                                          only_buckets=only_buckets)
        Util.save_rank(EXPORT_RANK_PATH, exported_rank)
        recall = Recall(verbose)
        report = recall.evaluate(EXPORT_RANK_PATH)
        assert '0 - recall_at_1' in report
        assert '1 - recall_at_5' in report
        assert '2 - recall_at_10' in report
        assert '3 - recall_at_15' in report
        assert '4 - recall_at_20' in report
        assert '5 - recall_at_25' in report

    def test_format_retrieval_evaluation_then_successful(self, eclipse_test_dataset, prepare_dataset):
        DIR = eclipse_test_dataset.DIR_OUTPUT
        DOMAIN = prepare_dataset.DOMAIN
        METHOD = 'fake_model'
        EXPORT_RANK_PATH = os.path.join(DIR, 'exported_rank_{}.txt'.format(METHOD))
        data = prepare_dataset.train_preparation.get_data()
        # Categorical info
        info_dict = data.info_dict
        verbose = False
        retrieval = Retrieval(DOMAIN, info_dict, verbose)
        # Evaluation
        buckets = data.buckets
        test = data.test_data
        bug_set = data.bug_set
        issues_by_buckets = data.issues_by_buckets
        bug_ids = data.bug_test_ids
        model = FakeModel()
        method = 'bert'
        only_buckets = False # Include all dups
        
        _, exported_rank, _ = retrieval.evaluate(buckets, 
                                                          test, 
                                                          bug_set, 
                                                          model, 
                                                          issues_by_buckets, 
                                                          bug_ids, 
                                                          method=method, 
                                                          only_buckets=only_buckets)

        assert len(exported_rank) > 0
        first_pos = exported_rank[0]
        query, dups = first_pos.split('|')
        rank = dups.split(',')
        expected_query_elements = 2
        expected_dup_elements = 2
        assert len(query.split(':')) == expected_query_elements
        assert len(rank) > 0
        assert len(rank[0].split(":")) == expected_dup_elements

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_retrieval_evaluation_siameseTAT_then_successful(self, eclipse_test_dataset):
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
        encoder = train.get_bug_encoder()

        EXPORT_RANK_PATH = os.path.join(DIR, 'exported_rank_{}.txt'.format(MODEL_NAME))
        data = train.train_preparation.get_data()
        # Categorical info
        info_dict = data.info_dict
        verbose = True
        retrieval = Retrieval(DOMAIN, info_dict, verbose)
        # Evaluation
        buckets = data.buckets
        test = data.test_data
        bug_set = data.bug_set
        issues_by_buckets = data.issues_by_buckets
        bug_ids = data.bug_test_ids
        only_buckets = False # Include all dups
        
        METHOD = 'bert'
        recall_at_25, exported_rank, _ = retrieval.evaluate(buckets, 
                                                          test, 
                                                          bug_set, 
                                                          encoder, 
                                                          issues_by_buckets, 
                                                          bug_ids, 
                                                          method=METHOD, 
                                                          only_buckets=only_buckets)
        assert len(exported_rank) > 0
        assert recall_at_25 > 0.0
