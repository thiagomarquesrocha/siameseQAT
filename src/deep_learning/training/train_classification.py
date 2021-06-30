import logging
import numpy as np
from src.utils.keras_utils import KerasUtils
from src.deep_learning.model.compile_model import compile_model
from src.deep_learning.model.siameseQAT_classifier import SiameseQATClassifier
from src.deep_learning.training.training_preparation import TrainingPreparation
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger('TrainClassification')

class TrainClassification:

    INVALID_INPUT_SIZE = 0

    def __init__(self, model, MODEL_NAME, 
                PRETRAINED_MODEL, DIR, DOMAIN, PREPROCESSING, 
                EPOCHS=10, BATCH_SIZE=64, BATCH_SIZE_TEST=128):
        self.model = model
        self.MODEL_NAME = MODEL_NAME
        self.PRETRAINED_MODEL = PRETRAINED_MODEL
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        # Test batch is composed by an anchor, pos and neg
        self.BATCH_SIZE_TEST = BATCH_SIZE_TEST
        self.DOMAIN = DOMAIN # eclipse, netbeans, openoffice
        self.PREPROCESSING = PREPROCESSING # bert, baseline or fake
        self.DIR = DIR
        self.TOKEN_END = 102 # TODO: Read from BERT pretrained

    def run(self):
        self.pre_load_model()
        self.prepare_data()
        self.prepare_validation_data()
        self.create_model()
        self.train_model()
        return self

    def train_model(self):
        pass

    def create_model(self):
        self.model = SiameseQATClassifier(self.model, 
                            title_size=self.TITLE_SIZE, 
                            desc_size=self.DESC_SIZE, 
                            categorical_size=self.CATEGORICAL_SIZE, 
                            topic_size=self.TOPIC_SIZE)

        # Compile model
        self.model = compile_model(self.model)

    def get_model(self):
        return self.model

    def prepare_data(self):
        self.train_preparation = TrainingPreparation(self.DIR, 
                                                    self.DOMAIN, 
                                                    self.PREPROCESSING,
                                                    self.TITLE_SIZE, 
                                                    self.DESC_SIZE,
                                                    self.TOKEN_END)
        self.train_preparation.run()

    def pre_load_model(self):
        #self.model = KerasUtils.load_model(self.PRETRAINED_MODEL, self.MODEL_NAME)
        KerasUtils.load_weights(self.PRETRAINED_MODEL, self.model)
        self.CATEGORICAL_SIZE = self.get_input_size('categorical')
        self.TITLE_SIZE = self.get_input_size('title_token')
        self.DESC_SIZE = self.get_input_size('desc_token')
        self.TOPIC_SIZE = self.get_input_size('topic')

    def get_input_size(self, input_name):
        try:
            return self.model.get_layer(input_name).output_shape[1]
        except:
            return self.INVALID_INPUT_SIZE

    def batch_siamese(self):
        while True:
            _, validation_sample, valid_sim = self.generate_batch_data(self.BATCH_SIZE, 'train')
            yield (validation_sample, valid_sim) 

    def prepare_validation_data(self):
        _, self.validation_sample, self.valid_sim = self.generate_batch_data(self.BATCH_SIZE_TEST, 'test')

    def generate_batch_data(self, batch_size_test, mode='test'):
        data = self.train_preparation.get_data()
        categorical_size = data.categorical_size
        buckets = data.buckets
        issues_by_buckets = data.issues_by_buckets
        if mode == 'test':
            bug_ids = data.bug_test_ids
            set_data = data.test_data
            bug_set = data.bug_set
        else: # train
            bug_ids = data.bug_train_ids
            set_data = data.train_data
            bug_set = data.bug_set
        # we want a constant validation group to have a frame of reference for model performance
        batch_triplets_valid, _, input_sample, input_pos, input_neg, _ = self.train_preparation.batch_iterator(
                                                                            bug_set, buckets, 
                                                                            set_data,
                                                                            bug_ids,
                                                                            batch_size_test,
                                                                            issues_by_buckets)
        validation_sample = {}
        # Add Categorical
        if self.CATEGORICAL_SIZE > 0:
            info_a = np.concatenate([input_sample['info'], input_sample['info']])
            info_b = np.concatenate([input_pos['info'], input_neg['info']])
            validation_sample['categorical_0'] = info_a
            validation_sample['categorical_1'] = info_b
        # Add Desc
        if self.DESC_SIZE > 0:
            desc_a = np.concatenate([input_sample['description']['token'], input_sample['description']['token']])
            desc_a_seg = np.concatenate([input_sample['description']['segment'], input_sample['description']['segment']])
            desc_b = np.concatenate([input_pos['description']['token'], input_neg['description']['token']])
            desc_b_seg = np.concatenate([input_pos['description']['segment'], input_neg['description']['segment']])
            validation_sample['desc_token_0'] = desc_a
            validation_sample['desc_segment_0'] = desc_a_seg
            validation_sample['desc_token_1'] = desc_b
            validation_sample['desc_segment_1'] = desc_b_seg
        # Add Title
        if self.TITLE_SIZE > 0:
            title_a = np.concatenate([input_sample['title']['token'], input_sample['title']['token']])
            title_a_seg = np.concatenate([input_sample['title']['segment'], input_sample['title']['segment']])
            title_b = np.concatenate([input_pos['title']['token'], input_neg['title']['token']])
            title_b_seg = np.concatenate([input_pos['title']['segment'], input_neg['title']['segment']])
            validation_sample['title_token_0'] = title_a
            validation_sample['title_segment_0'] = title_a_seg
            validation_sample['title_token_1'] = title_b
            validation_sample['title_segment_1'] = title_b_seg
        # Add Topic
        if self.TOPIC_SIZE > 0:
            topic_a = np.concatenate([input_sample['topics'], input_sample['topics']])
            topic_b = np.concatenate([input_pos['topics'], input_neg['topics']])
            validation_sample['topic_0'] = topic_a
            validation_sample['topic_1'] = topic_b
        # Add Label
        batch_size_normalized = batch_size_test // 2
        pos = np.full((1, batch_size_normalized), 1)
        neg = np.full((1, batch_size_normalized), 0)
        sim = np.concatenate([pos, neg], -1)[0]
        encoder = LabelEncoder()
        sim = encoder.fit_transform(sim)
        valid_sim = to_categorical(sim)
            
        return batch_triplets_valid, validation_sample, valid_sim
