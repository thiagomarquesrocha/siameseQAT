import logging
from utils.keras_utils import KerasUtils
from deep_learning.model.compile_model import compile_model
from deep_learning.model.siameseQAT_classifier import SiameseQATClassifier
from deep_learning.training.training_preparation import TrainingPreparation

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
        self.create_model()
        return self

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
