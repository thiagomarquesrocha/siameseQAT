from src.deep_learning.model.siamese_base_model import SiameseBaseModel
from src.deep_learning.model.model_base import ModelBase
from src.deep_learning.loss.triplet_loss import triplet_loss_output
from keras.models import Model

class SiameseTA():
    
    def __init__(self, model_name, title_size=10, desc_size=100, 
                    categorical_size=10, topic_size=0, 
                    number_of_BERT_layers=8):
        
        # If NOT SiameseTAT then disables the topic feature
        if not isinstance(self, SiameseTAT):
            topic_size = 0
        
        model = SiameseBaseModel(title_size=title_size, desc_size=desc_size, 
                            categorical_size=categorical_size, topic_size=topic_size, 
                            number_of_BERT_layers=number_of_BERT_layers).get_model()
        inputs = model.input
        output = model.output
        model = Model(inputs = inputs, outputs = output, name = model_name)
        self.model = model

    def get_model(self):
        return self.model

    def get_metrics(self):
        return None

    def get_loss(self):
        return triplet_loss_output

class SiameseTAT(SiameseTA):
    def __init__(self, model_name, title_size=10, desc_size=100, 
                    categorical_size=10, topic_size=30, 
                    number_of_BERT_layers=8):
        super().__init__(model_name=model_name, title_size=title_size, 
                        desc_size=desc_size, categorical_size=categorical_size, 
                        topic_size=topic_size, 
                        number_of_BERT_layers=number_of_BERT_layers)
