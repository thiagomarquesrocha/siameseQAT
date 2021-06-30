import tensorflow as tf
from tensorflow.keras.layers import Lambda, concatenate
from tensorflow.keras.models import Model
from src.deep_learning.model.siamese_base_model import SiameseBaseModel
from src.deep_learning.model.model_base import ModelBase
from src.deep_learning.loss.quintet_loss import quintet_loss, QuintetWeights, \
                                            quintet_trainable, quintet_loss_output, \
                                            TL_w, TL_w_centroid, TL, TL_centroid

class SiameseQA():

    def __init__(self, model_name, title_size=10, desc_size=100, 
                    categorical_size=10, topic_size=0, 
                    number_of_BERT_layers=8, trainable=False):
        
        # If NOT SiameseQAT then disables the topic feature
        if not isinstance(self, SiameseQAT):
            topic_size = 0
        
        model = SiameseBaseModel(title_size=title_size, desc_size=desc_size, 
                            categorical_size=categorical_size, topic_size=topic_size, 
                            number_of_BERT_layers=number_of_BERT_layers).get_model()
        inputs = model.input
        TL_loss = Lambda(quintet_loss, name='quintet_loss')(model.output)
    
        tl_l = Lambda(lambda x:tf.reshape(x[0], (1,)), name='TL')(TL_loss)
        tl_l_c = Lambda(lambda x:tf.reshape(x[1], (1,)), name='TL_centroid')(TL_loss)
        
        TL_w = QuintetWeights(output_dim=1, trainable=trainable)(tl_l)
        TL_centroid_w = QuintetWeights(output_dim=1, trainable=trainable)(tl_l_c)
        
        TL_weight = Lambda(lambda x:tf.reshape(x[1], (1,)), name='TL_weight')(TL_w)
        TL_centroid_weight = Lambda(lambda x:tf.reshape(x[1], (1,)), name='TL_centroid_weight')(TL_centroid_w)
        
        output = concatenate([tl_l, tl_l_c, TL_weight, TL_centroid_weight])
        output = Lambda(quintet_trainable, name='quintet_trainable')(output)
        model = Model(inputs = inputs, outputs = output, name = model_name)

        self.model = model
    
    def get_model(self):
        return self.model

    def get_metrics(self):
        return [TL_w, TL_w_centroid, TL, TL_centroid]

    def get_loss(self):
        return quintet_loss_output

class SiameseQAT(SiameseQA):
    
    def __init__(self, model_name, title_size=10, desc_size=100, 
                    categorical_size=10, topic_size=30, 
                    number_of_BERT_layers=8, trainable=False):
        super().__init__(model_name=model_name, title_size=title_size, 
                        desc_size=desc_size, categorical_size=categorical_size, 
                        topic_size=topic_size, 
                        number_of_BERT_layers=number_of_BERT_layers, 
                        trainable=trainable)