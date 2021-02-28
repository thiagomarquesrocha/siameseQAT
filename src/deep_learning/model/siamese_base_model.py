from deep_learning.model.bert_model import BERTModel
from deep_learning.model.mlp_model import MLPModel
from deep_learning.model.siamese_model import SiameseModel
from deep_learning.model.model_base import ModelBase
from keras.models import Model
import keras

class SiameseBaseModel():

    def __init__(self, title_size=10, desc_size=100, categorical_size=2, 
                    topic_size=40, number_of_BERT_layers=8):
        model_name = 'bug_embed'
        input_list = {}
        model_list = {}

        if title_size > 0:
            title_feat = BERTModel(seq_len=title_size, number_of_layers=number_of_BERT_layers, model_name='TitleFeatureModel')
            input_list['title_token']   = { 'input_size' : title_size }
            input_list['title_segment'] = { 'input_size' : title_size }
            model_list['title_feat'] = {
                'input' : ['title_token', 'title_segment'],
                'model' : title_feat
            }
        if desc_size > 0:
            desc_feat = BERTModel(seq_len=desc_size, number_of_layers=number_of_BERT_layers, model_name='DescFeatureModel')
            input_list['desc_token']   = { 'input_size' : desc_size }
            input_list['desc_segment'] = { 'input_size' : desc_size }
            model_list['desc_feat'] =  {
                'input' : ['desc_token', 'desc_segment'],
                'model' : desc_feat
            }
        if topic_size > 0:
            topic_feat = MLPModel(input_size=topic_size, input_name="TopicFeatureBugInput", model_name='TopicFeatureModel')
            input_list['topic']  = { 'input_size' : topic_size }
            model_list['topic'] =  {
                'input' : ['topic'],
                'model' : topic_feat
            }
        if categorical_size > 0:
            categorical_feat = MLPModel(input_size=categorical_size, input_name="CategoricalFeatureBugInput", model_name='CategoricalFeatureModel')
            input_list['categorical']  = { 'input_size' : categorical_size }
            model_list['categorical'] =  {
                'input' : ['categorical'],
                'model' : categorical_feat
            }

        model = SiameseModel(model_name, input_list, model_list)
        
        self.model = model

    def get_model(self):
        return self.model.get_model()