import pytest
import os
import keras
from deep_learning.model.bert_model import BERTModel
from deep_learning.model.mlp_model import MLPModel
from deep_learning.model.siamese_model import SiameseModel

class TestDeepLearningModels:
    
    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_create_siamese_model_then_successful(self):
        keras.backend.clear_session()
        model_name = 'SiameseModel'
        title_size = 1
        desc_size = 1
        categorical_size = 1
        topic_size = 1
        
        title_feat = BERTModel(seq_len=title_size, number_of_layers=1, model_name='TitleFeatureModel')
        desc_feat = BERTModel(seq_len=desc_size, number_of_layers=1, model_name='DescFeatureModel')
        topic_feat = MLPModel(input_size=topic_size, input_name="TopicFeatureBugInput", model_name='TopicFeatureModel')
        categorical_feat = MLPModel(input_size=categorical_size, input_name="CategoricalFeatureBugInput", model_name='CategoricalFeatureModel')

        input_list = {
            'title_token'   : { 'input_size' : title_size },
            'title_segment' : { 'input_size' : title_size },
            'desc_token'    : { 'input_size' : desc_size },
            'desc_segment'  : { 'input_size' : desc_size },
            'categorical'   : { 'input_size' : categorical_size },
            'topic'         : { 'input_size' : topic_size }
        }

        model_list = {
            'title_feat' : {
                'input' : ['title_token', 'title_segment'],
                'model' : title_feat
            },
            'desc_feat' : {
                'input' : ['desc_token', 'desc_segment'],
                'model' : desc_feat
            },
            'categorical' : {
                'input' : ['categorical'],
                'model' : categorical_feat
            },
            'topic' : {
                'input' : ['topic'],
                'model' : topic_feat
            }
        }
        model = SiameseModel(model_name, input_list, model_list)
        model.compile(optimizer="Adam", loss="mse")
        assert model != None

    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_create_bert_model_then_successful(self):
        model = BERTModel(seq_len=20, model_name='FeatureBERTModel', number_of_layers=1)
        model.compile(optimizer="Adam", loss="mse")
        assert model != None

    def test_create_mlp_model_then_successful(self):
        model = MLPModel(input_size=10, input_name="FeatureBugInput", model_name='FeatureMLPModel')
        model.compile(optimizer="Adam", loss="mse")
        assert model != None