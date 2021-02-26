import pytest
import os
import keras
from deep_learning.model.bert_model import BERTModel
from deep_learning.model.mlp_model import MLPModel

class TestDeepLearningModels:
  
    @pytest.mark.skipif(not os.path.exists('uncased_L-12_H-768_A-12'), reason="does not run without pretrained bert")
    def test_create_bert_model_then_successful(self):
        model = BERTModel(seq_len=20, model_name='FeatureBERTModel', number_of_layers=1)
        model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        assert model != None

    def test_create_mlp_model_then_successful(self):
        model = MLPModel(input_size=10, input_name="FeatureBugInput", model_name='FeatureMLPModel')
        model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        assert model != None
