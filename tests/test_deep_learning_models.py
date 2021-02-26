import pytest
import keras
from deep_learning.model.bert_model import BERTModel

class TestDeepLearningModels:

    # def test_fail(self):
    #     assert False

    def test_create_bert_model_then_successful(self):
        model = BERTModel(seq_len=20, name='textual', number_of_layers=1).get_model()
        model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        assert model != None
