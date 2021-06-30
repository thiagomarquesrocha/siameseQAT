from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from src.deep_learning.model.model_base import ModelBase

class MLPModel(ModelBase):

    # Number of units in output
    OUTPUT_LAYER = 300

    def __init__(self, input_size, input_name, model_name):
        
        info_input = Input(shape=(input_size, ), name=input_name)
        model = Dense(self.OUTPUT_LAYER, activation='tanh')(info_input)
        model = Model(inputs=[info_input], outputs=[model], name = model_name)
        super().__init__(model, self.OUTPUT_LAYER, name=model_name)