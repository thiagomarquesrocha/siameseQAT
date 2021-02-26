from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from deep_learning.model.model_base import ModelBase

class MLPModel(ModelBase):

    # Number of units in output
    OUTPUT_LAYER = 300

    def __init__(self, input_size):
        
        info_input = Input(shape=(input_size, ), name='Feature_BugInput')
        model = Dense(self.OUTPUT_LAYER, activation='tanh')(info_input)
        model = Model(inputs=[info_input], outputs=[model], name = 'FeatureMlpGenerationModel')
        super().__init__(model)