import keras

class ModelBase(keras.Model):
    
    def __init__(self, model):
        super(ModelBase, self).__init__()
        self.model = model

    def call(self, inputs):
        return self.model(inputs)

    def compute_output_shape(self, input_shape):
         return input_shape