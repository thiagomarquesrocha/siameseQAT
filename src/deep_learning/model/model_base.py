import keras

class ModelBase(keras.Model):
    
    def __init__(self, model, output_int_shape):
        super(ModelBase, self).__init__()
        self.model = model
        self.output_int_shape = output_int_shape

    def call(self, inputs):
        return self.model(inputs)

    def compute_output_shape(self, input_shape):
         return (None, self.output_int_shape)

    def get_model(self):
        return self.model

    def get_config(self):
        config = {}
        config.update({ "output_int_shape" : self.output_int_shape})
        return config