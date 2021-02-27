from keras.layers import Input, concatenate
from keras.models import Model
from deep_learning.model.model_base import ModelBase

class SiameseModel(ModelBase):

    def __init__(self, model_name, input_list, model_list):
        
        # Inputs
        for key, obj in input_list.items():
            obj['input'] = Input(shape = (obj['input_size'], ), name = key) 
        # Outputs
        for obj in model_list.values():
            model_input = [input_list[i]['input'] for i in obj['input']]
            obj['feat'] = obj['model'](model_input)
        
        # Concatenate model features
        inputs = [tensor['input'] for tensor in input_list.values()]
        cancat_list = [model['feat'] for model in model_list.values()]
        embed = concatenate(cancat_list, name = 'concatenated_feat_{}'.format(model_name))

        # input layer for labels
        input_labels = Input(shape=(1,), name='input_label')
        inputs.append(input_labels)
        # concatenating the labels + embeddings
        output = concatenate([input_labels, embed])
        model = Model(inputs=inputs, outputs=[output], name = model_name)

        super().__init__(model)