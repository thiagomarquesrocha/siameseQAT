from keras.layers import Input, concatenate
from keras.models import Model
from keras import backend as K
from src.deep_learning.model.model_base import ModelBase
from src.utils.util import Util
import logging

logger = logging.getLogger('SiameseModel')

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
        inputs = [tensor['input'] for tensor in Util.sort_dict_by_key(input_list).values()]
        concat_list = [model['feat'] for model in Util.sort_dict_by_key(model_list).values()]
        embed = concatenate(concat_list, name = 'concatenated_{}'.format(model_name))

        # input layer for labels
        input_labels = Input(shape=(1,), name='input_label')
        inputs.append(input_labels)
        logger.debug("Inputs: {}".format(inputs))
        # concatenating the labels + embeddings
        output = concatenate([input_labels, embed])
        model = Model(inputs=inputs, outputs=[output], name = model_name)
        OUTPUT_LAYER = K.int_shape(input_labels) + K.int_shape(embed)
        super().__init__(model, OUTPUT_LAYER)