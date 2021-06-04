from keras.layers import Input
from src.utils.util import Util

class ClassifierModel:

    def __init__(self, input_list, model_list):
        
        # Inputs
        for key, obj in input_list.items():
            obj['input'] = Input(shape = (obj['input_size'], ), name = key)

        # Outputs
        for obj in model_list.values():
            model_input = [input_list[i]['input'] for i in obj['input']]
            obj['feat'] = obj['model'](model_input)

        # Concatenate model features
        self.inputs = [tensor['input'] for tensor in Util.sort_dict_by_key(input_list).values()]
        self.model = [model['feat'] for model in Util.sort_dict_by_key(model_list).values()]


        

    