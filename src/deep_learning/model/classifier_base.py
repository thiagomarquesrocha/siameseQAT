from keras.layers import Dense, Dropout, Activation, concatenate
from keras.models import Model
from deep_learning.model.classifier_model import ClassifierModel
import numpy as np

class ClassifierBase:

    NUMBER_OF_UNITS = 2

    def __init__(self, model, title_size=0, desc_size=0, 
                    categorical_size=0, topic_size=0):
        model_name = 'bug_classifier'
        
        encoder = model.get_layer('concatenated_bug_embed')
        bugs_inputs = []
        bugs_embed = []
        for i in range(2):
            input_list = {}
            model_list = {}

            if title_size > 0:
                title_feat = model.get_layer('title_encoder')
                input_list['title_token_{}'.format(i)]   = { 'input_size' : title_size }
                input_list['title_segment_{}'.format(i)] = { 'input_size' : title_size }
                model_list['title_feat'] = {
                    'input' : ['title_token_{}'.format(i), 'title_segment_{}'.format(i)],
                    'model' : title_feat,
                    'name'  : 'title_encoder'
                }
            if desc_size > 0:
                desc_feat = model.get_layer('description_encoder')
                input_list['desc_token_{}'.format(i)]   = { 'input_size' : desc_size }
                input_list['desc_segment_{}'.format(i)] = { 'input_size' : desc_size }
                model_list['desc_feat'] =  {
                    'input' : ['desc_token_{}'.format(i), 'desc_segment_{}'.format(i)],
                    'model' : desc_feat,
                    'name'  : 'description_encoder'
                }
            if topic_size > 0:
                topic_feat = model.get_layer('topic_encoder')
                input_list['topic_{}'.format(i)]  = { 'input_size' : topic_size }
                model_list['topic'] =  {
                    'input' : ['topic_{}'.format(i)],
                    'model' : topic_feat,
                    'name'  : 'topic_encoder',
                }
            if categorical_size > 0:
                categorical_feat = model.get_layer('categorical_encoder')
                input_list['categorical_{}'.format(i)]  = { 'input_size' : categorical_size }
                model_list['categorical'] =  {
                    'input' : ['categorical_{}'.format(i)],
                    'model' : categorical_feat,
                    "name"  : "categorical_encoder"
                }

            bug_feat = ClassifierModel(input_list, model_list)
            bugs_inputs.append(bug_feat.inputs)
            bug_embed = encoder(bug_feat.model)
            bugs_embed.append(bug_embed)
        
        x = concatenate(bugs_embed, name='bugs') # 

        for _ in range(self.NUMBER_OF_UNITS):
            x = Dense(64)(x)
            x = Dropout(0.25)(x)
            x = Activation('tanh')(x)

        inputs = np.concatenate(bugs_inputs).tolist()
        output = Dense(2, activation = 'softmax', name = 'softmax')(x)
        
        model = Model(inputs=inputs, outputs=[output], name=model_name)

        self.model = model

    def get_model(self):
        return self.model