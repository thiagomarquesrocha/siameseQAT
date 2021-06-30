from keras_bert import compile_model, get_model
from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras.layers import Dense, Average, Concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Input, Bidirectional, LSTM
from tensorflow.keras.models import Model
from src.utils.bert_utils import BertUtils
from src.deep_learning.model.model_base import ModelBase
from tensorflow.keras import backend as K

class BERTModel(ModelBase):
    
    # Number of units in output
    OUTPUT_LAYER = 300

    def __init__(self, seq_len, model_name, number_of_layers=8):

        # Load pretrained BERT
        config_path, model_path, vocab_path, token_dict = BertUtils.pretrained_bert()

        model = load_trained_model_from_checkpoint(
            config_path,
            model_path,
            training=False, # # The input layers and output layer will be returned if `training` is `False`
            seq_len=seq_len,
            trainable=False, # Whether the model is trainable. The default value is the same with `training`
            output_layer_num=4, # # The number of layers whose outputs will be concatenated as a single output.
        )

        inputs = model.inputs[:2]
        outputs = model.output
        bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(outputs)
        avg_output = GlobalAveragePooling1D()(bi_lstm)
        max_output = GlobalMaxPooling1D()(bi_lstm)
        outputs = Concatenate()([avg_output, max_output])
        self.OUTPUT_LAYER = K.int_shape(outputs)
        
        model = Model(inputs, outputs, name=model_name)

        super().__init__(model, self.OUTPUT_LAYER, name=model_name)