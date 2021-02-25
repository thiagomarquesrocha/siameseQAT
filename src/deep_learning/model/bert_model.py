from keras_bert import compile_model, get_model
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Dense, Average, GlobalAveragePooling1D
from keras.models import Model
from utils.util import Util
from deep_learning.model.model_base import ModelBase

class BERTModel(ModelBase):
    
    def __init__(self, seq_len, name, number_of_layers=8):

        # Load pretrained BERT
        config_path, model_path, vocab_path, token_dict = Util.pretrained_bert()

        model = load_trained_model_from_checkpoint(
            config_path,
            model_path,
            training=True,
            use_adapter=True,
            seq_len=seq_len,
            trainable=['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i + 1) for i in range(12-number_of_layers, 13)] +
            ['Encoder-{}-FeedForward-Adapter'.format(i + 1) for i in range(12-number_of_layers, 13)] +
            ['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i + 1) for i in range(12-number_of_layers, 13)] +
            ['Encoder-{}-FeedForward-Norm'.format(i + 1) for i in range(number_of_layers)],
        )

        compile_model(model)
        inputs = model.inputs[:2]
        layers = ['Encoder-{}-MultiHeadSelfAttention-Adapter', 'Encoder-{}-FeedForward-Adapter', 
        'Encoder-{}-MultiHeadSelfAttention-Norm', 'Encoder-{}-FeedForward-Norm']
        outputs = []
        for i in range(1, 13):
            outputs += [ model.get_layer(layer.format(number_of_layers)).output for layer in layers ]
        outputs = Average()(outputs)
        outputs = GlobalAveragePooling1D()(outputs)
        outputs = Dense(300, activation='tanh')(outputs)
        
        model = Model(inputs, outputs, name='FeatureBERTGenerationModel{}'.format(name))

        super().__init__(model)