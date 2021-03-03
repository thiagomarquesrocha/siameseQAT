import os
from keras_bert import load_vocabulary

class BertUtils:

    @staticmethod
    def pretrained_bert(pretrained_path='uncased_L-12_H-768_A-12'):
        config_path = os.path.join(pretrained_path, 'bert_config.json')
        model_path = os.path.join(pretrained_path, 'bert_model.ckpt')
        vocab_path = os.path.join(pretrained_path, 'vocab.txt')

        token_dict = load_vocabulary(vocab_path)

        return config_path, model_path, vocab_path, token_dict