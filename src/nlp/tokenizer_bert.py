import os
from nlp.tokenizer import Tokenizer
from keras_bert import Tokenizer as KerasBertTokenizer
from keras_bert import load_vocabulary

class TokenizerBert(Tokenizer):
    
    def __init__(self):
        pretrained_path = 'uncased_L-12_H-768_A-12'
        config_path = os.path.join(pretrained_path, 'bert_config.json')
        model_path = os.path.join(pretrained_path, 'bert_model.ckpt')
        vocab_path = os.path.join(pretrained_path, 'vocab.txt')

        token_dict = load_vocabulary(vocab_path)
        print("Total vocabulary loaded: {}".format(len(token_dict)))

        self.tokenizer = KerasBertTokenizer(token_dict)

    def apply(self, text):
        text = " ".join(self.tokenizer.tokenize(str(text)))
        return text

    def encode(self, text, max_len):
        return self.tokenizer.encode(text, max_len=max_len)