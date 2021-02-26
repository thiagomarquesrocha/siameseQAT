from nlp.tokenizer import Tokenizer

class TokenizerFake(Tokenizer):
    
    def __init__(self):
        pass 

    def apply(self, text):
        # Implement any transformation
        return text

    def encode(self, text, max_len):
        return text, text