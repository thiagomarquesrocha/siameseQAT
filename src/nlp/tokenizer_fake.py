from src.nlp.tokenizer import Tokenizer

class TokenizerFake(Tokenizer):
    
    def __init__(self):
        pass 

    def apply(self, text):
        # Implement clean text transformation
        return text

    def encode(self, text, max_len):
        return text, text