from src.nlp.tokenizer import Tokenizer

class TokenizerBase(Tokenizer):
    
    def __init__(self):
        pass 

    def apply(self, text):
        # Implement any transformation
        return text