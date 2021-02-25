import abc

class Tokenizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def apply(text):
        pass