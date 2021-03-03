from deep_learning.model.classifier_base import ClassifierBase

class SiameseQATClassifier:
    
    def __init__(self, model, title_size=0, desc_size=0, 
                    categorical_size=0, topic_size=0):
        
        model = ClassifierBase(model, title_size=title_size, desc_size=desc_size, 
                    categorical_size=categorical_size, topic_size=topic_size).get_model()

        self.model = model
    
    def get_model(self):
        return self.model

    def get_metrics(self):
        return ['accuracy']

    def get_loss(self):
        return 'binary_crossentropy'