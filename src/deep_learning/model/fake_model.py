import numpy as np

class FakeModel:

    def predict(self, inputs):
        n_lines = inputs[0].shape[0]
        return [np.ones_like(range(10))] * n_lines 