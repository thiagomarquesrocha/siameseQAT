import pytest
import tensorflow as tf
from deep_learning.loss.triplet_loss import triplet_loss

class TestDeepLearningLosses:

    def test_triplet_loss_then_successful(self):
        vector = tf.constant([[1.0, 1.0, 2.0], [2.0, 3.0, 4.0], [1.0, 1.0, 2.0], 
                 [1.0, 1.0, 2.0], [3.0, 3.0, 4.0], [3.0, 1.0, 2.0]])
        tl = triplet_loss(vector)
        sess = tf.Session()
        sess.run(tl)
        assert True