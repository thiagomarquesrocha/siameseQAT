import pytest
import tensorflow as tf
import numpy as np
from src.deep_learning.loss.triplet_loss import triplet_loss
from src.deep_learning.loss.quintet_loss import quintet_loss

tf.compat.v1.disable_eager_execution()

class TestDeepLearningLoss:

    @pytest.mark.skip(reason="fix input for new tensorflow")
    def test_quintet_loss_then_successful(self):
        with tf.compat.v1.Session() as sess:
            a = np.array([[1.0, 1.0, 2.0], [2.0, 3.0, 4.0], [1.0, 1.0, 2.0], 
                    [1.0, 1.0, 2.0], [3.0, 3.0, 4.0], [3.0, 1.0, 2.0]], np.float64)
            vector = tf.constant(a)
            tl = quintet_loss(vector)
            res = sess.run(tl).tolist()
            expected_loss = [0.25, 5.1875] # triplet, triplet_centroid
            assert len(res) == 2
            assert res == expected_loss

    @pytest.mark.skip(reason="fix input for new tensorflow")
    def test_triplet_loss_then_successful(self):
        with tf.compat.v1.Session() as sess:
            a = np.array([[1.0, 1.0, 2.0], [2.0, 3.0, 4.0], [1.0, 1.0, 2.0], 
                    [1.0, 1.0, 2.0], [3.0, 3.0, 4.0], [3.0, 1.0, 2.0]], np.float64)
            vector = tf.constant(a)
            tl = triplet_loss(vector)
            res = sess.run(tl)
            expected_loss = 0.25
            assert res == expected_loss