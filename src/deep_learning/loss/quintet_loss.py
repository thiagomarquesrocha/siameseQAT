from tensorflow import keras
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from src.deep_learning.loss.triplet_loss import triplet_loss
from src.deep_learning.loss.pairwise_loss import pairwise_distance, \
                                                masked_maximum, masked_minimum

class QuintetWeights(Layer):

    def __init__(self, output_dim, trainable=False, **kwargs):
        self.output_dim = output_dim
        self.trainable = trainable
        super(QuintetWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = tf.reshape(self.add_weight(name='quintet_kernel_weight', 
                                      shape=(input_shape[0], self.output_dim),
                                      initializer=keras.initializers.Ones(),
                                      trainable=self.trainable), (1, 1))
        super(QuintetWeights, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = tf.reshape(x, (1, 1))
        return [K.dot(x, self.kernel), self.kernel]

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]

def quintet_loss(inputs):
    margin = 1.
    labels = inputs[:, :1]
 
    labels = tf.cast(labels, dtype='int32')

    embeddings = inputs[:, 1:]

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size  
    batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])

    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)
    
    semi_hard_negatives_mask = math_ops.equal(pdist_matrix, masked_maximum(pdist_matrix, adjacency_not))
    
    # Remove false negatives with similarity equal to the true negatives
    semi_hard_negatives_mask = tf.reshape(tf.cast(semi_hard_negatives_mask, 'int32'), (-1, 1)) * tf.reshape(tf.cast(adjacency_not, 'int32'), (-1, 1))
    semi_hard_negatives_mask = tf.cast(tf.reshape(semi_hard_negatives_mask, (batch_size, batch_size)), 'bool')
    
    # Recovery the bug label from semi-hard-negatives
    label_matrix = tf.repeat(tf.reshape(labels, (1, -1)), repeats=[batch_size], axis=0)
    semi_hard_negatives_ids = tf.reshape(label_matrix, (-1, 1)) * tf.cast(tf.reshape(semi_hard_negatives_mask, (-1, 1)), 'int32')
    semi_hard_negatives_ids = tf.reshape(semi_hard_negatives_ids, (batch_size, batch_size))

    i = tf.constant(0)
    most_freq_matrix = tf.Variable([])
    def most_frequent(i, most_freq_matrix):
        batch = tf.gather(semi_hard_negatives_ids, i)
        neg_label_default = [tf.unique(batch)[0][0]]
        batch = tf.boolean_mask(batch, tf.greater(batch, 0))
        unique, _, count = tf.unique_with_counts(batch)
        max_occurrences = tf.reduce_max(count)
        max_cond = tf.equal(count, max_occurrences)
        max_numbers = tf.squeeze(tf.gather(unique, tf.where(max_cond)))
        max_numbers = tf.cond(tf.cast(tf.size(unique) > 1, tf.bool), lambda: unique[0], lambda: max_numbers)
        max_numbers = tf.cond(tf.cast(tf.shape(unique) == 0, tf.bool), 
                              lambda: neg_label_default, 
                              lambda: max_numbers)
        most_freq_matrix = tf.concat([most_freq_matrix, [max_numbers]], axis=0)
        return [tf.add(i, 1), most_freq_matrix]
    _, negatives_ids = tf.while_loop(lambda i, _: i<batch_size, 
                                        most_frequent, 
                                        [i, most_freq_matrix],
                                       shape_invariants=[i.get_shape(),
                                                   tf.TensorShape([None])])
    negatives_ids = tf.cast(negatives_ids, 'int32')
    labels_neg = tf.reshape(negatives_ids, (-1, 1))
    mask_negatives = math_ops.equal(labels_neg, semi_hard_negatives_ids)
    mask_negatives = tf.cast(mask_negatives, 'float32')
    labels_neg = tf.cast(labels_neg, 'float32')
    
    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))
    
    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    # Include the anchor to positives
    mask_positives_centroids = math_ops.cast(adjacency, dtype=dtypes.float32)

    # centroid pos 
    embed_pos = tf.matmul(mask_positives, embeddings)
    num_of_pos = tf.reduce_sum(mask_positives_centroids, axis=1, keepdims=True)
    centroid_embed_pos = tf.math.xdivy(embed_pos, num_of_pos)
    labels_pos = tf.cast(labels, dtype=dtypes.float32)

    # centroid negs
    embed_neg = tf.matmul(mask_negatives, embeddings)
    num_of_neg = tf.reduce_sum(mask_negatives, axis=1, keepdims=True)
    centroid_embed_neg = tf.math.xdivy(embed_neg, num_of_neg)
    
    i = tf.constant(0)
    batch_centroid_matrix = tf.Variable([])
    def iter_centroids(i, batch_centroid_matrix):
        # anchor
        anchor = [tf.gather(embeddings, i)]
        label_pos = [tf.gather(labels_pos, i)]
        # centroid pos
        centroid_pos = [tf.gather(centroid_embed_pos, i)]
        # centroid neg
        centroid_neg = [tf.gather(centroid_embed_neg, i)]
        label_neg = [tf.gather(labels_neg, i)]
        # new batch
        new_batch = tf.concat([anchor, centroid_pos, centroid_neg], axis=0)
        new_labels = tf.concat([label_pos, label_pos, label_neg], axis=0)
        # Batch anchor + centroid_positive + centroid_negative
        batch_anchor_centroids = tf.concat([new_labels, new_batch], axis=1)
        TL_single_triplet = triplet_loss(batch_anchor_centroids)
        batch_centroid_matrix = tf.concat([batch_centroid_matrix, [TL_single_triplet]], axis=0)
        
        return [tf.add(i, 1), batch_centroid_matrix]
    _, batch_centroid_matrix = tf.while_loop(lambda i, a: i<batch_size, 
                                        iter_centroids, 
                                        [i, batch_centroid_matrix],
                                       shape_invariants=[i.get_shape(),
                                                   tf.TensorShape([None])])
    
    TL_centroid = tf.reduce_mean(batch_centroid_matrix)
    TL = triplet_loss(inputs)
   
    return K.stack([TL, TL_centroid], axis=0)

def quintet_trainable(inputs):
    TL = inputs[0]
    TL_centroid = inputs[1]
    TL_anchor_w = inputs[2]
    TL_centroid_w = inputs[3]
    
    TL_anchor_w = math_ops.maximum(TL_anchor_w, 0.0)
    TL_centroid_w = math_ops.maximum(TL_centroid_w, 0.0)

    sum_of_median = tf.reduce_sum([ TL * TL_anchor_w, TL_centroid * TL_centroid_w])
    sum_of_weigths = tf.reduce_sum([TL_anchor_w, TL_centroid_w])
    weigthed_median = tf.truediv(sum_of_median, sum_of_weigths)    
    return K.stack([weigthed_median, TL_anchor_w, TL_centroid_w, TL, TL_centroid], axis=0)

def quintet_loss_output(y_true, y_pred):
    return tf.reduce_mean(y_pred[0])

def TL_w(y_true, y_pred):
    return tf.reduce_mean(y_pred[1])

def TL_w_centroid(y_true, y_pred):
    return tf.reduce_mean(y_pred[2])

def TL(y_true, y_pred):
    return tf.reduce_mean(y_pred[3])

def TL_centroid(y_true, y_pred):
    return tf.reduce_mean(y_pred[4])