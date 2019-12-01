import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

# Pandas's display settings
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def rmse(label, prop):
    return tf.sqrt(tf.reduce_mean(tf.square(label - prop + 1e-10), axis=-1))


def cross_entropy(label, prop):
    return -(label * tf.log(tf.clip_by_value(prop, 1e-10, 1.)) +
             (1 - label) * tf.log(tf.clip_by_value(1. - prop, 1e-10, 1.)))


def batch_norm(x, axis=-1, training=True):
    return tf.layers.batch_normalization(
        inputs=x, axis=axis,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def fc_layer(x, units, training=True, dropout=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        inputs = tf.layers.dense(x, units=units, activation=None, use_bias=False)
        inputs = tf.nn.relu(batch_norm(inputs, training=training))
        if dropout:
            return tf.layers.dropout(inputs, rate=0.25, training=training, name='output')
        else:
            return inputs


def bilinear_layer(x, units, training=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        shortcut = x
        inputs = fc_layer(x, units, training=training, name='fc_0')
        inputs = fc_layer(inputs, units, training=training, name='fc_1')
        return tf.add_n([inputs, shortcut], name='output')
