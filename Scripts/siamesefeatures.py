from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, merge, BatchNormalization, Activation, Input, Merge
from keras import backend as K


def create_base_network(input_dim):

    input = Input(shape=(input_dim,))
    dense1 = Dense(128)(input)
    bn1 = BatchNormalization()(dense1)
    relu1 = Activation('relu')(bn1)

    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization()(dense2)
    res2 = merge([relu1, bn2], mode='sum')
    relu2 = Activation('relu')(res2)

    dense3 = Dense(128)(relu2)
    bn3 = BatchNormalization()(dense3)
    res3 = Merge(mode='sum')([relu2, bn3])
    relu3 = Activation('relu')(res3)

    feats = merge([relu3, relu2, relu1], mode='concat')
    bn4 = BatchNormalization()(feats)

    model = Model(input=input, output=bn4)

    return model



def create_network(input_dim):

    base_network = create_base_network(input_dim)

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    L1_distance = lambda x: K.abs(x[0] - x[1])

    both = merge([processed_a, processed_b], mode=L1_distance, output_shape=lambda x: x[0])
    prediction = Dense(1, activation='sigmoid')(both)

    model = Model(input=[input_a, input_b], output=prediction)

    return model