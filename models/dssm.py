# -*- coding:utf-8 -*-
"""
Author:
    Zhe Wang,734914022@qq.com

Reference:
    [1] Huang P S, He X, Gao J, et al. Learning deep structured semantic models for web search using clickthrough data[C]// ACM International Conference on Conference on Information & Knowledge Management. ACM, 2013:2333-2338.
"""

from tensorflow.python.keras.models import Model

from deepctr.inputs import input_from_feature_columns, build_input_features, combined_dnn_input
from deepctr.layers.core import DNN, PredictionLayer
from utils import Cosine_Similarity


def DSSM(user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True, dnn_hidden_units=(300, 300, 128), dnn_activation='tanh',
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary'):
    """Instantiates the Deep Structured Semantic Model architecture.
    :param user_dnn_feature_columns:An iterable containing user's features used by deep part of the model.
    :param item_dnn_feature_columns:An iterable containing item's the features used by deep part of the model.
    :param gamma: smoothing factor in the softmax function for DSSM
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    user_features = build_input_features(user_dnn_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_dnn_feature_columns,
                                                                         l2_reg_embedding, init_std, seed)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_dnn_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features, item_dnn_feature_columns,
                                                                         l2_reg_embedding, init_std, seed)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    user_dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  dnn_use_bn, seed, name="user_embedding")(user_dnn_input)

    item_dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  dnn_use_bn, seed, name="item_embedding")(item_dnn_input)

    score = Cosine_Similarity(user_dnn_out, item_dnn_out, gamma=gamma)

    output = PredictionLayer(task, False)(score)

    model = Model(inputs=user_inputs_list+item_inputs_list, outputs=output)

    return model




