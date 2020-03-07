import numpy as np
from collections import OrderedDict, Counter
import pandas as pd
import tensorflow as tf
from deepctr.layers.utils import div, reduce_sum

def Negative_Sample(data, user_col, item_col, label_col, ratio, method_id=2):
    """
    :param data: training data
    :param user_col: user column name
    :param item_col: item column name for negative sampling
    :param label_col: label column name
    :param ratio: negative sample ratio, >= 1
    :param method_id: {0 : "random sampling", 1: "sampling method used in word2vec", 2: "tencent RALM sampling"}
    :return: new_dataframe, (user_id, item_id, label)
    """
    if not isinstance(ratio, int) or ratio < 1:
        raise ValueError("ratio means neg/pos, it should be greater than or equal to 1")
    items_cnt = Counter(data[item_col])
    items_cnt_order = OrderedDict(sorted((items_cnt.items()), key=lambda x: x[1], reverse=True))
    user_pos_item = data[data[label_col] == 1].drop(label_col, axis=1).groupby(user_col).agg(list).reset_index()
    if method_id == 0:
        def sample(row):
            neg_items = np.random.choice(list(items_cnt.keys()), size=ratio, replace=False)
            neg_items = [neg for neg in neg_items if neg not in row[item_col]]
            return neg_items
        user_pos_item['neg_'+item_col] = user_pos_item.apply(sample, axis=1)
    elif method_id == 1:
        items_cnt_freq = {item: count/len(items_cnt) for item, count in items_cnt_order.items()}
        p_sel = {item: np.sqrt(1e-5/items_cnt_freq[item]) for item in items_cnt_order}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        def sample(row):
            neg_items = np.random.choice(list(items_cnt.keys()), size=ratio, replace=False,p=p_value)
            neg_items = [neg for neg in neg_items if neg not in row[item_col]]
            return neg_items
        user_pos_item['neg_'+item_col] = user_pos_item.apply(sample, axis=1)
    elif method_id == 2:
        p_sel = {item: (np.log(k + 2) - np.log(k + 1) / np.log(len(items_cnt_order) + 1)) for item, k in
                 items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        def sample(row):
            neg_items = np.random.choice(list(items_cnt.keys()), size=ratio, replace=False, p=p_value)
            neg_items = [neg for neg in neg_items if neg not in row[item_col]]
            return neg_items
        user_pos_item['neg_'+item_col] = user_pos_item.apply(sample, axis=1)
    else:
        raise ValueError("method id should in (0,1,2)")
    neg_data = pd.DataFrame({user_col: user_pos_item[user_col], 'neg_'+item_col: user_pos_item['neg_'+item_col]})
    neg_data = neg_data.rename(columns={'neg_' + item_col: item_col}, inplace=False)
    pos_data = pd.DataFrame({user_col: user_pos_item[user_col], item_col: user_pos_item[item_col]})
    pos_data[label_col] = 1
    neg_data[label_col] = 0
    neg_data = neg_data.explode('uid')
    pos_data = pos_data.explode('uid')
    return pd.concat([pos_data, neg_data])


def Cosine_Similarity(query, candidate, gamma=1, axis=-1):
    query_norm = tf.norm(query, axis=axis)
    candidate_norm = tf.norm(candidate, axis=axis)
    cosine_score = reduce_sum(tf.multiply(query, candidate), -1)
    cosine_score = div(cosine_score, query_norm*candidate_norm+1e-8)
    cosine_score = tf.clip_by_value(cosine_score, -1, 1.0)*gamma
    return cosine_score