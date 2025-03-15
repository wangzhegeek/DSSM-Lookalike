import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import OrderedDict, Counter

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


def cosine_similarity(query, candidate, gamma=1):
    """
    计算余弦相似度
    :param query: 查询向量
    :param candidate: 候选向量
    :param gamma: 缩放因子
    :return: 余弦相似度
    """
    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(query, candidate, dim=1)
    # 裁剪并缩放
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0) * gamma
    return cosine_sim


class EarlyStopping:
    """早停机制，在验证集上连续多轮指标不再提升时停止训练"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 在多少个epoch内验证集指标没有提升后停止训练
            verbose (bool): 是否打印早停信息
            delta (float): 指标提升的最小变化量，小于此值视为没有提升
            path (str): 保存最佳模型的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss 