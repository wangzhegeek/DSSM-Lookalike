import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class DSSMDataset(Dataset):
    """
    DSSM模型的数据集类
    """
    def __init__(self, data, user_features, item_features, label_col='label'):
        """
        初始化数据集
        :param data: 数据DataFrame
        :param user_features: 用户特征列名列表
        :param item_features: 物品特征列名列表
        :param label_col: 标签列名
        """
        self.data = data
        self.user_features = user_features
        self.item_features = item_features
        self.label_col = label_col
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取样本
        sample = self.data.iloc[idx]
        
        # 提取特征
        features = {}
        for feature in self.user_features:
            features[feature] = torch.tensor(sample[feature], dtype=torch.long)
        
        for feature in self.item_features:
            features[feature] = torch.tensor(sample[feature], dtype=torch.long)
        
        # 提取标签
        label = torch.tensor(sample[self.label_col], dtype=torch.float32)
        
        return features, label


def create_data_loader(data, user_features, item_features, label_col='label', batch_size=256, shuffle=True, num_workers=0):
    """
    创建数据加载器
    :param data: 数据DataFrame
    :param user_features: 用户特征列名列表
    :param item_features: 物品特征列名列表
    :param label_col: 标签列名
    :param batch_size: 批次大小
    :param shuffle: 是否打乱数据
    :param num_workers: 数据加载的工作线程数
    :return: 数据加载器
    """
    dataset = DSSMDataset(data, user_features, item_features, label_col)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return data_loader


def collate_fn(batch):
    """
    数据批次整理函数
    :param batch: 批次数据
    :return: 整理后的批次数据
    """
    features_batch = {}
    labels = []
    
    # 获取第一个样本的特征名
    feature_names = batch[0][0].keys()
    
    # 初始化特征字典
    for name in feature_names:
        features_batch[name] = []
    
    # 收集所有样本的特征和标签
    for features, label in batch:
        for name, value in features.items():
            features_batch[name].append(value)
        labels.append(label)
    
    # 将特征列表转换为张量
    for name in feature_names:
        features_batch[name] = torch.stack(features_batch[name])
    
    # 将标签列表转换为张量
    labels = torch.stack(labels)
    
    return features_batch, labels 