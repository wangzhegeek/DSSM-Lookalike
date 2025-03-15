import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import cosine_similarity

class DNN(nn.Module):
    """
    深度神经网络模块，用于特征提取
    """
    def __init__(self, input_dim, hidden_units, activation='tanh', dropout_rate=0, use_bn=True, init_std=0.0001):
        """
        初始化DNN模块
        :param input_dim: 输入维度
        :param hidden_units: 隐藏层单元数列表，如 [300, 300, 128]
        :param activation: 激活函数，默认为tanh
        :param dropout_rate: dropout比率
        :param use_bn: 是否使用批归一化
        :param init_std: 权重初始化标准差
        """
        super(DNN, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_units[0]))
        if self.use_bn:
            self.bn_layers.append(nn.BatchNorm1d(hidden_units[0]))
            
        # 构建剩余隐藏层
        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            if self.use_bn:
                self.bn_layers.append(nn.BatchNorm1d(hidden_units[i+1]))
        
        # 设置激活函数
        if activation.lower() == 'relu':
            self.activation = F.relu
        elif activation.lower() == 'tanh':
            self.activation = torch.tanh
        elif activation.lower() == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Activation function {activation} not supported")
        
        # 初始化权重
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0, std=init_std)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            if self.dropout_rate > 0:
                x = self.dropout(x)
        return x


class DSSM(nn.Module):
    """
    Deep Structured Semantic Model (DSSM)
    """
    def __init__(self, user_feature_dims, item_feature_dims, embedding_dim=20, 
                 dnn_hidden_units=(300, 300, 128), dnn_activation='tanh',
                 dnn_dropout=0, dnn_use_bn=True, gamma=1, init_std=0.0001):
        """
        初始化DSSM模型
        :param user_feature_dims: 用户特征维度字典，格式为 {feature_name: vocab_size}
        :param item_feature_dims: 物品特征维度字典，格式为 {feature_name: vocab_size}
        :param embedding_dim: 嵌入维度
        :param dnn_hidden_units: DNN隐藏层单元数列表
        :param dnn_activation: DNN激活函数
        :param dnn_dropout: DNN dropout比率
        :param dnn_use_bn: DNN是否使用批归一化
        :param gamma: 余弦相似度缩放因子
        :param init_std: 权重初始化标准差
        """
        super(DSSM, self).__init__()
        
        self.user_feature_names = list(user_feature_dims.keys())
        self.item_feature_names = list(item_feature_dims.keys())
        self.gamma = gamma
        
        # 创建用户特征嵌入层
        self.user_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in user_feature_dims.items():
            self.user_embeddings[feature_name] = nn.Embedding(vocab_size, embedding_dim)
            # 初始化嵌入层
            nn.init.normal_(self.user_embeddings[feature_name].weight, mean=0, std=init_std)
        
        # 创建物品特征嵌入层
        self.item_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in item_feature_dims.items():
            self.item_embeddings[feature_name] = nn.Embedding(vocab_size, embedding_dim)
            # 初始化嵌入层
            nn.init.normal_(self.item_embeddings[feature_name].weight, mean=0, std=init_std)
        
        # 计算用户和物品特征的总嵌入维度
        user_embedding_dim = len(user_feature_dims) * embedding_dim
        item_embedding_dim = len(item_feature_dims) * embedding_dim
        
        # 创建用户和物品的DNN
        self.user_dnn = DNN(user_embedding_dim, dnn_hidden_units, dnn_activation, 
                           dnn_dropout, dnn_use_bn, init_std)
        self.item_dnn = DNN(item_embedding_dim, dnn_hidden_units, dnn_activation, 
                           dnn_dropout, dnn_use_bn, init_std)
        
        # 二分类输出层
        self.output_layer = nn.Sigmoid()
    
    def forward(self, features):
        """
        前向传播
        :param features: 特征字典，包含用户和物品特征
        :return: 预测分数
        """
        # 提取用户特征嵌入
        user_embeddings = []
        for feature_name in self.user_feature_names:
            user_embeddings.append(self.user_embeddings[feature_name](features[feature_name]))
        
        # 提取物品特征嵌入
        item_embeddings = []
        for feature_name in self.item_feature_names:
            item_embeddings.append(self.item_embeddings[feature_name](features[feature_name]))
        
        # 拼接用户特征嵌入
        if len(user_embeddings) > 1:
            user_embedding = torch.cat(user_embeddings, dim=1)
        else:
            user_embedding = user_embeddings[0]
        
        # 拼接物品特征嵌入
        if len(item_embeddings) > 1:
            item_embedding = torch.cat(item_embeddings, dim=1)
        else:
            item_embedding = item_embeddings[0]
        
        # 通过DNN获取用户和物品的表示
        user_dnn_out = self.user_dnn(user_embedding)
        item_dnn_out = self.item_dnn(item_embedding)
        
        # 计算余弦相似度
        cosine_sim = cosine_similarity(user_dnn_out, item_dnn_out, self.gamma)
        
        # 输出预测分数
        output = self.output_layer(cosine_sim)
        
        return output
    
    def get_user_embedding(self, features):
        """
        获取用户嵌入表示
        :param features: 特征字典
        :return: 用户嵌入
        """
        # 提取用户特征嵌入
        user_embeddings = []
        for feature_name in self.user_feature_names:
            user_embeddings.append(self.user_embeddings[feature_name](features[feature_name]))
        
        # 拼接用户特征嵌入
        if len(user_embeddings) > 1:
            user_embedding = torch.cat(user_embeddings, dim=1)
        else:
            user_embedding = user_embeddings[0]
        
        # 通过DNN获取用户表示
        user_dnn_out = self.user_dnn(user_embedding)
        
        return user_dnn_out
    
    def get_item_embedding(self, features):
        """
        获取物品嵌入表示
        :param features: 特征字典
        :return: 物品嵌入
        """
        # 提取物品特征嵌入
        item_embeddings = []
        for feature_name in self.item_feature_names:
            item_embeddings.append(self.item_embeddings[feature_name](features[feature_name]))
        
        # 拼接物品特征嵌入
        if len(item_embeddings) > 1:
            item_embedding = torch.cat(item_embeddings, dim=1)
        else:
            item_embedding = item_embeddings[0]
        
        # 通过DNN获取物品表示
        item_dnn_out = self.item_dnn(item_embedding)
        
        return item_dnn_out 