import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    DSSM, 
    Negative_Sample, 
    create_data_loader, 
    train_model, 
    evaluate_model, 
    get_embeddings
)


if __name__ == "__main__":
    # 设置随机种子
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    data = pd.read_csv('./data/train_sample.csv')
    
    # 数据预处理
    data['label'] = data['label'].replace(-1, 0)
    print(data.head())
    
    # 定义特征
    sparse_features = ['aid', 'uid']
    user_features = ['aid']
    item_features = ['uid']
    
    # 填充缺失值
    data[sparse_features] = data[sparse_features].fillna('-1')
    target = ['label']
    
    # 负采样
    data = Negative_Sample(data, 'aid', 'uid', 'label', 10, method_id=2)
    
    # 标签编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    # 构建特征维度字典
    user_feature_dims = {feat: data[feat].nunique() for feat in user_features}
    item_feature_dims = {feat: data[feat].nunique() for feat in item_features}
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=seed)  # 0.25 * 0.8 = 0.2
    
    # 创建数据加载器
    train_loader = create_data_loader(train_data, user_features, item_features, 'label', batch_size=256, shuffle=True)
    val_loader = create_data_loader(val_data, user_features, item_features, 'label', batch_size=256, shuffle=False)
    test_loader = create_data_loader(test_data, user_features, item_features, 'label', batch_size=256, shuffle=False)
    
    # 创建模型
    model = DSSM(
        user_feature_dims=user_feature_dims,
        item_feature_dims=item_feature_dims,
        embedding_dim=20,
        dnn_hidden_units=(300, 300, 128),
        dnn_activation='tanh',
        dnn_dropout=0,
        dnn_use_bn=True,
        gamma=1,
        init_std=0.0001
    ).to(device)
    
    # 打印模型结构
    print(model)
    
    # 定义损失函数和优化器
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=10,
        early_stopping_patience=3,
        model_save_path='./saved_model/dssm_pytorch.pt'
    )
    
    # 评估模型
    test_loss, test_log_loss, test_auc = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # 获取嵌入
    user_embeddings, item_embeddings = get_embeddings(
        model=model,
        data_loader=test_loader,
        device=device
    )
    
    # 打印嵌入形状
    print("User embedding shape:", user_embeddings.shape)
    print("Item embedding shape:", item_embeddings.shape)
    
    # 保存嵌入
    os.makedirs('./saved_model', exist_ok=True)
    np.save('./saved_model/user_embedding_pytorch.npy', user_embeddings)
    np.save('./saved_model/item_embedding_pytorch.npy', item_embeddings)
    
    print("Done!") 