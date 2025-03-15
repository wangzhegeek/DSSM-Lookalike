import torch
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
import os
from .utils import EarlyStopping


def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs=10, early_stopping_patience=5, model_save_path='../saved_model/dssm_pytorch.pt',
                scheduler=None):
    """
    训练DSSM模型
    :param model: DSSM模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 设备（CPU或GPU）
    :param epochs: 训练轮数
    :param early_stopping_patience: 早停耐心值
    :param model_save_path: 模型保存路径
    :param scheduler: 学习率调度器
    :return: 训练历史记录
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=model_save_path)
    
    # 初始化训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
    
    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0
        
        # 训练一个epoch
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for features, labels in train_bar:
            # 将数据移动到设备
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累加损失
            train_loss += loss.item() * labels.size(0)
            train_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # 验证模式
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        # 验证一个epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for features, labels in val_bar:
                # 将数据移动到设备
                for key in features:
                    features[key] = features[key].to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # 累加损失
                val_loss += loss.item() * labels.size(0)
                
                # 收集预测和标签
                val_preds.append(outputs.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                
                val_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # 计算AUC
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)
        history['val_auc'].append(val_auc)
        
        # 打印epoch结果
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f} - '
              f'Val AUC: {val_auc:.4f}')
        
        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    
    return history


def evaluate_model(model, test_loader, criterion, device):
    """
    评估DSSM模型
    :param model: DSSM模型
    :param test_loader: 测试数据加载器
    :param criterion: 损失函数
    :param device: 设备（CPU或GPU）
    :return: 损失和AUC
    """
    # 评估模式
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    
    # 测试循环
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Evaluating')
        for features, labels in test_bar:
            # 将数据移动到设备
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # 累加损失
            test_loss += loss.item() * labels.size(0)
            
            # 收集预测和标签
            test_preds.append(outputs.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            
            test_bar.set_postfix({'loss': loss.item()})
    
    # 计算平均测试损失
    test_loss /= len(test_loader.dataset)
    
    # 计算AUC和对数损失
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_log_loss = log_loss(test_labels, test_preds)
    
    print(f'Test Loss: {test_loss:.4f} - '
          f'Test Log Loss: {test_log_loss:.4f} - '
          f'Test AUC: {test_auc:.4f}')
    
    return test_loss, test_log_loss, test_auc


def get_embeddings(model, data_loader, device):
    """
    获取用户和物品嵌入
    :param model: DSSM模型
    :param data_loader: 数据加载器
    :param device: 设备（CPU或GPU）
    :return: 用户嵌入和物品嵌入
    """
    # 评估模式
    model.eval()
    user_embeddings = []
    item_embeddings = []
    
    # 获取嵌入循环
    with torch.no_grad():
        for features, _ in tqdm(data_loader, desc='Getting embeddings'):
            # 将数据移动到设备
            for key in features:
                features[key] = features[key].to(device)
            
            # 获取用户和物品嵌入
            user_emb = model.get_user_embedding(features)
            item_emb = model.get_item_embedding(features)
            
            # 收集嵌入
            user_embeddings.append(user_emb.cpu().numpy())
            item_embeddings.append(item_emb.cpu().numpy())
    
    # 拼接嵌入
    user_embeddings = np.concatenate(user_embeddings)
    item_embeddings = np.concatenate(item_embeddings)
    
    return user_embeddings, item_embeddings 