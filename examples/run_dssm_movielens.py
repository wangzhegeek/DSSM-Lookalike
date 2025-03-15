import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    DSSM, 
    create_data_loader, 
    train_model, 
    evaluate_model, 
    get_embeddings
)


def plot_training_history(history):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 绘制AUC曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('AUC Curve')
    
    plt.tight_layout()
    plt.savefig('./saved_model/training_history_movielens.png')
    plt.close()


def load_movielens_data(ratings_file, users_file, movies_file):
    """
    加载MovieLens-1M数据集
    :param ratings_file: 评分文件路径
    :param users_file: 用户文件路径
    :param movies_file: 电影文件路径
    :return: 处理后的数据
    """
    # 加载评分数据
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(ratings_file, sep='::', names=ratings_cols, engine='python', encoding='latin1')
    
    # 加载用户数据
    users_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users = pd.read_csv(users_file, sep='::', names=users_cols, engine='python', encoding='latin1')
    
    # 加载电影数据
    movies_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(movies_file, sep='::', names=movies_cols, engine='python', encoding='latin1')
    
    # 将评分转换为二分类问题（喜欢/不喜欢）
    # 评分 >= 4 视为喜欢（1），否则视为不喜欢（0）
    ratings['label'] = (ratings['rating'] >= 4).astype(int)
    
    # 合并数据
    data = pd.merge(ratings, users, on='user_id')
    data = pd.merge(data, movies, on='movie_id')
    
    # 提取电影类型
    data['genres'] = data['genres'].apply(lambda x: x.split('|')[0])  # 只使用第一个类型
    
    return data


def generate_negative_samples(data, neg_ratio=1):
    """
    生成负样本
    :param data: 原始数据
    :param neg_ratio: 负样本比例
    :return: 包含正负样本的数据
    """
    # 获取所有用户和电影
    all_users = data['user_id'].unique()
    all_movies = data['movie_id'].unique()
    
    # 创建用户已评分电影的字典
    user_rated_movies = defaultdict(set)
    for _, row in data.iterrows():
        user_rated_movies[row['user_id']].add(row['movie_id'])
    
    # 生成负样本
    neg_samples = []
    for user_id in all_users:
        rated_movies = user_rated_movies[user_id]
        # 随机选择用户未评分的电影作为负样本
        unrated_movies = list(set(all_movies) - rated_movies)
        if len(unrated_movies) < neg_ratio:
            continue
        
        # 随机选择neg_ratio个未评分电影
        neg_movie_ids = random.sample(unrated_movies, neg_ratio)
        
        # 获取用户信息
        user_info = data[data['user_id'] == user_id].iloc[0]
        
        for movie_id in neg_movie_ids:
            # 获取电影信息
            movie_info = data[data['movie_id'] == movie_id].iloc[0]
            
            # 创建负样本
            neg_sample = {
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': 0,
                'label': 0,
                'gender': user_info['gender'],
                'age': user_info['age'],
                'occupation': user_info['occupation'],
                'genres': movie_info['genres']
            }
            neg_samples.append(neg_sample)
    
    # 将负样本转换为DataFrame
    neg_df = pd.DataFrame(neg_samples)
    
    # 合并正样本和负样本
    pos_df = data[['user_id', 'movie_id', 'rating', 'label', 'gender', 'age', 'occupation', 'genres']]
    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    
    return combined_df


if __name__ == "__main__":
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    data = load_movielens_data(
        ratings_file='./data/ml-1m/ratings.dat',
        users_file='./data/ml-1m/users.dat',
        movies_file='./data/ml-1m/movies.dat'
    )
    
    # 查看数据
    print("原始数据示例:")
    print(data.head())
    
    # 查看正样本比例
    pos_count = (data['label'] == 1).sum()
    total_count = len(data)
    print(f"正样本数量: {pos_count} ({pos_count/total_count:.2%})")
    
    # 为了加快处理速度，我们只使用一部分数据
    print(f"原始数据大小: {len(data)}")
    data = data.sample(frac=0.1, random_state=seed)
    print(f"采样后数据大小: {len(data)}")
    
    # 生成负样本
    balanced_data = generate_negative_samples(data, neg_ratio=1)
    
    # 查看平衡后的数据
    pos_count_after = (balanced_data['label'] == 1).sum()
    total_count_after = len(balanced_data)
    print(f"平衡后 - 正样本数量: {pos_count_after} ({pos_count_after/total_count_after:.2%})")
    print(f"平衡后 - 总样本数量: {total_count_after}")
    
    # 定义特征
    categorical_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'genres']
    user_features = ['user_id', 'gender', 'age', 'occupation']
    item_features = ['movie_id', 'genres']
    
    # 标签编码
    for feat in categorical_features:
        lbe = LabelEncoder()
        balanced_data[feat] = lbe.fit_transform(balanced_data[feat])
    
    # 构建特征维度字典
    user_feature_dims = {feat: balanced_data[feat].nunique() for feat in user_features}
    item_feature_dims = {feat: balanced_data[feat].nunique() for feat in item_features}
    
    print("用户特征维度:", user_feature_dims)
    print("物品特征维度:", item_feature_dims)
    
    # 划分训练集、验证集和测试集
    train_data, test_data = train_test_split(balanced_data, test_size=0.2, random_state=seed, stratify=balanced_data['label'])
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=seed, stratify=train_data['label'])
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 创建数据加载器
    batch_size = 256
    train_loader = create_data_loader(train_data, user_features, item_features, 'label', batch_size=batch_size, shuffle=True)
    val_loader = create_data_loader(val_data, user_features, item_features, 'label', batch_size=batch_size, shuffle=False)
    test_loader = create_data_loader(test_data, user_features, item_features, 'label', batch_size=batch_size, shuffle=False)
    
    # 创建模型
    embedding_dim = 16
    model = DSSM(
        user_feature_dims=user_feature_dims,
        item_feature_dims=item_feature_dims,
        embedding_dim=embedding_dim,
        dnn_hidden_units=(128, 64, 32),
        dnn_activation='relu',
        dnn_dropout=0.2,
        dnn_use_bn=True,
        gamma=5,
        init_std=0.01
    ).to(device)
    
    # 打印模型结构
    print(model)
    
    # 定义损失函数和优化器
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练模型
    epochs = 20
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        early_stopping_patience=5,
        model_save_path='./saved_model/dssm_movielens.pt',
        scheduler=scheduler
    )
    
    # 绘制训练历史曲线
    plot_training_history(history)
    
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
    np.save('./saved_model/user_embedding_movielens.npy', user_embeddings)
    np.save('./saved_model/item_embedding_movielens.npy', item_embeddings)
    
    # 示例：为特定用户推荐电影
    def recommend_movies(user_id, top_k=10):
        """为指定用户推荐电影"""
        # 获取用户嵌入
        user_indices = test_data[test_data['user_id'] == user_id].index
        if len(user_indices) == 0:
            print(f"用户ID {user_id} 在测试集中不存在")
            return []
            
        user_idx = user_indices[0] - test_data.index[0]  # 调整为相对于测试集的索引
        if user_idx >= len(user_embeddings):
            print(f"用户索引 {user_idx} 超出嵌入范围 {len(user_embeddings)}")
            return []
            
        user_emb = user_embeddings[user_idx]
        
        # 计算用户嵌入与所有电影嵌入的余弦相似度
        similarities = []
        movie_ids = []
        movie_indices = {}
        
        # 为每个电影ID创建索引映射
        for i, (idx, row) in enumerate(test_data.drop_duplicates('movie_id').iterrows()):
            movie_indices[row['movie_id']] = i
        
        # 计算相似度
        for movie_id, idx in movie_indices.items():
            if idx >= len(item_embeddings):
                continue
                
            movie_emb = item_embeddings[idx]
            
            # 计算余弦相似度
            similarity = np.dot(user_emb, movie_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(movie_emb))
            similarities.append(similarity)
            movie_ids.append(movie_id)
        
        # 获取相似度最高的top_k个电影
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_movie_ids = [movie_ids[i] for i in top_indices]
        top_similarities = [similarities[i] for i in top_indices]
        
        # 获取电影标题
        movie_id_to_title = dict(zip(data['movie_id'], data['title']))
        top_movie_titles = [movie_id_to_title.get(movie_id, f"Movie {movie_id}") for movie_id in top_movie_ids]
        
        return list(zip(top_movie_titles, top_similarities))
    
    # 为随机用户推荐电影
    # 确保从测试集中选择用户ID
    test_user_ids = test_data['user_id'].unique()
    if len(test_user_ids) > 0:
        random_user_id = random.choice(test_user_ids)
        recommendations = recommend_movies(random_user_id, top_k=10)
        
        print(f"\n为用户 {random_user_id} 的电影推荐:")
        for i, (title, score) in enumerate(recommendations, 1):
            print(f"{i}. {title} (相似度: {score:.4f})")
    else:
        print("测试集中没有可用的用户ID")
    
    print("\nDone!") 