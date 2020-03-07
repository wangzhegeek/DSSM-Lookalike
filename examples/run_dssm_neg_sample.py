import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

from models.dssm import DSSM
from deepctr.inputs import SparseFeat, get_feature_names
from tensorflow.python.keras.models import Model
from utils import Negative_Sample


if __name__ == "__main__":
    data = pd.read_csv('../data/train_sample.csv')

    data['label'] = data['label'].replace(-1, 0)
    print(data.head())

    sparse_features = ['aid', 'uid']
    # dense_features = []

    user_features = ['aid']
    item_features = ['uid']

    data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # Negative_Sample
    data = Negative_Sample(data, 'aid', 'uid', 'label', 10, method_id=2)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    user_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=20)
                           for i, feat in enumerate(user_features)]
    item_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=20)
                           for i, feat in enumerate(item_features)]

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in sparse_features}
    test_model_input = {name: test[name] for name in sparse_features}

    # 4.Define Model,train,predict and evaluate
    model = DSSM(user_feature_columns, item_feature_columns, task='binary')
    model.summary()
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2,)
    model.save_weights('../saved_model/dssm.ckpt')
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    user_embedding_model = Model(inputs=model.input, outputs=model.get_layer("user_embedding").output)
    item_embedding_model = Model(inputs=model.input, outputs=model.get_layer("item_embedding").output)
    user_embedding = user_embedding_model.predict(test_model_input)
    item_embedding = item_embedding_model.predict(test_model_input)

    print("user embedding shape: ", user_embedding.shape)
    print("item embedding shape: ", item_embedding.shape)

    np.save('../saved_model/user_embedding.npy', user_embedding)
    np.save('../saved_model/user_embedding.npy', item_embedding)
