import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('./train.csv')
for row_index, row in train_data.iterrows():
    count = 0
    for column_index, val in row.items():
        if val == -1:
            count += 1
    train_data.at[row_index, 'missing_values'] = count
print(train_data)
y = train_data['target']
X = train_data.drop(['id','target'], axis=1)

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2019)

cat_features = [row for row in X if 'cat' in row]
print(cat_features)
def one_hot_encoding(train, test, cat_features):
    '''Function to one-hot-encode categorical features'''
    temp = pd.concat([train, test])
    temp = pd.get_dummies(temp, columns = cat_features)
    train = temp.iloc[:train.shape[0],:]
    test = temp.iloc[train.shape[0]:, :]
    return train, test
X_train_pd, X_cv_pd = one_hot_encoding(X_train, X_cv, cat_features)

scaler = StandardScaler()
scaler.fit(X_train_pd)
X_train = scaler.transform(X_train_pd)
X_cv = scaler.transform(X_cv_pd)
print(X_train)

def compute_gini(Y, Y_cap):
  temp = np.asarray(np.c_[Y, Y_cap, np.arange(len(Y))], dtype=np.float)
  temp1 = temp[np.lexsort((temp[:,2],-1*temp[:,1]))]
  loss = temp1[:,0].sum()
  gini = temp1[:,0].cumsum().sum() / loss

  gini -=(len(Y) + 1) / 2
  return gini / len(Y)

def normalized_gini(Y, Y_cap):
  return compute_gini(Y, Y_cap) / compute_gini(Y, Y)

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1)
xgb_model.fit(X_train, y_train)
train_pred = xgb_model.predict_proba(X_train)[:,1]
test_pred = xgb_model.predict_proba(X_cv)[:,1]

train_xgb_score = []
test_xgb_score = []
        
train_xgb_score.append(roc_auc_score(y_train, train_pred))
test_xgb_score.append(roc_auc_score(y_cv, test_pred))
gini_xgb1 = normalized_gini(y_cv, test_pred)
print(gini_xgb1)

xgb_model.save_model("best_model.json")