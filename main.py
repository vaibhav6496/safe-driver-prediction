import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler

st.header("Safe driver prediction app")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("train.csv")

best_xgboost_model = xgb.XGBClassifier()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show train dataframe sample'):
    data[:10]

st.subheader("Please input the features")

missing_values = 0
input_list = []
for col_names in data.columns:
    if col_names != 'id' and col_names != 'target':
        if 'cat' in col_names:
            user_ip = int(st.number_input(col_names, step=1, format='%d', value=1))
        else:
            user_ip = st.number_input(col_names)
        input_list.append(user_ip)
        if (user_ip == -1):
            missing_values += 1

st.number_input('missing_values', value=missing_values)
input_list.append(missing_values)

input_list.insert(0, 0)
input_list.insert(0, 0)

data['missing_values'] = missing_values
cols = data.columns
# print(cols)
# print(data.shape)

#cols.append('missing_values')

X = pd.DataFrame.from_dict({'row': input_list}, orient='index',
                       columns=cols)

X = X.drop(['id','target'], axis=1)
data = data.drop(['id','target'], axis=1)

cat_features = [row for row in X if 'cat' in row]
print(X)
print(data.head(1))
def one_hot_encoding(train, input, cat_features):
    '''Function to one-hot-encode categorical features'''
    temp = pd.concat([train, input])
    temp = pd.get_dummies(temp, columns = cat_features)
    print(temp.shape)
    train = temp.iloc[:train.shape[0],:]
    input = temp.iloc[train.shape[0]:, :]
    return train, input

X_train_pd, X = one_hot_encoding(data, X, cat_features)
print(X.shape)

scaler = StandardScaler()
scaler.fit(X_train_pd)
X_train = scaler.transform(X_train_pd)

if st.button('Make Prediction'):
    inputs = np.expand_dims(X_train[0], 0)
    prediction = best_xgboost_model.predict(inputs)
    if (prediction[0] == 0):
        st.write("Driver will not file the claim")
    else:
        st.write("Driver will file the claim")
