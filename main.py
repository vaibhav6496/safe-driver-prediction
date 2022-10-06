import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load

st.header("Safe driver prediction app")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("head_data.csv", index_col=0)

best_xgboost_model = xgb.XGBClassifier()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show train dataframe sample'):
    data

st.subheader("Please input the features")

missing_values = 0
input_list = []
for col_names in data.columns:
    if col_names != 'target':
        if 'cat' in col_names:
            user_ip = int(st.number_input(col_names, step=1, format='%d', value=1))
        else:
            user_ip = st.number_input(col_names)
        input_list.append(user_ip)
        if (user_ip == -1):
            missing_values += 1

st.number_input('missing_values', value=missing_values)
input_list.append(missing_values)

data['missing_values'] = missing_values
data = data.drop(['target'], axis=1)
cols = data.columns

X = pd.DataFrame.from_dict({'row': input_list}, orient='index',
                       columns=cols)

cat_features = [row for row in X if 'cat' in row]

ohe = load('ohe.sav')

array_hot_encoded = ohe.transform(X[cat_features])
data_hot_encoded = pd.DataFrame(array_hot_encoded.toarray())

data_other_cols = X.drop(columns=cat_features, axis = 1)

data_hot_encoded.reset_index(drop=True, inplace=True)
data_other_cols.reset_index(drop=True, inplace=True)

X_train_pd = pd.concat([data_hot_encoded, data_other_cols], axis=1)


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
