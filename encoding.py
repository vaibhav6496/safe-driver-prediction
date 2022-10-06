from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from joblib import dump

data = pd.read_csv("train.csv", index_col=0)
le = LabelEncoder()

cat_features = [row for row in data if 'cat' in row]
print(cat_features)

data[cat_features] = data[cat_features].apply(lambda col: le.fit_transform(col)) 
ohe = OneHotEncoder()

ohe = ohe.fit(data[cat_features])

dump(ohe, 'ohe.sav')


