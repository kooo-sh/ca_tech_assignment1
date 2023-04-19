# ライブラリのインポート
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# データのダウンロード
cov = fetch_covtype()
X_array = cov.data
t_array = cov.target

# DataFrameの作成
df = DataFrame(X_array, columns = cov.feature_names).assign(Ty=np.array(t_array))

# 事前にRandomForestで抽出した関係性の高い要素を14個抽出する
X_train = df[['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Wilderness_Area_2', 'Soil_Type_31', 'Aspect', 'Hillshade_9am', 'Hillshade_3pm', 'Hillshade_Noon', 'Wilderness_Area_0', 'Slope', 'Soil_Type_38']]
t_train = df['Ty']

# train&val, testにデータセットを分ける
x_train_val, x_test, t_train_val, t_test = train_test_split(X_train, t_train, test_size=0.2, random_state=1) 

# 標準化
std_scaler = StandardScaler()
std_scaler.fit(x_train_val)
x_train_val_std = std_scaler.transform(x_train_val)
x_test_std = std_scaler.transform(x_test)

# RandomForestを分類器に設定
param_grid = [{
    'n_estimators':[10, 15]
}]
rf = RFC(max_features='auto')

# Cross Validation
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
tuned_model = GridSearchCV(estimator=rf, 
                           param_grid=param_grid, 
                           cv=skf, return_train_score=True, scoring='accuracy')
tuned_model.fit(x_train_val_std, t_train_val)

# testでの精度を計算して出力
print('Test score: {}'.format(tuned_model.score(x_test_std, t_test)))