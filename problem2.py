# ライブラリのインポート
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# データのダウンロード
cov = fetch_covtype()
X_array = cov.data
t_array = cov.target

# DataFrameの作成
df = DataFrame(X_array, columns = cov.feature_names).assign(Ty=np.array(t_array))

# 事前にRandomForestで抽出した関係性の高い要素を14個抽出する
X_train = df[['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Wilderness_Area_2', 'Soil_Type_31', 'Aspect', 'Hillshade_9am', 'Hillshade_3pm', 'Hillshade_Noon', 'Wilderness_Area_0', 'Slope', 'Soil_Type_38']]
t_train = df['Ty']
X_train_all = df.drop('Ty', axis=1)

# train&val, testにデータセットを分ける
x_train_val, x_test, t_train_val, t_test = train_test_split(X_train, t_train, test_size=0.2, random_state=1) 
x_train_val_all, x_test_all, t_train_val_all, t_test_all = train_test_split(X_train, t_train, test_size=0.2, random_state=1) 

# 標準化
std_scaler = StandardScaler()
std_scaler.fit(x_train_val)
x_train_val_std = std_scaler.transform(x_train_val)
x_test_std = std_scaler.transform(x_test)

std_scaler_all = StandardScaler()
std_scaler_all.fit(x_train_val_all)
x_train_val_std_all = std_scaler_all.transform(x_train_val_all)
x_test_std_all = std_scaler_all.transform(x_test_all)

# RandomForest, LinearSVC, DecisionTreeをモデルに設定
param_grid_rf = [{
    'n_estimators':[10, 15]
}]
rf = RFC(max_features='auto')

param_grid_sv = [{
    'penalty':['l1', 'l2']
}]
sv = LinearSVC()

param_grid_dt = [{
    'max_features':['auto', 'sqrt']
}]
dt = DecisionTreeClassifier()

# 特徴量については全ての要素と主要な14個の要素で比較する
feature_engineering = ['all_factors', 'selected_14factors']
models = {'RandomForest':[rf, param_grid_rf], 'LinearSV':[sv, param_grid_sv], 'DescisionTree':[dt, param_grid_dt]}

# モデル、特徴量タイプごとの交差検証
for model_k, model_v in models.items():
  for feature_type in feature_engineering:
    print('Model : ', model_k)
    print('Feature type : ', feature_type)

    skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)
    tuned_model = GridSearchCV(estimator=model_v[0], 
                              param_grid=model_v[1], 
                              cv=skf, return_train_score=True, scoring='accuracy')    
    if feature_type == 'all_factors':
      tuned_model.fit(x_train_val_std_all, t_train_val_all)
      print('Test score: {}'.format(tuned_model.score(x_test_std_all, t_test_all)))
    else:
      tuned_model.fit(x_train_val_std, t_train_val)
      print('Test score: {}'.format(tuned_model.score(x_test_std, t_test))) 

    # df = pd.DataFrame(tuned_model.cv_results_).T 
    # filename = feature_type+model_k+'.csv'
    # df.to_csv(filename) 