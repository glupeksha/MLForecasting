print("starting...")

import lightgbm as lgb
import pandas as pd
import os
import matplotlib.pyplot as plt
print("loading training data...")
data_path = 'data/'
dataframes_path = 'dataframes/'

#get DataFrames
all_data1 = pd.read_pickle(os.path.join(dataframes_path, 'saleswith12lags1.pickle'))
#all_data2 = pd.read_pickle(os.path.join(dataframes_path, 'saleswith12lags2.pickle'))
#all_data3 = pd.read_pickle(os.path.join(dataframes_path, 'saleswith12lags3.pickle'))

all_data=pd.concat([all_data1], ignore_index=True)
del all_data1#,all_data2,all_data3

all_data = all_data.reset_index().drop('index',axis=1)

print("spliting training data into X,y...")
X = all_data.drop(['item_cnt_month','date',],axis=1)
y = all_data.item_cnt_month.values
X.drop('date_block_num',axis=1,inplace=True)

lgb_params = {
    'objective':'regression',
    'metric':'rmse',
    'learning_rate':0.03,
    'colsample_bytree':0.75,
    'min_data_in_leaf':64,
    'subsample':0.75,
    'bagging_seed':64,
    'num_leaves':64,
    'bagging_freq':1,
    'seed':1204
}

evals_result = {}
print("Start training the model..")
lgb_model = lgb.train(lgb_params,lgb.Dataset(X,label=y),verbose_eval=10,evals_result=evals_result)
print("plot matricx recorded during training..")
ax=lgb.plot_matric(evals_result,metric='11')
plt.show()

print("plot feature importance..")
ax=lgb.plot_matric(gbm,max_num_features=10)
plt.show()

