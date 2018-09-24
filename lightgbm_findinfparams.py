print("starting...")
from utils import *

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import lightgbm as lgb
import pandas as pd
import os
import numpy as np

#%matplotlib inline
data_path = 'data/'
data_path = 'dataframes/'
seed=1204

submission_path=data_path+'submission/'
fold_path = 'fold_data/'


cv_loss_list=[]
n_iteration_list=[]
def score(params):
    print("Training with params: ")
    print(params)
    cv_losses=[]
    cv_iteration=[]
    for (train_idx,val_idx) in cv:
        print("Training Indexes: "+str(train_idx[0])+":"+str(train_idx[-1]))
        print("Validation Indexes: "+str(val_idx[0])+":"+str(val_idx[-1]))
        cv_train = X.iloc[train_idx]
        cv_val = X.iloc[val_idx]
        cv_y_train = y[train_idx]
        cv_y_val = y[val_idx]
        lgb_model = lgb.train(params, lgb.Dataset(cv_train, label=cv_y_train), 2000, 
                          lgb.Dataset(cv_val, label=cv_y_val), verbose_eval=False, 
                          early_stopping_rounds=100, categorical_feature=(0,1,6))
       
        train_pred = lgb_model.predict(cv_train,lgb_model.best_iteration+1)
        val_pred = lgb_model.predict(cv_val,lgb_model.best_iteration+1)
        
        val_loss = root_mean_squared_error(cv_y_val,val_pred)
        train_loss = root_mean_squared_error(cv_y_train,train_pred)
        print('Train RMSE: {}. Val RMSE: {}'.format(train_loss,val_loss))
        print('Best iteration: {}'.format(lgb_model.best_iteration))
        cv_losses.append(val_loss)
        cv_iteration.append(lgb_model.best_iteration)
    print('6 fold results: {}'.format(cv_losses))
    cv_loss_list.append(cv_losses)
    n_iteration_list.append(cv_iteration)
    
    mean_cv_loss = np.mean(cv_losses)
    print('Average iterations: {}'.format(np.mean(cv_iteration)))
    print("Mean Cross Validation RMSE: {}\n".format(mean_cv_loss))
    return {'loss': mean_cv_loss, 'status': STATUS_OK}

def get_cv_idxs(df,start,end):
    result=[]
    for i in range(start,end+1):
        dates = df.date_block_num
        train_idx = np.array(df.loc[dates <i].index)
        val_idx = np.array(df.loc[dates == i].index)
        result.append((train_idx,val_idx))
    return np.array(result)

def optimize(space,seed=seed,max_evals=5):
    
    best = fmin(score, space, algo=tpe.suggest, 
        # trials=trials, 
        max_evals=max_evals)
    return best

print("loading training data...")
data_path = 'data/'
dataframes_path = 'dataframes/'

#get DataFrames
all_data1 = pd.read_pickle(os.path.join(dataframes_path, 'saleswith12lags1.pickle'))
all_data2 = pd.read_pickle(os.path.join(dataframes_path, 'saleswith12lags2.pickle'))
all_data3 = pd.read_pickle(os.path.join(dataframes_path, 'saleswith12lags3.pickle'))

all_data=pd.concat([all_data1,all_data2,all_data3], ignore_index=True)
del all_data1,all_data2,all_data3

all_data = all_data.reset_index().drop('index',axis=1)

print("spliting training data into X,y...")
X = all_data.drop(['item_cnt_month','date',],axis=1)
y = all_data.item_cnt_month.values
X.drop('date_block_num',axis=1,inplace=True)

print("Creating Cross Validation Set...")
cv = get_cv_idxs(all_data,28,33)

print("Defining Hyperparameters...")
space = {
#     'max_depth': hp.choice('max_depth', np.arange(3, 15, dtype=int)),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'min_data_in_leaf': hp.choice('min_data_in_leaf',np.arange(5, 30,1, dtype=int)),
    'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
    'seed':seed,
    'objective': 'regression',
    'metric':'rmse',
    'boosting':'gdbt',
}
print("Start looking for best parameters...")
best_hyperparams = optimize(space,max_evals=5)
print("The best hyperparameters are: ")
print(best_hyperparams)
