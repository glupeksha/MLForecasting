import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,make_scorer
from math import sqrt

def downcast_dtypes(df):
	'''
	Changes column types in the dataframe: 
		
		`float64` type to `float32`
		`int64`   type to `int32`
	'''

	# Select columns to downcast
	float_cols = [c for c in df if df[c].dtype == "float64"]
	int_cols =   [c for c in df if df[c].dtype == "int64"]

	# Downcast
	df[float_cols] = df[float_cols].astype(np.float32)
	df[int_cols]   = df[int_cols].astype(np.int32)

	return df

def get_all_data(data_path,filename):
	all_data = pd.read_pickle(data_path + filename)
	all_data = downcast_dtypes(all_data)
	all_data = all_data.reset_index().drop('index',axis=1)
	return all_data

def get_cv_idxs(df,start,end):
	result=[]
	for i in range(start,end+1):
		dates = df.date_block_num
		train_idx = np.array(df.loc[dates <i].index)
		val_idx = np.array(df.loc[dates == i].index)
		result.append((train_idx,val_idx))
	return np.array(result)

def get_X_y(df,end,clip=20):
	# don't drop date_block_num
	df = df.loc[df.date_block_num <= end]
	cols_to_drop=['item_cnt_month'] + df.columns.values[6:12].tolist()
	#print(cols_to_drop)
	y = np.clip(df.item_cnt_month.values,0,clip)
    #np.clip(df.item_cnt_month.values,0,clip)
	X = df.drop(cols_to_drop,axis=1)
	return X,y

def root_mean_squared_error(truth,pred):
	return sqrt(mean_squared_error(truth,pred))