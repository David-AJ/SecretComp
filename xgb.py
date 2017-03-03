import numpy as np 
import xgboost as xgb
from sklearn import cross_validation
import Evaluation as e

train_X = np.load("train_X.npy")
train_Y = np.load("train_Y.npy")


params={'booster':'gbtree',
	    'objective': 'reg:linear',
	    'eval_metric':'rmse',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':10,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    }
def en(prediction_list,actual_list):
	result = 0.0
	label = actual_list.get_label()
	num = len(prediction_list)
	for i in range(num):
		if prediction_list[i] + label[i] != 0:
		    result += abs((prediction_list[i] - label[i])/float(prediction_list[i] + label[i]))
	return 'loss',result / num

loss = []
k_fold = cross_validation.KFold(n=len(train_Y), n_folds=2)
for train,test in k_fold:
	dataset = xgb.DMatrix(train_X[train],label=train_Y[train])
	watchlist = [(dataset,'train')]
	model = xgb.train(params,dataset,num_boost_round=35,evals=watchlist)
	test_set = xgb.DMatrix(train_X[test])
	predict = model.predict(test_set)
	loss.append(e.evalutaion(train_Y[test],predict))
print np.array(loss).mean()
