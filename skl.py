import numpy as np 
from sklearn import cross_validation as cv
from sklearn.ensemble import GradientBoostingRegressor as GBR
import Evaluation as e

train_X = np.load("train_X2.npy")
train_Y = np.load("train_Y2.npy")

k_fold = cv.KFold(n=len(train_Y), n_folds=10)
loss=[]
for train,test in k_fold:
	print "fitting GBRT..."
	model =  GBR(n_estimators=450, learning_rate=0.01,max_depth=10, random_state=0,loss='huber').fit(train_X[train],train_Y[train])
	print "predicting..."
	predict = model.predict(train_X[test])
	core = e.evalutaion(train_Y[test],predict)
	loss.append(core)
	print core
	break
print np.array(loss).mean()
