import numpy as np 
from sklearn import cross_validation as cv
from sklearn.ensemble import GradientBoostingRegressor as GBR
import Evaluation as e

train_X = np.load("train_X.npy")
train_Y = np.load("train_Y.npy")

k_fold = cv.KFold(n=len(train_Y), n_folds=3)
loss=[]
for train,test in k_fold:
	print "fitting GBRT..."
	model =  GBR(n_estimators=100, learning_rate=0.01,max_depth=5, random_state=0,loss='huber').fit(train_X[train],train_Y[train])
	print "predicting..."
	predict = model.predict(train_X[test])
	loss.append(e.evalutaion(train_Y[test],predict))
print np.array(loss).mean()
