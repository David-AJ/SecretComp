#-*-coding:utf-8-*- 
import pandas as pd 
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
# 如果csv文件无列标题,即第一行为数据
# 参数加 header = None声明无标题 
# 或加 names = [str1,str2,str3,...]自定义标题

# user_view
view = pd.read_csv('user_view.txt',names=['uid','sid','date'])
v2 = pd.read_csv('extra_user_view.txt',names=['uid','sid','date'])

# 合并两个数据
view = view.append(v2,ignore_index=True)
del v2
# # 对字符串数据切片（取0到-9）,将日期精确到天
# view.date = view.date.str.slice(0,-9)
# 按date排序
view = view.sort_values(by='date')
view.to_csv('view_sorted.txt',index=False)
# view per day
view.date = view.date.str[0:-9]
vpd = view.groupby(['sid','date']).size()
# view windows sum
for i in vpd.index.levels[0]:
	p = vpd.loc[i]
	idx = pd.MultiIndex.from_arrays([np.random.randint(i,i+1,len(vpd.loc[i])),vpd.loc[i].index],names=('sid','date'))
	if i==1:
		shovpd = pd.concat([pd.rolling_sum(p,3),pd.rolling_sum(p,5),pd.rolling_sum(p,7),\
		pd.rolling_sum(p,9),pd.rolling_sum(p,11),pd.rolling_sum(p,13),\
		pd.rolling_sum(p,15)],axis=1)
		shovpd.index = idx 
	else:
		temp = pd.concat([pd.rolling_sum(p,3),pd.rolling_sum(p,5),pd.rolling_sum(p,7),\
    	pd.rolling_sum(p,9),pd.rolling_sum(p,11),pd.rolling_sum(p,13),\
    	pd.rolling_sum(p,15)],axis=1)
    	temp.index = idx
		shovpd = shovpd.append(temp)
# view prediction using VAR model
shovpd = shovpd.dropna()


# shop
shop = pd.read_csv('shop_info.txt',names=['sid','city','locid','per_pay','score','cmt','level','c1','c2','c3'])
shop.score = shop.score.fillna(-1)
shop.cmt = shop.cmt.fillna(-1)
shop.c1 = shop.c1.fillna(u'无')
shop.c2 = shop.c2.fillna(u'无')
shop.c3 = shop.c3.fillna(u'无')

city = shop.city.value_counts().index.tolist()
citymap = {city[i]:i for i in range(len(city))}

c1 = shop.c1.value_counts().index.tolist()
c1map = {c1[i]:i for i in range(len(c1))}

c2 = shop.c2.value_counts().index.tolist()
c2map = {c2[i]:i for i in range(len(c2))}

c3 = shop.c3.value_counts().index.tolist()
c3map = {c3[i]:i for i in range(len(c3))}


shop.city = shop.apply(lambda x: citymap[x.city],axis=1)
shop.c1 = shop.apply(lambda x: c1map[x.c1],axis=1)
shop.c2 = shop.apply(lambda x: c2map[x.c2],axis=1)
shop.c3 = shop.apply(lambda x: c3map[x.c3],axis=1)
shop = shop.drop('sid',axis=1)
shop = shop.astype(int).as_matrix()
np.save('shop',shop)


# user_pay
pay = pd.read_csv('user_pay.txt',names=['uid','sid','date'])
pay = pay.sort_values(by='date')
pay.to_csv('pay_sorted.txt',index=False)

# 对字符串数据切片（取0到-9）,将日期精确到天
pay.date = pay.date.str.slice(0,-9)

# pay per day 以sid和date划分group，然后计算大小
ppd = pay.groupby(['sid','date']).size()
ppd = ppd.astype(int) 
ppd.to_csv('pay_per_day.txt')
p_date = ppd.index.levels[1].tolist()
# # 下次读取时，由于第0,1列为index
# # 加参数index_col=[0,1]
# ppd = pd.read_csv('pay_per_day2.txt',index_col=[0,1],names=['sid','date','pay'])
for i in ppd.index.levels[0]:
	p = ppd.loc[i]
	idx = pd.MultiIndex.from_arrays([np.random.randint(i,i+1,len(ppd.loc[i])),ppd.loc[i].index],names=('sid','date'))
	if i==1:
		shoppd = pd.concat([pd.rolling_mean(p,3).shift(1),pd.rolling_mean(p,5).shift(1),pd.rolling_mean(p,7).shift(1),\
		pd.rolling_mean(p,9).shift(1),pd.rolling_mean(p,11).shift(1),pd.rolling_mean(p,13).shift(1),\
		pd.rolling_mean(p,15).shift(1)],axis=1)
		shoppd.index = idx 
    else:
		temp = pd.concat([pd.rolling_mean(p,3).shift(1),pd.rolling_mean(p,5).shift(1),pd.rolling_mean(p,7).shift(1),\
    	pd.rolling_mean(p,9).shift(1),pd.rolling_mean(p,11).shift(1),pd.rolling_mean(p,13).shift(1),\
    	pd.rolling_mean(p,15).shift(1)],axis=1)
    	temp.index = idx
    	shoppd = shoppd.append(temp)

shoppd = shoppd.dropna()



# holiday
h = pd.read_csv('holiday.csv')
h.date = h.date.astype(str)
h.date = h.date.str.slice(0,4)+'-'+h.date.str.slice(4,6)+'-'+h.date.str.slice(6,8)
# find train date
h_train = h[h.date.isin(p_date)]
# predict date
newdate = np.array([pd.date_range('11/1/2016',periods=14,freq='D').astype(str),[0,0,0,0,1,1,0,0,0,0,0,1,1,0]])
h_predict = pd.DataFrame(newdate.T,columns=['date','holiday'])
holiday = h_train.append(h_predict)
holiday.index = holiday.date
holiday = holiday.drop('date',axis=1)
holiday = holiday.astype(int) 
holiday = holiday.replace(2,1)
# 0->工作日00000 1->放假00001 2->第一工作日00010 3->最后工作日00100 4->第一放假01000 5->最后放假10000
h = holiday.as_matrix()
for i in range(len(h)):
	if h[i-1]&i<len(h):
		if h[i]==0:
			if h[i-1]==1:
				h[i]=2
				h[i-1]=5
			elif h[i+1]==1:
				h[i]=3
				h[i+1]=4
enc = OneHotEncoder()
holiday.holiday = enc.fit_transform(h).toarray().tolist()
holiday.to_csv('h_day.csv')

# weather
w = pd.read_csv('weather_all.csv',names=['city','date','high','low','weather','wind','windl'])
w = w.drop('wind',axis=1)
w.date = w.date.astype(str)
# 筛选出与测试城市日期有关的数据  p_date缺少预测日期
w = w[w.city.isin(citymap.keys())&w.date.isin(p_date)]
w = w.fillna(u'无')
w.city = w.apply(lambda x: citymap[x.city],axis=1)

w.weather = LabelBinarizer().fit_transform(w.weather).tolist()

windl = w.windl.value_counts().index.tolist()
windlmap = {windl[i]:i for i in range(len(windl))}


w.windl = w.apply(lambda x: windlmap[x.windl],axis=1)

w.index = [w.city,w.date]
w = w.drop(['city','date'],axis=1)
w.to_csv('weather.csv')


# wirte maps 
f = open('maps.pkl','w')
f2 = open('maps.txt','w')
for i in [citymap,c1map,c2map,c3map,weathermap,windlmap]:
	f2.writelines("map: \n")
	for j in i:
		f2.writelines(j.decode('utf8')+" : "+str(i[j])+'\n')
	pickle.dump(i,f)
f.close()
f2.close()



def createData(ppd,shovpd,shoppd):
	shoppd = shoppd[shoppd.index.isin(shovpd.index)]
	X = []
	Y = []
	for sid,date in shoppd.index:
		temp = [i for i in shop[sid-1]]
		# temp = [float(sid)]
		weather = w.loc[shop[sid-1][0],date].as_matrix()
		temp.extend(weather[0:2])
		temp.extend(weather[2])
		temp.extend([weather[3]])
		temp.extend(holiday.loc[date].as_matrix()[0])
		temp.extend(shovpd.loc[sid,date].as_matrix())
		temp.extend(shoppd.loc[sid,date].as_matrix())
		X.append(temp)
		Y.append(ppd.loc[sid,date].pay)
	return np.array(X),np.array(Y)

train_X,train_Y=createData(ppd,shovpd,shoppd)
clf = Lasso()
clf.fit(train_X,train_Y)
select = np.array(clf.coef_)
train_X = train_X.T[np.where(select!=0)].T
# np.save('train_X',train_X)
np.save('train_X2',train_X)
# np.save('train_Y',train_Y)
np.save('train_Y2',train_Y)

# [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in kfold] 

