import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import pickle

data = pd.read_csv("2017RankingEngg.csv")
data0 = pd.read_csv("2016RankingEngg.csv")
data2 = pd.read_csv("2018RankingEngg.csv")
data3 = pd.read_csv("2019RankingEngg1.csv")
data4 = pd.read_csv("2020RankingEngg.csv")


c0 = np.array([data0.iloc[0]])
for i in range(99):
    x = np.array(data0.iloc[i+1])
    c0 = np.concatenate((c0,[x]),axis=0)

c1 = np.array([data.iloc[0]])
for i in range(data.shape[0]-1):
    x = np.array(data.iloc[i+1])
    c1 = np.concatenate((c1,[x]),axis=0)
    
c2 = np.array([data2.iloc[0]])
for i in range(99):
    x = np.array(data2.iloc[i+1])
    c2 = np.concatenate((c2,[x]),axis=0)
    
c3 = np.array([data3.iloc[0]])
for i in range(168):
    x = np.array(data3.iloc[i+1])
    c3 = np.concatenate((c3,[x]),axis=0)

c4 = np.array([data4.iloc[0]])
for i in range(data4.shape[0]-1):
    x = np.array(data4.iloc[i+1])
    c4 = np.concatenate((c4,[x]),axis=0)
    
averaged_data = np.zeros((169,5))
for i in range(100):
    for j in range(5):
        averaged_data[i,j] = (c0[i,j]*100/max(c0[:,j])+c1[i,j]*100/max(c1[:,j])+c2[i,j]*100/max(c2[:,j])+c3[i,j]*100/max(c3[:,j]))/4.0

for i in range(100,169):
    for j in range(5):
        averaged_data[i,j] = c3[i,j]*100/max(c3[:,j])
        
d = np.array([[1]])
for i in range(168):
    d = np.concatenate((d,[[i+1]]),axis=0)
    
sc = np.zeros((169,1))
for i in range(169):
    sc[i,0] = 0.3*averaged_data[i,0]+0.3*averaged_data[i,1]+0.2*averaged_data[i,2]+0.1*averaged_data[i,3]+0.1*averaged_data[i,4]

test = np.zeros((169,1))
c_test = c4*100/[max(c3[:,0]),max(c3[:,1]),max(c3[:,2]),max(c3[:,3]),max(c3[:,4])]
for i in range(169):
    test[i,0] = 0.3*c_test[i,0]+0.3*c_test[i,1]+0.2*c_test[i,2]+0.1*c_test[i,3]+0.1*c_test[i,4]


regr = linear_model.LinearRegression()
poly = PolynomialFeatures(3,interaction_only=False)
x = poly.fit_transform(sc)
regr.fit(x,d)

file_name = "rank_model.pkl"
file_name1="polynomial_transform.pkl"

pickle.dump(regr,open(file_name,"wb"))
pickle.dump(poly,open(file_name1,"wb"))