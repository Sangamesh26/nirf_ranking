import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import pickle


data1 = pd.read_csv("2017_univ_RankingEngg.csv")
data2 = pd.read_csv("2018_univ_RankingEngg.csv")
data3 = pd.read_csv("2019_univ_RankingEngg.csv")
data4 = pd.read_csv("2020_univ_RankingEngg.csv")

c1 = np.array([data1.iloc[0]])
for i in range(data1.shape[0]-1):
    x = np.array(data1.iloc[i+1])
    c1 = np.concatenate((c1,[x]),axis=0)
    
c2 = np.array([data2.iloc[0]])
for i in range(99):
    x = np.array(data2.iloc[i+1])
    c2 = np.concatenate((c2,[x]),axis=0)
    
c3 = np.array([data3.iloc[0]])
for i in range(99):
    x = np.array(data3.iloc[i+1])
    c3 = np.concatenate((c3,[x]),axis=0)

c4 = np.array([data4.iloc[0]])
for i in range(data4.shape[0]-1):
    x = np.array(data4.iloc[i+1])
    c4 = np.concatenate((c4,[x]),axis=0)

data_c1 = np.array([0.3*c1[:,0]+0.3*c1[:,1]+0.2*c1[:,2]+0.1*c1[:,3]+0.1*c1[:,4]]).T
data_c2 = np.array([0.3*c2[:,0]+0.3*c2[:,1]+0.2*c2[:,2]+0.1*c2[:,3]+0.1*c2[:,4]]).T
data_c3 = np.array([0.3*c3[:,0]+0.3*c3[:,1]+0.2*c3[:,2]+0.1*c3[:,3]+0.1*c3[:,4]]).T

c3c2 = np.subtract(data_c3,data_c2)
c2c1 = np.subtract(data_c2,data_c1)
lm = np.subtract(c3c2,c2c1)/2
addable_avg = (lm**2)**0.5

d = np.array([[1]])
for i in range(99):
    d = np.concatenate((d,[[i+2]]),axis=0)
    
test = np.array([0.3*c4[:,0]+0.3*c4[:,1]+0.2*c4[:,2]+0.1*c4[:,3]+0.1*c4[:,4]]).T

dataset = np.zeros((94,6,1))
data_next = np.zeros(94,)

for i in range(dataset.shape[0]):
    l = np.array((data_c3[i:i+dataset.shape[1]]+addable_avg[i:i+dataset.shape[1]]))
    l = l.reshape(dataset.shape[1],1)
    dataset[i] = l

for i in range(dataset.shape[0]):
    data_next[i] = data_c3[i+dataset.shape[1]]
    
"""model = tf.keras.models.Sequential([
  #tf.keras.layers.Lambda(lambda x:input_shape=[None]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True,input_shape=(dataset.shape[1],1))),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dense(1)])

#optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss="mean_absolute_error",
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=["mae"])

history = model.fit(x = dataset,y = data_next, epochs=1000)"""

"""result = data_c3"""

"""di = np.zeros((120,1))
for i in range(120):
    di[i] = i+1
    
tes = dataset[-1]
for i in range(20):
    pred = model.predict(tes.reshape(1,dataset.shape[1],1))
    result=np.append(result,pred)
    tes = np.append(tes,pred)
    tes = tes[-dataset.shape[1]:]
    
result = result[:-4]

diff = result[-2]-result[-1]

for i in range(34):
    result=np.append(result,result[-1]-diff)"""
    
    
results = pd.read_csv("results_of_university.csv")
result1 = np.array(results.iloc[:,1])
result = result1.reshape(result1.shape[0],1)


di = np.zeros((150,1))
for i in range(150):
    di[i] = i+1
    
result1 = np.array(result)
for i in range(100):
    result1[i] = result[i] + addable_avg[i]
result1 = result1.reshape(result.shape[0],1)

regr = linear_model.LinearRegression()
poly = PolynomialFeatures(5,interaction_only=False)
x = poly.fit_transform(result1)
regr.fit(x,di)


file_name1 = "rank_model_univ.pkl"
file_name2 = "poly_univ.pkl"
pickle.dump(regr,open(file_name1,"wb"))
pickle.dump(poly,open(file_name2,"wb"))