
# read university data
import pandas as pd

# Data Gathering
university=pd.read_csv('Universities.csv')
university.head()

# Data Processing
# ignore the string column 
x=university.iloc[:,1:]
x.head()

# Normalization 0-1
from sklearn.preprocessing import MinMaxScaler
norm=MinMaxScaler()
for i in x.columns:
    x[i]=norm.fit_transform(x[i].values.reshape(-1, 1))

# Model Creation K means clustering
from sklearn.cluster import KMeans
#Plot Elbow curve to get k value 
#total within sum of squared
twss=[]
for i in range(2,10):
    model=KMeans(n_clusters=i).fit(x)
    twss.append(model.inertia_)

# Model Visualization
import matplotlib.pyplot as plt
plt.plot(range(2,10),twss,'-bo')
plt.xlabel('Value of K')
plt.ylabel('TWSS')
plt.show()

#If you know the k value
from sklearn.cluster import KMeans
model=KMeans(n_clusters=3).fit(x)
model.labels_

university['label']=model.labels_
university.iloc[:,1:].groupby('label').mean()
