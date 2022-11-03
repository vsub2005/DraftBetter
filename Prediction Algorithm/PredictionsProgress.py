#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
sevenYearStats = pd.read_csv("/Users/vinaysubramanian/Downloads/PolygenceProject/Prediction Algorithm/2015-2021FantasyStats.csv")
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
sevenYearStats = sevenYearStats[:].fillna(0)
from notebook.services.config import ConfigManager
c = ConfigManager()
c.update('notebook', {"CodeCell": {"cm_config": {"autoCloseBrackets": True}}})

def nextYear(x):
    truth = sevenYearStats.loc[:, 'PlayerID'] == x['PlayerID']
    print(x.loc['Rank'])
    year = x['Year']
    newDF = sevenYearStats.loc[truth]
    truth2 = sevenYearStats.loc[truth, 'Year'] == (year + 1)
    
    if sum(truth2) >0:
        value = newDF.loc[truth2, 'FantasyPoints']
        print(value)
        return value.item()
    else:
        return 0
    
x = sevenYearStats.apply(lambda x: nextYear(x), axis=1)
sevenYearStats.loc[:, 'NextYearPoints'] = x


# In[7]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

columns = list(sevenYearStats.columns[6:])
columns.remove("NextYearPoints")
columns.remove("Year")
columns

X_train, X_test, y_train, y_test = train_test_split(sevenYearStats.loc[:, columns], sevenYearStats.loc[:, "NextYearPoints"], test_size=0.3, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train.loc[:, columns])
normalizedData = scaler.transform(X_train.loc[:, columns])

scaler2 = StandardScaler()
scaler2.fit(X_test.loc[:, columns])
normalizedData2 = scaler2.transform(X_test.loc[:, columns])

maximum = 0
maxValue = 0
for i in range(1, 200):
    neigh = KNeighborsRegressor(n_neighbors=i)
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    print(i, score, maxValue)
    if(score > maxValue):
        maxValue = score
        maximum = i
print(maximum)


# In[8]:


predictions = neigh.predict(X_test)
predictions


# In[10]:


y_test


# In[11]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

columns = list(sevenYearStats.columns[6:])
columns.remove("NextYearPoints")
columns.remove("Year")

X_train, X_test, y_train, y_test = train_test_split(sevenYearStats.loc[:, columns], sevenYearStats.loc[:, "NextYearPoints"], test_size=0.3, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train.loc[:, columns])
normalizedData = scaler.transform(X_train.loc[:, columns])

scaler2 = StandardScaler()
scaler2.fit(X_test.loc[:, columns])
normalizedData2 = scaler2.transform(X_test.loc[:, columns])

maximum = 0
maxValue = 0


model = DecisionTreeRegressor()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)


# In[ ]:




