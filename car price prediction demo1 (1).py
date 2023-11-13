#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

import warnings
warnings.filterwarnings('ignore')



# In[89]:


data = pd.read_csv('C:/Users/gfddb/Downloads/car_price_prediction.csv')


# In[90]:


data.head()


# In[91]:


data.shape


# In[92]:


data.info()


# In[93]:


data.duplicated().sum()


# In[94]:


# drop duplicate values
data.drop_duplicates(inplace= True)


# In[95]:


data.describe()


# In[96]:


for col in data.columns:
    print(f'Category in {col} is :\n {data[col].unique()}\n')
    print('\\'*50)


# In[97]:


# Replacing '-' with 0
data['Levy']=data['Levy'].replace('-','0')

# Converting Levy type to float
data['Levy'] = data['Levy'].astype('float64')
dtime = dt.datetime.now()
data['Age']=dtime.year - data['Prod. year']
data = data.drop('Prod. year',axis=1)
data.head()


# In[98]:


data=data.drop(['ID','Doors'],axis=1)
# Replacing 'Km' with ''  
data['Mileage'] =data['Mileage'].str.replace('km',"")

# Converting Mileage type to int64
data.Mileage = data.Mileage.astype('Int64')

# Replacing 'Turbo' with '' 
data['Engine volume'] = data['Engine volume'].str.replace('Turbo','')

# Converting Levy type to float
data['Engine volume'] = data['Engine volume'].astype('float64')
data['Engine volume'].unique()
data.head()


# In[99]:


data.hist(bins=25,figsize=(15,10),color='peru')
plt.show()


# In[100]:


top_10_cars = data.Manufacturer.value_counts().sort_values(ascending=False)[:10]
top_10_cars


# In[101]:


plt.figure(figsize=(15, 10))
sns.barplot(x=top_10_cars, y=top_10_cars.index,palette='hot',linewidth = 4)
plt.title('Top10 The Most Frequent Cars',loc='center',fontweight='bold',fontsize=18)
plt.xlabel('Frequency',fontsize=20)
plt.ylabel('Cars',fontsize=20)
plt.tight_layout()
plt.show()


# In[102]:


top_10_cars_means_prices = [data[data['Manufacturer']==i]['Price'].mean() for i in list(top_10_cars.index)]
plt.figure(figsize=(15,10))
plt.plot(top_10_cars.index,top_10_cars_means_prices,color='r',
         linewidth = 4,marker='o',markersize = 20)
plt.title('Top 10 Cars by Average Price',loc='center',fontweight='bold',fontsize=18)
plt.ylabel('Average Price',fontsize=20)
plt.xlabel('Cars',fontsize=20)
plt.tight_layout()
plt.show()


# In[103]:


cor= data.select_dtypes(exclude=object).corr()
sns.heatmap(cor, annot= True, linewidths= 0.5,cmap='hot')
plt.title('Correlation Heatmap')
plt.show()


# In[104]:


# Lets define the numeric columns and deal with outliers
numeric_data = data.select_dtypes(exclude=object)
for col in numeric_data:
    q1 = data[col].quantile(0.75)
    q2 = data[col].quantile(0.25)
    iq = q1 - q2
    
    low = q2-1.5*iq
    high = q1-1.5*iq
    outlier = ((numeric_data[col]>high) | (numeric_data[col]<low)).sum()

    total = numeric_data[col].shape[0]
    print(f"Total Outliers in {col} are :{outlier}---{round(100*(outlier)/total,2)}%")
if outlier>0:
    data = data.loc[(data[col]<=high) & (data[col]>=low) ]


# In[105]:


obdata = data.select_dtypes(include=object)
numdata = data.select_dtypes(exclude=object)
for i in range(0,obdata.shape[1]):
    obdata.iloc[:,i] = lab.fit_transform(obdata.iloc[:,i])
data = pd.concat([obdata,numdata],axis=1)
data


# In[107]:


x= data.drop('Price',axis=1)
y= data['Price']


# In[ ]:


# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=5)
algorithm = ['LinearRegression','DecisionTreeClassifier','RandomForestClassifier','GradientBoostingRegressor','SVR']
R2=[]
RMSE = []
def models(model):
    model.fit(x_train,y_train)
    pre = model.predict(x_test)
    r2 = r2_score(y_test,pre)
    rmse = np.sqrt(mean_squared_error(y_test,pre))
    R2.append(r2)
    RMSE.append(rmse)
    score = model.score(x_test,y_test)
    print(f'The Score of Model is :{score}')
model1 = LinearRegression()
model2 = DecisionTreeRegressor()
model3 = RandomForestRegressor()
model4 = GradientBoostingRegressor()
model5 = SVR()
models(model1)
models(model2)
models(model3)
models(model4)
models(model5)


# In[109]:


df = pd.DataFrame({'Algorithm':algorithm, 'R2_score': R2, 'RMSE':RMSE})
df


# In[110]:


fig = plt.figure(figsize=(20,8))
plt.plot(df.Algorithm,df.R2_score ,label='R2_score',lw=5,color='peru',marker='v',markersize = 15)
plt.legend(fontsize=15)
plt.show()


# In[111]:


fig = plt.figure(figsize=(20,8))
plt.plot(df.Algorithm,df.RMSE ,label='RMSE',lw=5,color='r',marker='o',markersize = 10)
plt.legend(fontsize=15)
plt.show()


# In[ ]:




