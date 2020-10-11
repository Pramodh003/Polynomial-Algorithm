#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('ad50mm.csv')


# In[3]:


data.head()


# In[4]:


x = data.adv.values


# In[5]:


y = data.tumor.values


# In[6]:


x= x.reshape(-1,1)


# In[7]:


x


# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
## setting configs plot size 5x4 inches and seaborn style whitegrid
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams ['figure.figsize'] = 5,4
sb.set_style('whitegrid')


# In[9]:


fig = plt.figure()


# In[10]:


rcParams['figure.figsize'] = 8,10
ax = fig.add_axes([0,0,1,1])
fig = plt.figure()
fig, ax = plt.subplots(2,2)#2 rows and 3 columns

ax[0,0].plot(x,y)
ax[0,1].scatter(x,y)
ax[1,0].pie(y)
ax[1,1].plot(x)


plt.show()


# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
poly_reg = PolynomialFeatures(degree=6)
x_poly = poly_reg.fit_transform(x)


# In[12]:


x_poly


# In[13]:


model = LinearRegression()
model.fit(x_poly,y)
y_poly_pred = model.predict(x_poly)


# In[14]:


y_poly_pred


# In[15]:


y


# In[16]:


import numpy as np


# In[17]:


rmse = np.sqrt(mean_squared_error(y,y_poly_pred))


# In[18]:


r2 = r2_score(y,y_poly_pred)


# In[19]:


print(rmse)
print(r2)


# In[20]:


import operator


# In[21]:


plt.scatter(x,y,s=10)
sort_axis = operator.itemgetter(0)


# In[22]:


sorted_zip = sorted(zip(x,y_poly_pred),key=sort_axis)


# In[23]:


x,y_poly_pred = zip(*sorted_zip)


# In[24]:


plt.xlabel('E field intencity',fontsize=18,color='black')
plt.ylabel('Tumour Size',fontsize=18)
plt.plot(x,y_poly_pred,color='m')
df = pd.DataFrame(x,y_poly_pred) 
print('DataFrame:\n', df)
gfg_csv_data = df.to_csv('GfG60.csv', header = False) 
print('\n CSV String:\n', gfg_csv_data) 


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)


# In[26]:


from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_regression_model(degree):
  "Creates a polynomial regression model for the given degree"
  
  poly_features = PolynomialFeatures(degree=degree)
  
  # transforms the existing features to higher degree features.
  x_train_poly = poly_features.fit_transform(x_train)
  
  # fit the transformed features to Linear Regression
  poly_model = LinearRegression()
  poly_model.fit(x_train_poly, y_train)
  
  # predicting on training data-set
  y_train_predicted = poly_model.predict(x_train_poly)
  
  # predicting on test data-set
  y_test_predict = poly_model.predict(poly_features.fit_transform(x_test))
  
  # evaluating the model on training dataset
  rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
  r2_train = r2_score(y_train, y_train_predicted)
  
  # evaluating the model on test dataset
  rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
  r2_test = r2_score(y_test, y_test_predict)
  
  print("The model performance for the training set")
  print("-------------------------------------------")
  print("RMSE of training set is {}".format(rmse_train))
  print("R2 score of training set is {}".format(r2_train))
  
  print("\n")
  
  print("The model performance for the test set")
  print("-------------------------------------------")
  print("RMSE of test set is {}".format(rmse_test))
  print("R2 score of test set is {}".format(r2_test))


# In[27]:


create_polynomial_regression_model(6)


# In[28]:


from sklearn.preprocessing import PolynomialFeatures


# In[ ]:




