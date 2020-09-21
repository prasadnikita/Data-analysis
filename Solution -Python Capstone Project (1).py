#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[122]:


#reading file
customer_churn = pd.read_csv("customer_churn.csv") 


# In[123]:


#finding the first few rows
customer_churn.head()


# In[124]:


#Extracting 5th column
customer_5=customer_churn.iloc[:,4] 
customer_5.head()


# In[125]:


#Extracting 15th column
customer_15=customer_churn.iloc[:,14] 
customer_15.head()


# In[127]:


#'Extracting male senior citizen with payment method-> electronic check'
senior_male_electronic=customer_churn[(customer_churn['gender']=='Male') & (customer_churn['SeniorCitizen']==1) & (customer_churn['PaymentMethod']=='Electronic check')]
senior_male_electronic.head(10)


# In[136]:


#tenure>70 or monthly charges>100
customer_total_tenure=customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]
customer_total_tenure.head(10)


# In[129]:


customer_total_tenure1 = customer_churn[(customer_churn["tenure"] > 70) | (customer_churn["MonthlyCharges"] > 100)]
customer_total_tenure1.head()


# In[130]:


#contract is 'two year', payment method is 'Mailed Check', Churn is 'Yes'
two_mail_yes=customer_churn[(customer_churn['Contract']=='Two year') & (customer_churn['PaymentMethod']=='Mailed check') & (customer_churn['Churn']=='Yes')]
two_mail_yes


# In[138]:


#Extracting 333 random records
customer_333=customer_churn.sample(n=333)
customer_333.head()


# In[132]:


len(customer_333)


# In[143]:


#count of levels of churn column
customer_churn['InternetService'].value_counts()


# In[77]:


#-------------------------------Data Visualization------------------#


# In[78]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[142]:


#bar-plot for 'InternetService' column
x=customer_churn['InternetService'].value_counts().keys().tolist()
y=customer_churn['InternetService'].value_counts().tolist()
plt.bar(x,y,color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of categories')
plt.title('Distribution of Internet Service')


# In[144]:


#histogram for 'tenure' column
plt.hist(customer_churn['tenure'],color='green',bins=30)
plt.title('Distribution of tenure')


# In[149]:


#scatterplot 
plt.scatter(x=customer_churn['tenure'].head(20),y=customer_churn['MonthlyCharges'].head(20),color='brown')
plt.xlabel('Tenure of Customer')
plt.ylabel('Monthly Charges of Customer')
plt.title('Tenure vs Monthly Charges')
plt.grid(True)


# In[150]:


#Box-plot by using pandas
customer_churn.boxplot(column='tenure',by=['Contract'])


# In[83]:


#Box-plot using seaborn
import seaborn as sns
sns.boxplot(x='Contract', y='tenure', data=customer_churn, width=0.3)


# In[84]:


#-----------------------Linear Regresssion----------------------


# In[85]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[86]:


x=pd.DataFrame(customer_churn['tenure'])
y=customer_churn['MonthlyCharges']


# In[87]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[88]:


#building the model
from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(x_train,y_train)


# In[89]:


#predicting the values
y_pred = simpleLinearRegression.predict(x_test)


# In[90]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
rmse


# In[91]:


#----------------------------------Logistic Regression-------------------------------


# In[92]:


x=pd.DataFrame(customer_churn['MonthlyCharges'])
y=customer_churn['Churn']


# In[93]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.65,random_state=0)


# In[94]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[95]:


y_pred = logmodel.predict(x_test)


# In[96]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test),accuracy_score(y_pred,y_test)


# In[97]:


#--------------Multiple logistic regression-------------------


# In[98]:


x=pd.DataFrame(customer_churn.loc[:,['MonthlyCharges','tenure']])
y=customer_churn['Churn']


# In[99]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=0)


# In[100]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[117]:


y_pred = logmodel.predict(x_test)


# In[118]:


from sklearn.metrics import confusion_matrix,accuracy_score    
print(confusion_matrix(y_pred,y_test),accuracy_score(y_pred,y_test))  


# In[103]:


#---------------decision tree---------------


# In[104]:


x=pd.DataFrame(customer_churn['tenure'])
y=customer_churn['Churn']


# In[105]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  


# In[106]:


from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(x_train, y_train)  


# In[107]:


y_pred = classifier.predict(x_test)  


# In[108]:


from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))   
print(accuracy_score(y_test, y_pred))  


# In[109]:


#--------------random forest---------------------


# In[110]:


x=customer_churn[['tenure','MonthlyCharges']]
y=customer_churn['Churn']


# In[111]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  


# In[112]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)


# In[113]:


y_pred=clf.predict(x_test)


# In[120]:


from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))   
print(accuracy_score(y_test, y_pred))  


# In[ ]:




