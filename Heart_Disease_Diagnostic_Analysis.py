#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Reading the "heart.csv" file

# In[2]:


df = pd.read_csv("heartdisease.csv")


# let's see few samples of data

# In[3]:


df.sample(3)


# seems like all the fields are in numeric.

# As said in description,
# Let's understand the data,
# > 1. age 
# > 2. sex 
# > 3. chest pain type (4 values) 
# > 4. resting blood pressure 
# > 5. serum cholestoral in mg/dl 
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved 
# > 9. exercise induced angina 
# > 10. oldpeak = ST depression induced by exercise relative to rest 
# > 11. the slope of the peak exercise ST segment 
# > 12. number of major vessels (0-3) colored by flourosopy 
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# ***analyzing Categorical Data*** :

# Let's plot these categorical values, based on target

# In[4]:


temp = (df.groupby(['target']))['cp'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "cp", data = temp).set_title("Chest Pain vs Heart Disease")


# From the above plot, we can understand that, chest pain of **type - 2** constitutes most of the chest Pain Category for Heart Disease.

# In[6]:


temp = (df.groupby(['target']))['fbs'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "fbs", data = temp).set_title("FBS vs Target")


# We can eliminate this feature while building model

# In[7]:


temp = (df.groupby(['target']))['restecg'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "restecg", data = temp).set_title("resting electrocardiographic results vs Heart Disease")
                     


# "restecg" of **type 1** are more prone to Heart Disease compared to that of other types.

# In[7]:


temp = (df.groupby(['target']))['ca'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "ca", data = temp).set_title("ca vs Heart Disease")
                     


# 
# * ca having of **type 0** are high in Heart Disease patients.

# In[8]:


temp = (df.groupby(['target']))['thal'].value_counts(normalize=True).mul(100).reset_index(name = "percentage")
sns.barplot(x = "target", y = "percentage", hue = "thal", data = temp).set_title("thal vs Heart Disease")
                     


# 
# * **Type - 3** is common in people who are not affected with Heart Disease 
# * **Type - 2** is common in Heart Disease victims.

# Finished of Categorical Data, let's check about other attributes.

# In[9]:


df.info()


# Let's plot boxplot and check whether there are any outliers in any columns

# In[9]:


df.boxplot()
plt.xticks(rotation = 90)


# seems like some of the columns have worst outliers

# In[11]:


import seaborn as sns
sns.heatmap(df.corr())


# In[12]:


plt.boxplot(df.trestbps)


# In[13]:


Q3 = df.trestbps.quantile(.75)
Q1 = df.trestbps.quantile(.25)
IQR = Q3 - Q1
df = df[~((df.trestbps < Q1 - 1.5*IQR) | (df.trestbps > Q3 + 1.5*IQR))]


# In[14]:


Q3 = df.chol.quantile(.75)
Q1 = df.chol.quantile(.25)
IQR = Q3 - Q1
df = df[~((df.chol < Q1 - 1.5*IQR) | (df.chol > Q3 + 1.5*IQR))]


# In[15]:


df.boxplot()
plt.xticks(rotation = 90)


# In[16]:


sns.countplot(x="target", data=df,hue = 'sex').set_title("GENDER - Heart Diseases")


# In[17]:


df.sample()


# **Let's start  Analyzing the data and find some insights.**

# 1. average age of male who got stroke

# In[18]:


df[(df.target ==  1) & (df.sex == 1)].age.mean()


# 2. average age of female who got stroke

# In[19]:


df[(df.target ==  1) & (df.sex == 0)].age.mean()


# 3. Average values of features that are responsible for disease for female

# In[20]:


df[(df.target ==  1) & (df.sex == 0)].describe()[1:2]


# 4. Average values of features that are responsible for disease for male

# In[21]:


df[(df.target ==  1) & (df.sex == 1)].describe()[1:2]


# 5. Average values of features that are responsible for not having disease for female

# In[22]:


df[(df.target ==  0) & (df.sex == 0)].describe()[1:2]


# 6. Average values of features that are responsible for not having disease for male

# In[23]:


df[(df.target ==  0) & (df.sex == 0)].describe()[1:2]


# we Should use "mode" on Categorical data

# In[24]:


x = df.age.tolist()
after_x = []
for i in x:
    if i < 20:
        after_x.append("teenager")
    elif i < 30:
        after_x.append("20 - 30")
    elif i < 40:
        after_x.append("30 - 40")
    elif i < 50:
        after_x.append("40 - 50")
    elif i < 60:
        after_x.append("50 - 60")
    else:
        after_x.append("senior citizen")
df["age_category"] = after_x


# In[25]:


df.sample()


# In[26]:


for_analyzing = df.groupby(["age_category","sex","target"]).agg({"age":"mean", "trestbps":"mean", "chol":"mean", "thalach":"mean",     "exang":"mean","oldpeak":"mean", "slope":"mean","fbs" : pd.Series.mode,     "cp" : pd.Series.mode, "restecg": pd.Series.mode,"ca":pd.Series.mode,"thal":pd.Series.mode})
for_analyzing


# In[30]:


df.sample()


# In[31]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(df.iloc[:,:-2],df.iloc[:,-2],)


# In[32]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
dpen = []
for i in range(5,11):
    model = XGBClassifier(max_depth = i)
    model.fit(train_x,train_y)
    target = model.predict(test_x)
    dpen.append(accuracy_score(test_y, target))
    print("accuracy : ",dpen[i-5])
print("Best accuracy: ",max(dpen))


# In[ ]:




