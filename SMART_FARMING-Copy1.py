#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# loading the dataset
crop_data=pd.read_csv("Crop_recommendation.csv")
crop_data


# In[3]:


#rows and columns
crop_data.shape


# ### Checking basic information against columns

# In[62]:


crop_data.info()


# In[6]:


# dataset columns
crop_data.columns


# ### Changing the name of label to Crop for readability

# In[7]:


crop_data.rename(columns = {'label':'Crop'}, inplace = True)
crop_data


# ### Statistical inference of the dataset

# In[59]:


crop_data.describe()


# ### Statistical inference of the dataset

# In[9]:


crop_data.describe()


# ### Dropping missing values 

# In[58]:


crop_data = crop_data.dropna()
crop_data


# ### Visualizing the features

# In[11]:


ax = sns.pairplot(crop_data)
ax


# In[12]:


crop_data.Crop.unique()


# ### Get top 5 most frequent growing crops

# In[63]:


n = 5
crop_data['Crop'].value_counts()[:5].index.tolist()


# In[14]:


sns.barplot(crop_data["Crop"], crop_data["temperature"])
plt.xticks(rotation = 90)


# In[15]:


sns.barplot(crop_data["Crop"], crop_data["ph"])
plt.xticks(rotation = 90)


# In[16]:


sns.barplot(crop_data["Crop"], crop_data["humidity"])
plt.xticks(rotation = 90)


# In[17]:


sns.barplot(crop_data["Crop"], crop_data["rainfall"])
plt.xticks(rotation = 90)


# In[18]:


crop_data.corr()


# In[19]:


sns.heatmap(crop_data.corr(), annot =True)
plt.title('Correlation Matrix')


# ### Shuffling the dataset to remove order

# In[56]:


from sklearn.utils import shuffle

df  = shuffle(crop_data,random_state=5)
df.head()


# In[22]:


# Selection of Feature and Target variables.
x = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['Crop']


# In[23]:


# Encoding target variable 
y = pd.get_dummies(target)
y


# In[24]:


# Splitting data set - 25% test dataset and 75% 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state= 0)

print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)


# In[26]:


# Importing necessary libraries for multi-output classification

from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# ## Naive Bayes Classification

# In[27]:


gnb = GaussianNB()
model = MultiOutputClassifier(gnb, n_jobs=-1)
model.fit(x_train, y_train)


# In[28]:


gnb_pred = model.predict(x_test)
gnb_pred


# In[29]:


# Calculating Accuracy
from sklearn.metrics import accuracy_score
a1 = accuracy_score(y_test.values.argmax(axis=1), gnb_pred.argmax(axis=1))
a1


# In[30]:


# creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values.argmax(axis=1), gnb_pred.argmax(axis=1))
#cm = confusion_matrix(y_test, gnb_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');


# In[32]:


from sklearn import metrics
# Print the confusion matrix
print(metrics.confusion_matrix(y_test.values.argmax(axis=1), gnb_pred.argmax(axis=1)))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_test.values.argmax(axis=1), gnb_pred.argmax(axis=1), digits=3))


# # Decision Tree Classification

# In[33]:


# Training
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=6)
multi_target_decision = MultiOutputClassifier(clf, n_jobs=-1)
multi_target_decision.fit(x_train, y_train)


# In[34]:


# Predicting test results 
decision_pred = multi_target_decision.predict(x_test)
decision_pred


# In[35]:


# Calculating Accuracy
from sklearn.metrics import accuracy_score
a2 = accuracy_score(y_test.values.argmax(axis=1), decision_pred.argmax(axis=1))
a2


# In[36]:


# creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values.argmax(axis=1), decision_pred.argmax(axis=1))
#cm = confusion_matrix(y_test, gnb_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');


# In[37]:


from sklearn import metrics
# Print the confusion matrix
print(metrics.confusion_matrix(y_test.values.argmax(axis=1), decision_pred.argmax(axis=1)))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_test.values.argmax(axis=1), decision_pred.argmax(axis=1), digits=3))


# # Random Forest Classification

# In[38]:


# Training
forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(x_train, y_train)


# In[39]:


# Predicting test results 
forest_pred = multi_target_forest.predict(x_test)
forest_pred


# In[40]:


# Calculating Accuracy
from sklearn.metrics import accuracy_score
a3 = accuracy_score(y_test.values.argmax(axis=1), forest_pred.argmax(axis=1))
a3


# In[41]:


# creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values.argmax(axis=1), forest_pred.argmax(axis=1))
#cm = confusion_matrix(y_test, gnb_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');


# In[42]:


from sklearn import metrics
# Print the confusion matrix
print(metrics.confusion_matrix(y_test.values.argmax(axis=1), forest_pred.argmax(axis=1)))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_test.values.argmax(axis=1), forest_pred.argmax(axis=1), digits=3))


# ## KNN Classifier

# In[45]:


from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier()
model = MultiOutputClassifier(knn_clf, n_jobs=-1)
model.fit(x_train, y_train)


# In[46]:


knn_pred = model.predict(x_test)
knn_pred


# In[47]:


# Calculating Accuracy
from sklearn.metrics import accuracy_score
a4 = accuracy_score(y_test.values.argmax(axis=1), knn_pred.argmax(axis=1))
a4


# In[48]:


# creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values.argmax(axis=1), knn_pred.argmax(axis=1))
#cm = confusion_matrix(y_test, gnb_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');


# In[49]:


from sklearn import metrics
# Print the confusion matrix
print(metrics.confusion_matrix(y_test.values.argmax(axis=1), knn_pred.argmax(axis=1)))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_test.values.argmax(axis=1), knn_pred.argmax(axis=1), digits=3))


# # Gradient Boosting

# In[50]:


from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier()
model = MultiOutputClassifier(gb_clf, n_jobs=-1)
model.fit(x_train, y_train)


# In[51]:


gf_pred = model.predict(x_test)
gf_pred


# In[52]:


# Calculating Accuracy
from sklearn.metrics import accuracy_score
a5 = accuracy_score(y_test.values.argmax(axis=1), gf_pred.argmax(axis=1))
a5


# In[53]:


# creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values.argmax(axis=1), gf_pred.argmax(axis=1))
#cm = confusion_matrix(y_test, gnb_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');


# In[54]:


from sklearn import metrics
# Print the confusion matrix
print(metrics.confusion_matrix(y_test.values.argmax(axis=1), gf_pred.argmax(axis=1)))

# Print the precision and recall, among other metrics
print(metrics.classification_report(y_test.values.argmax(axis=1), gf_pred.argmax(axis=1), digits=3))


# # Complete
