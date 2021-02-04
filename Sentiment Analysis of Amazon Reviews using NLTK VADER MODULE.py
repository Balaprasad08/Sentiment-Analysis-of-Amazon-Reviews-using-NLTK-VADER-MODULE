#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir('E:\\prasad\\practice\\My Working Projects\\Completed\\NLP\\Sentiment Analysis of Amazon Reviews using NLTK VADER MODULE')


# In[3]:


df=pd.read_csv('amazonreviews.csv',encoding='ISO-8859-1')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


sns.countplot(df['label'])
plt.show()


# ### Use NLTK VADER Module

# In[8]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[9]:


sid=SentimentIntensityAnalyzer()


# In[10]:


sid.polarity_scores(df.loc[0]['review'])


# ## Polarity Scores

# In[11]:


df['scores']=df['review'].apply(lambda X:sid.polarity_scores(X))
df.head()


# ## Compound Score

# In[12]:


df['compound']=df['scores'].apply(lambda X: X['compound'])
df.head()


# ## Pos & Neg

# In[13]:


df['comp_score']=df['compound'].apply(lambda X: 'pos' if X>=0 else 'neg')
df.head()


# ### Check The Model Accuracy

# In[14]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[15]:


accuracy_score(df['label'],df['comp_score'])


# In[16]:


confusion_matrix(df['label'],df['comp_score'])


# In[17]:


print(classification_report(df['label'],df['comp_score']))


# In[ ]:




