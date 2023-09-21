#!/usr/bin/env python
# coding: utf-8

# Ejercicio de Regresión Logistica (Cindy Moreno)
# 

# In[100]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os 
import matplotlib.pyplot  as  plt 
import matplotlib.pylab  as  ptr 
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[2]:


datos = pd.read_csv("../Documents/Analisis de datos Fase 2/Regresión Logistica/framingham.csv")


# In[7]:


datos.head(16)


# In[13]:


datos.describe()


# In[49]:


X = datos[['diabetes','male','age']]
y = datos[['glucose']]


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


model = LogisticRegression()


# In[57]:


datos[['diabetes','glucose']].head(16)


# In[135]:


datos[['glucose','diabetes']].plot.scatter(x='glucose',y='diabetes')


# In[136]:


# pruebas de parametro
w = 0.09
b = -4.0


# In[140]:


# puntos de la recta
x = np.linspace(0,datos['BMI'].max(),100)
y = 1/(1+np.exp(-(w*x+b)))

datos.plot.scatter(x='BMI', y='diabetes')
plt.plot(x,y, '-r')
plt.ylim(0,datos['diabetes'].max()*1.1)
plt.show()


# In[138]:


# grafica de la recta
datos.plot.scatter(x='diabetes',y='glucose')
plt.plot(x, y, color='black')
plt.ylim(0,datos['glucose'].max()*1.1)
plt.scatter(x, y, color='#A9E2F3')
# plt.grid()
plt.xlabel('Diabetes')
plt.ylabel('Glucose 1:Positivo 0:Negativo')
plt.show()


# In[72]:


datos.value_counts().sort_index()


# In[103]:


datos.hist()


# In[109]:


cantidad_diabetes = datos.groupby(['diabetes']).count()['age']


# In[105]:


cantidad_diabetes


# In[ ]:




