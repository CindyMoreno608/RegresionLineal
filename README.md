# RegresionLineal
Regresión LinealUNAD

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot  as  plt 
import matplotlib.pylab  as  ptr 
from pylab import *

# In[6]:
AnalisisDat = pd.read_csv("../Documents/Analisis de datos Fase 2/Regresión Lineal/data.csv")


# In[8]:
AnalisisDat.head(5)


# In[9]:
print(AnalisisDat.metro)


# In[10]:
print('Cantidad de datos')
print(AnalisisDat.data.shape)


# In[13]:
AnalisisDat.columns

# In[16]:
AnalisisDat[['metro','precio']].head()


# In[34]:
AnalisisDat.plot.scatter(x='metro',y='precio')
plt.show()


# In[48]:
x=[5,15,20,25]
y=[375,487,450,500]
ptr.plot(x,y)
ptr.show()


# In[49]:
AnalisisDat.plot.scatter(x='metro',y='precio')
ptr.plot(x,y)
ptr.show()


# In[53]:
x=arange(10.)


# In[69]:
ptr.plot(x)
plot(x)


# In[ ]:




