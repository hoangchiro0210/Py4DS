#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


# In[3]:


df = pd.read_csv('xAPI-Edu-Data.csv')
df.head(10)


# In[4]:


# it is....


# In[5]:


for i in range(1,17):
    print(df.iloc[:,i].value_counts())


# In[6]:


sns.pairplot(df, hue = 'Class')
# Ở đồ thị này ta nhận thấy rằng :
#  Class L luôn có đồ thị lệch phải, có nghĩa là chỉ số về số lần giơ tay, đọc tài liệu ,đọc thông báo hay thảo luận đều thấp
#  Class M luôn có đồ thị đa đỉnh, có nghĩa là chỉ số về số lần giơ tay, đọc tài liệu ,đọc thông báo hay thảo luận đạt mức trung bình 
#  Class H luôn có đồ thị đa đỉnh hoặc lệch trái, có nghĩa là chỉ số về số lần giơ tay, đọc tài liệu ,đọc thông báo hay thảo luận đạt mức từ trung bình trở lên


# In[7]:


plt.figure(figsize = (14,12))
sns.heatmap(df.corr(), linewidths= .1, cmap= 'YlGnBu', annot = True)
plt.yticks(rotation = 0)
# Hình này cho thấy sự tương quan giữa 3 cột  raisedHands,VisitedResouces và AnouncementView
# Còn lại cột Dicussion có hệ số tương quan khá thấp đối với các thuộc tính khác 


# In[8]:


P_Satis = sns.countplot(x='Class', data = df, linewidth  = 2 ,edgecolor = sns.color_palette('dark'))
# Hình này cho thấy biểu đồ cột giữa 3 loại học sinh trong cột Class. Ta thấy rằng data gần như cân bằng 


# In[9]:


df.Class.value_counts(normalize = True).plot(kind = 'bar')
# Như hình trên nhưng chỉ số được biểu diễn dưới dạng tỉ trọng


# In[10]:


df.Class.value_counts()


# In[11]:


df.Class.value_counts(normalize = True)


# In[12]:


plt.subplots(figsize = (20,8))
df['raisedhands'].value_counts().sort_index().plot.bar()
plt.title('No. of times vs No. of students raised their hands on particular time')
plt.xlabel('No. of times,students raised their hands', fontsize =14)
plt.ylabel('No. of students, on particular times',fontsize = 14)
plt.show()
### not finish


# In[13]:


df.raisedhands.plot(kind = 'hist', bins = 100,figsize = (20,10), grid = 'True')
plt.xlabel('raisedhands')
plt.legend(loc = 'upper right')
plt.title('Raised hands Histogram')
plt.show()


# In[14]:


Raised_hand = sns.boxplot(x= 'Class', y = 'raisedhands',data =df)
plt.show()

# Hình này cho thấy số lần giơ tay của 3 loại học sinh :
# Trong khi class L nằm ở gần cuối (số lần giơ tay thấp) thì class M và H đều cho thấy số lần giơ tay khá cao


# In[15]:


Facegrid = sns.FacetGrid(df, hue = 'Class', size = 6)
Facegrid.map(sns.kdeplot, 'raisedhands',shade = True)
Facegrid.set(xlim = (0,df['raisedhands'].max()))
Facegrid.add_legend()
plt.show()

# Đồ thị này , class L bị lệch phải có nghĩa là chỉ số giơ tay rất thấp 
#              class H có đồ thị lệch trái có nghĩa là chỉ số giơ tay cao
#              class M có đồ thị đa đỉnh nằm giữa 2 class L và H có chỉ số giơ tay ở mức trung bình 


# #### Exploring 'ParentschoolSatisfaction' feature use sns.countplot, df.groupby and pd.crosstab functions 

# In[16]:


df.groupby(['ParentschoolSatisfaction'])['Class'].value_counts()


# In[17]:


pd.crosstab(df['Class'],df['ParentschoolSatisfaction'])


# In[18]:


sns.countplot(x = 'ParentschoolSatisfaction',data = df, hue ='Class', palette= 'bright')
plt.show()

# Đồ thị này cho thấy rằng :
# Các gia đình hài lòng về trường chiếm phần lớn từ class M và H 
# Các gia đình không hài lòng về trường thì chiếm nhiều ở class L


# ## Pie Chart

# In[21]:


labels = df.ParentschoolSatisfaction.unique()
colors = ['blue', 'green']
explode = [0,0]
sizes = df.ParentschoolSatisfaction.value_counts().values

plt.figure(figsize =(7,7))
plt.pie(sizes, explode = explode , labels= labels, colors=colors,autopct= '%1.1f%%')
plt.title('Parent school Satisfaction in Data', fontsize = 15)
plt.legend()
plt.show()

# Hình này cho thấy tỉ lệ hài lòng của các phụ huynh 
# Có đến gần 60% hài lòng về nhà trường (chiếm phần lớn) và chỉ gần 40% không hài lòng 


# In[ ]:




