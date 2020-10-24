#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


# In[3]:


df = pd.read_csv('creditcard.csv')
df


# In[ ]:





# In[4]:


df['Class'].unique()


# In[5]:


df["Class"] = np.where(df['Class'] == 1, 'Fraud', 'Non-fraud') 


# In[6]:


df['Class'].value_counts()


# In[7]:


sns.countplot(x='Class', data = df, linewidth  = 2 ,edgecolor = sns.color_palette('dark'))
plt.title('Class Distribution ',fontsize = 16)
plt.show()
# O đồ thị dưới, ta nhận thấy sự mất cân bằng giữa số người gian lận và không gian lận trong giao dịch ở cột Class của data 


# In[8]:


df.Class.value_counts(normalize = True).plot(kind = 'bar')
plt.title('Class Distribution ',fontsize = 16)
plt.show()
# Đồ thị này tương tự như trên nhưng được biểu diễn dưới dạng tỷ lệ 


# In[9]:


labels = df.Class.unique()
colors = ['green', 'blue']
explode = [0,0]
sizes = df.Class.value_counts().values

plt.figure(figsize =(7,7))
plt.pie(sizes, explode = explode , labels= labels, colors=colors,autopct= '%1.1f%%')
plt.title('Fraud and Non-Fraud in Data', fontsize = 15)
plt.legend()
plt.show()
# Tương tự như đồ thị trên nhưng cho thấy sự mất cân bằng lớn giữa những người gian lận và không gian lận


# In[10]:


print(df.Class.value_counts())
print('*'*20)
print(df.Class.value_counts(normalize= True))


# In[11]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = (df['Time']/3600).values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()

# 2 đồ thị dưới cho ta thấy được số tiền giao dịch và thời điểm giao dịch của data
# Đối với hình 1 , ta thấy rằng đồ thị này lệch phải (tức là có độ lệch dương), cho thấy lượng tiền giao dịch càng lớn thì sẽ không được thực hiện nhiều
# Đối với hình 2 , thì ta thấy đây là 1 đồ thị đa đỉnh , cho thấy thời điểm xảy ra giao dịch. Trong khoảng Time [0,25000] và [90000,110000] sẽ ít giao dịch hơn thông thường 


# #  Balanced the data 

# ## Data sub-sampling and cleaning

# In[12]:


fraud = df[df['Class'] == 'Fraud']
non_fraud = df[df['Class'] == 'Non-fraud'].sample(len(fraud) * 5)
non_fraud.reset_index(drop=True, inplace=True)
fraud.reset_index(drop=True, inplace=True)
new_data = pd.concat([non_fraud, fraud]).sample(frac=1).reset_index(drop=True)
new_data.describe()

# Vì theo trên ta nhận thấy rằng data không balanced nên ta cần làm cho data trở nên balanced 


# In[13]:


print('Distribution of the Classes in the subsample dataset')
print(new_data['Class'].value_counts()/len(new_data))
sns.countplot('Class', data=new_data, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

# Sau khi sub-sampling and cleaning thì ta được đồ thị data['Class'] như sau


# In[48]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

Class_Amount =new_data.groupby('Class')['Amount'].sum()
Class_Amount.plot(kind = 'bar',ax=ax[0])
ax[0].set_title('Total amount spending to transaction of 2 type of class', fontsize=14)


sns.boxplot(x= 'Class', y = 'Amount',data =new_data,ax=ax[1])
ax[1].set_title('Boxplot about amount spending of 2 type of class ', fontsize=14)
plt.show()

# Đồ thị này cho thấy lượng tiền mà non-fraud và fraud giao dịch
# Ta nhận thấy rằng lượng giao dịch của non-fraud nhiều hơn đơn giản là vì fraud thì việc giao dịch là phạm pháp nên gặp rất nhiều hạn chế 
# Cả fraud và Non-fraud đều có Amount giao dịch rất hẹp nhưng có nhiều outliers


# In[47]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))


Class_Time =new_data.groupby('Class')['Time'].sum()
Class_Time.plot(kind = 'bar',ax =ax[0])
ax[0].set_title('Total time spending to transaction of 2 type of class', fontsize=14)

sns.boxplot(x= 'Class', y = 'Time',data =new_data,ax=ax[1])
ax[1].set_title('Boxplot about time spending of 2 type of class ', fontsize=14)
plt.show()


# Tương tự như trên , thời gian giao dịch của fraud cũng ít hơn non-fraud


# In[23]:




plt.figure(figsize = (14,12))
sns.heatmap(new_data.corr(), linewidths= .1, cmap= 'YlGnBu', annot = True)
plt.yticks(rotation = 0)
plt.show()
# O hình này ta nhìn thấy sự tương quan giữa các cột trong data 
# Các cột V2,V4,V11.. có correlation không cao nên khi lấy dữ liệu từ các cột này để đối chiếu với cột khác thì phải lấy cả 2 côt
# Còn lại là những cột có hệ số tương quan cao như V8,V13,V15... nên khi lấy dữ liệu này để so với cột khác thì ta có thể chọn 1 trong 2 cột


# In[26]:


Facegrid = sns.FacetGrid(new_data, hue = 'Class', size = 6)
Facegrid.map(sns.kdeplot, 'Time',shade = True)
# Facegrid.set(xlim = (0,(new_data['Time']).max()))
Facegrid.add_legend()
plt.show()
# Đồ thị này làm rõ hình trên khi trong những thời điểm nhất định thì non-fraud giao dịch nhiều hơn fraud và ngược lại


# In[56]:


print(non_fraud['Time'].describe())
print('*'*20)
print(fraud['Time'].describe())


# In[19]:


plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.title('Histogram of Time for non-fraudulent samples')
sns.distplot(non_fraud["Time"])
plt.subplot(1, 2, 2)
plt.title('Histogram of Time for fraudulent samples')
sns.distplot(fraud["Time"])
# Đồ thị dưới đây sẽ làm rõ hơn đồ thị trên 
# Đối với non-fraud : _ Thời gian giao dịch nhiều nhất của họ xảy ra ở 2 điểm 500000 và 150000 tức 1h trưa và 5h chiều
#                     _ Còn thời gian họ ít hoạt động là tại 100000 tức 3-4h sáng tức là thời gian đó họ đang ngủ
# Đối với fraud     : _ Thời gian giao dịch nhiều nhất của họ xảy ra ở điểm 400000 trở đi tức 11h trưa trở đi cho tới 2-3h sáng
#                     _ Còn thời gian họ ít hoạt động là tại 110000 tức 5 sáng trở đi, do đó là lúc mọi người đã dậy và chuẩn bị đi làm


# In[ ]:




