import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

#load dataset
df = pd.read_csv('xAPI-Edu-Data.csv')
df

### Exploratory data analysis(EDA)

for i in range(1,17):
    print(df.iloc[:,i].value_counts())
    print("*"*20)

df.rename(index=str, columns={'gender':'Gender', 
                              'NationalITy':'Nationality',
                              'raisedhands':'RaisedHands',
                              'VisITedResources':'VisitedResources'},
                               inplace=True)
df.columns

print("Class Unique Values : ", df["Class"].unique())
print("Topic Unique Values : ", df["Topic"].unique())
print("StudentAbsenceDays Unique Values : ", df["StudentAbsenceDays"].unique())
print("ParentschoolSatisfaction Unique Values : ", df["ParentschoolSatisfaction"].unique())
print("Relation Unique Values : ", df["Relation"].unique())
print("SectionID Unique Values : ", df["SectionID"].unique())
print("Gender Unique Values : ", df["Gender"].unique())

sns.pairplot(df,hue="ParentAnsweringSurvey")
plt.show()

####  In some metrics, we can see a stark difference between indexing in the other Class

### Exploring 'ParentAnsweringSurvey' label

P_Satis =sns.countplot(x="ParentAnsweringSurvey", data=df, linewidth=2, edgecolor=sns.color_palette('dark'))
plt.title("Countplot of ParentAnsweringSurvey", fontsize=18)
plt.show()

##### We can see balanced this plot  with answer Yes and No

df.ParentAnsweringSurvey.value_counts(normalize=True).plot(kind = "bar")

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot= True)
plt.yticks(rotation=0)
plt.show()

#### Correlation matrix, these factors each seem to have a pretty high correlation with each other(e.g., RaisedHands is well correlated with VisitedResources , and so on)

print(df.ParentAnsweringSurvey.value_counts())

print(df.ParentAnsweringSurvey.value_counts(normalize = True))

### exploring 'VisitedResources' feature

plt.subplots(figsize=(25,8))
df['VisitedResources'].value_counts().sort_index().plot.bar()
plt.title("No. of times", fontsize=18)
plt.xlabel("No. of times, student visited resource", fontsize=14)
plt.ylabel("NO. of student, on particular times", fontsize=14)
plt.show()

sns.boxplot(x="ParentAnsweringSurvey", y= "VisitedResources", data=df)
plt.title("Boxplot ParentAnsweringSurvey - VisitedResources", fontsize = 15)
plt.show()

#### In this boxplot,  Perent answer No spread the data evenly from 10 to 80 and mean is 30. Perent answer Yes is right skewed, mean is 80. 

Facetgrid = sns.FacetGrid(df, hue = "ParentAnsweringSurvey", size=6)
Facetgrid.map(sns.kdeplot, "VisitedResources", shade=True)
Facetgrid.set(xlim=(0,df['VisitedResources'].max()))
Facetgrid.add_legend()
plt.show()

#### Perent Answer Yes is left skewed,  Perent Answer No is more balanced than their. 

### Exploring 'ParentschoolSatisfaction' feature use sns.countplot, df.groupby and pd.crosstab function

df.groupby(['ParentschoolSatisfaction'])['ParentAnsweringSurvey'].value_counts()

pd.crosstab(df['ParentAnsweringSurvey'],df['ParentschoolSatisfaction'])

sns.countplot(x='ParentschoolSatisfaction',data=df, hue='ParentAnsweringSurvey',palette='bright')
plt.show()

#### There a couple of pretty clear patterns we can see here. For example, Good evaluation have highly imbalanced answer Yes  more than answer No, Bad evaluation have highly imbalanced answer No more than answer Yes.

labels = df.ParentschoolSatisfaction.unique()
colors = ["brown", "green"]
explode = [0,0]
sizes  = df.ParentschoolSatisfaction.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%')
plt.title("Parent school Satisfaction in Data", fontsize = 15)
plt.legend()
plt.show()

#### Good evaluation makes up 60.8 percent of the overall and Bad evaluation is 39.2 percent of the overall

