# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

## Introduction
'''
This file contains the Happiness Score for 153 countries along with the factors used to explain the
score. The Happiness Score is explained by the following factors: GDP per capita; Healthy Life
Expectancy; Social support; Freedom to make life choices; Generosity; Corruption Perception;
Residual error.
'''
# Load dataframe
df_HR20 = pd.read_csv("HappinessReport2020.csv")
df_HR20.head()

# remove columns that we will not use
df_HR20 = df_HR20.drop(['Standard error of ladder score', 'upperwhisker', 'upperwhisker', 'lowerwhisker', 'Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption'], axis=1)

df_HR20.columns

## Renaming Columns

df_HR20.rename(columns = {'Country name':'Country', 'Regional indicator':'Region', 'Ladder score': 'Ladder',
                  'Standard error of ladder score':'Standard Error', 'Logged GDP per capita':'Logged GDPPC',
                  'Social support':'Social Support', 'Healthy life expectancy':'Life Expectancy',
                  'Freedom to make life choices':'Freedom', 'Perceptions of corruption': 'Corruption'},inplace = True)

## Adding Columns to the Data

# Add a 'Rank' column to our data (luckily for us, the rows are already ordered from happiest to unhappiest)
df_HR20['Rank'] = range(1, 154)

quartile_index = np.percentile(df_HR20['Rank'], [25, 50, 75])
quartiles = pd.Series(df_HR20['Rank'].map(lambda x:(np.searchsorted(quartile_index, x) + 1)), name = 'Quartile')
df_HR20 = pd.concat([df_HR20, quartiles], axis = 1)

###### Later on, we may find it useful to have the countries split up into percentiles. Let's create a 'Quartile' column that denotes the quartile each country belongs to according to its overall happiness score/rank.

# Check our updated data with the new 'Rank' and 'Quartile' columns
print(df_HR20.head())

# Set style
plt.style.use('seaborn-whitegrid')

## Exploring 'Region' feature

fig = plt.figure(figsize = (15, 14))
ax = plt.axes()

countplot = sns.countplot('Region', data = df_HR20, saturation = 0.8, palette = 'tab10')
countplot.set_xticklabels(countplot.get_xticklabels(),fontsize = 12 ,rotation = 90)
countplot.set_title("Countplot by Region",fontsize = 13 ,y = 1.05);

#### Distribute data highly imbalanced in North America and ANZ, East Asia is lower than with another and Sub-Sahara Atrica is highest 

fig = plt.figure(figsize = (15, 14))
ax = plt.axes()

stacked_countplot = sns.countplot('Region', data = df_HR20, hue = 'Quartile')
stacked_countplot.set_xticklabels(countplot.get_xticklabels(), rotation = 90)
stacked_countplot.set_title("Countplot of Quartiles for Each Region", y = 1.05);
ax.legend(loc = "upper left", title = 'Quartile', title_fontsize = 18);

##### There a couple of pretty clear patterns we can see here. For example, Western Europe, North America, and Latin America all seem to be places where happiness is quite high, whereas places like South Asia and Sub-Saharan Africa appear to be quite unhappy.

### Exploring 'Logged GDPPC' feature

#### boxplot GDP of Region

plt.figure(figsize = (15, 14))
boxplot = sns.boxplot(x="Region", y= "Logged GDPPC", data=df_HR20)
boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation = 90)
plt.title("Boxplot Region - Logged GDPPC ", fontsize = 15)
plt.show()

#### GDP of all Region are high and Western Europe, Latin America and Caribbean, Southeast Asia, Sub-Saharan Africa have outlier, special Sub-Saharan Africa GDP spread data evenly from 7 to 9    

### Pie chart

labels = df_HR20.Region.unique()

sizes = df_HR20.Region.value_counts().values

plt.figure(figsize =(10,10))
plt.pie(sizes, labels= labels,autopct= '%1.1f%%',startangle=1)
plt.title('Region in Data', fontsize = 17)
plt.legend(bbox_to_anchor=(0.9, 0.4, 0.6, 0.1))
plt.show()

#### Western Europe accounts for the highest percentage, South Asia is lowest percentage

# Correlation

# Gather columns corresponding to the six measured values (Logged GDP per capita, social support, etc.)
feature_cols = ['Logged GDPPC', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']

pairplot = sns.pairplot(df_HR20, hue = 'Quartile',vars = feature_cols, corner = False)
pairplot.fig.suptitle("Pairplot of the 6 Happiness Metrics", fontsize = 24, y = 1.05);
plt.show()

###### In some metrics, we can see a stark difference between countries in the first quartile versus countries in the other quartiles (especially 'Corruption'). On the other hand, other measurements seem to be much less relevant in distinguishing happier countries from the rest (look at 'Generosity')

df_HR20_ft = pd.concat([df_HR20['Ladder'], df_HR20[feature_cols]], axis = 1)
plt.style.use('seaborn-white')
plt.rcParams['figure.figsize'] = (12,8)
sns.heatmap(df_HR20_ft.corr(), cmap = 'copper', annot = True)


plt.show()

##### It looks like the Logged GDPPC, Social Support, and Life Expectancy metrics all have a relatively high correlation with the overall score a country received. Also, these factors each seem to have a pretty high correlation with each other (e.g., Social Support is well correlated with Life Expectancy, and so on). On the other end of the spectrum, Generosity does not seem to have a sizeable correlation with any other measurement, including the Ladder score.





