#code done on google colab 

#Importing the libraries needed fot the project
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchvision.datasets.utils import download_url

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
%matplotlib inline

#Importing the dataset
data = pd.read_csv('/content/Suicides in India 2001-2012.csv')

#Taking out information of dataset
data.info()

#First 5 rows of dataframe
data.head()

#Checking for null values in dataset
data.isna().sum() #As per the observation there are no null values in dataset

data.nunique()

#Let's plot for age-group in the dataset
sns.countplot(data['Age_group'])

#Let's plot for gender in the dataset
sns.barplot(data.groupby('Gender').sum()['Total'].index,data.groupby('Gender').sum()['Total'])

#Data in Age_group
data['Age_group'].value_counts()

#Data in Type of reasons for suicide
data['Type'].value_counts()

#Number of males and females
data['Gender'].value_counts()

#Let's sort the data by year
print("The number of suicide cases between 2001 - 2012 are as follows\n",data.groupby("Year")["Total"].sum())

data.groupby("Year")["Total"].sum().plot(kind="line")
#According to the plot as the number of years increased the rate of suicide increased by time

plt.figure(figsize = (30,10))
ax=sns.countplot(y = "State", data = data)
ax.set_title("Suicide deaths per state", fontsize = 25)
plt.xlabel("Number",fontsize=17)
plt.ylabel("States", fontsize=18)
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), color='black', size=15, ha="center")

#Number of Deaths per state
data['State'].value_counts()

#Variable to store the dataframe of states with the total cases
states_word = pd.DataFrame(data.groupby(["State"])["Total"].sum()).reset_index()
states_word

#WordCloud of the dataset
from wordcloud import WordCloud as Wc
def Wordcloud():
  temp = {}
  for i in states_word["State"].values:
      temp[i]=int(states_word[states_word["State"]==i].Total)

  #Plotting using wordcloud
  WC = Wc(width=1500,height=1000,background_color='white').generate_from_frequencies(temp)
  plt.imshow(WC, interpolation='mitchell')
  plt.axis("off")
  plt.show()
Wordcloud()

#Age vs Gender
import plotly.express as ptx

fig = ptx.scatter(data, x="Age_group", y="Total", color="Gender",marginal_x="box", marginal_y="violin",
                   color_discrete_sequence=['cyan','blueviolet'])

fig.update_layout(
    xaxis_title="Age Group",
    yaxis_title="Suicide count",
)
fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))

fig.show()

fig = px.histogram(data, x="Age_group", y="Total", color="Gender", facet_col="Year",
                  marginal="box")
fig.show()

age_group = data[data['Age_group'] != '0-100+']

plt.figure(figsize=(15,10))
sns.barplot(age_group.groupby('Age_group').sum()['Total'].sort_values().index ,age_group.groupby('Age_group').sum()['Total'].sort_values().values)
plt.show()

#typecode vs gender to see the social status if the person is married or single or widow and so on
general_status = pd.DataFrame(data[data["Type_code"]=="Social_Status"].groupby(["Type","Gender"])["Total"].sum()).reset_index()
sns.barplot(x="Type", y="Total",hue="Gender", palette="deep", data=general_status)

#typecode vs gender to see if profession of the person
general_status = pd.DataFrame(data[data["Type_code"]=="Professional_Profile"].groupby(["Type","Gender"])["Total"].sum()).reset_index()
g = sns.barplot(x="Total", y="Type",hue="Gender", palette="rocket", data=general_status)

Proff_status = data[data['Type_code'] == 'Professional_Profile']
title = data['Type_code'].values[0]
suicide = Proff_status.groupby('Type').sum()['Total'].sort_values(ascending=False)

plt.figure(figsize=(15,10))
sns.barplot(y=suicide.index , x=suicide.values)
plt.show()

Causes_of_Death = data[data['Type_code'] == 'Causes']
title = data['Type_code'].values[0]
D = Causes_of_Death.groupby('Type').sum()['Total'].sort_values(ascending=False)

plt.figure(figsize=(15,10))
sns.barplot(y=D.index , x=D.values)
plt.show()

Death_by = data[data['Type_code'] == 'Means_adopted']
title = data['Type_code'].values[0]
D = Death_by.groupby('Type').sum()['Total'].sort_values(ascending=False)

plt.figure(figsize=(15,10))
sns.barplot(y=D.index , x=D.values)
plt.show()
