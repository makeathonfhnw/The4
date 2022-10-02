# %%
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from plotly.subplots import make_subplots

# %%
df_twitter = pd.read_csv('data\german_tweets_sentimental_analysed.csv')
df_appstores = pd.read_csv('data/app_play_trainset.csv')
df_appstores = df_appstores[df_appstores['language_prob'] == 'de']

# %%
data = pd.read_csv('data\play_store.csv')

# %%
df_powerpoint = df_twitter[['content', 'language_prob', 'Positive', 'Negative', 'Neutral']]

# %%
df_twitter

# %%
#Tweet Example
tweet =  df_powerpoint['content'][11]
rating = df_powerpoint['Negative'][11]
print(' TWEET:',tweet, '--> NEGATIVE RATING  :', rating )


# %%
df_appstores['date'].min(), df_twitter['date'].min()

# %%
#plot df_twitter_negative on y and time on x
fig = go.Figure()
fig.add_trace(go.Histogram(x=df_twitter['date'], y=df_twitter['Negative'], name='Twitter Negative'))
fig.add_trace(go.Histogram(x=df_appstores['date'], y=df_appstores['Negative'], name='Appstores Negative'))
fig.update_layout(title='Negative Sentimental Analysis', xaxis_title='Date', yaxis_title='Sentimental Rating')
fig.show()

# %%
#Line Chart for Negative Sentimental Analysis
date_low = '2021-01-01'
date_high = '2021-03-01'
df_twitter_date = df_twitter[(df_twitter['date'] > date_low)& (df_twitter['date'] < date_high)]
df_appstores_date = df_appstores[(df_appstores['date'] >  date_low)& (df_appstores['date'] < date_high)]


print(len(df_twitter_date['Negative']))
print(len(df_appstores_date['Negative']))


fig = go.Figure()
fig.add_trace(go.Histogram(x=df_twitter_date['date'], y=df_twitter_date['Negative'] / len(df_twitter_date["Negative"]), name='Twitter Negative',nbinsx=10))
fig.add_trace(go.Histogram(x=df_appstores_date['date'], y=df_appstores['Negative']/ len(df_appstores_date["Negative"]), name='Appstores Negative',nbinsx=10))
fig.update_layout(title='Negative Sentimental Analysis', xaxis_title='Date', yaxis_title='Sentimental Rating')
fig.show()



# %%

# df_twitter_date = df_twitter[(df_twitter['date'] > '2021-12-16')& (df_twitter['date'] < '2021-12-24')]
# df_appstores_date = df_appstores[(df_appstores['date'] > '2021-12-16')& (df_appstores['date'] < '2021-12-24')]

# df_twitter_date["Negative"]=(df_twitter_date["Negative"]-df_twitter_date["Negative"].mean())/df_twitter_date["Negative"].std()
# df_appstores_date["Negative"]=(df_appstores_date["Negative"]-df_appstores_date["Negative"].mean())/df_appstores_date["Negative"].std()

# fig = go.Figure()
# fig.add_trace(go.Histogram(x=df_twitter_date['date'], y=df_twitter_date['Negative'], name='Twitter Negative',nbinsx=10))
# fig.add_trace(go.Histogram(x=df_appstores_date['date'], y=df_appstores['Negative'], name='Appstores Negative',nbinsx=10))
# fig.update_layout(title='Negative Sentimental Analysis', xaxis_title='Date', yaxis_title='Sentimental Rating')
# fig.show()
