#%%
import pandas as pd 
from nltk.corpus import stopwords
import pickle as pkl
import dataframe
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import yake
import nltk
nltk.download()

# %%
df = pd.read_csv(r"C:\Users\tmand\OneDrive\Dokumente\Github\MaKEathonPostFinance\data\tweets_ats.csv")
df = dataframe.GetDataframeTweets(df)

# Daten einlesen
app_store = pkl.load(open("./data/app_store.pkl", "rb"))
app_store = pd.json_normalize(app_store)

play_store = pkl.load(open("./data/play_store.pkl", "rb"))
play_store = pd.json_normalize(play_store)
df = dataframe.GetDataframeStores(play_store,app_store)

# %%
df = df[(df["language_prob"] == "de")& (df["rating"]==1)]
#%%
german_stop_words = stopwords.words('german')

#%%
dataset = df["content"]
dataset.head(10)
# %%
kw_extractor = yake.KeywordExtractor()
keywords=[]
language = "de"
max_ngram_size = 5
deduplication_threshold = 0.9
numOfKeywords = 1
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
for i in dataset:
    temp= custom_kw_extractor.extract_keywords(i)
    
    for j in range(numOfKeywords):
        try:
            temp=temp[j][0].split()
            keywords.extend(temp)
        except IndexError:
            pass



# %%
for kw in keywords:
    print(kw)
keywords = [word.lower() for word in keywords]
# %%
dict_keywords = []
# for corps in keywords:
values, counts = np.unique(keywords, return_counts=True)
print(values,counts)

dict_keywords = dict(zip(values, counts))

# %%
dict_keywords = {k: v for k, v in dict_keywords.items() if v >= len(df)/100}
dict_keywords = {k: v for k, v in dict_keywords.items() if k not in german_stop_words}

# %%
import matplotlib.pyplot as plt

wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(dict_keywords)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
# %%
