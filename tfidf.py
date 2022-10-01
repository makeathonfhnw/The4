#%%
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd 
from nltk.corpus import stopwords
import pickle as pkl
import dataframe
import numpy as np
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

x = df[df['content'].str.contains("Bitte")]
# x = df[df["content"].str.contains("digitalisierung")]["content"]


#%%
german_stop_words = stopwords.words('german')

#%%
dataset = df["content"]
dataset.head(10)
# %%
tfIdfTransformer = TfidfTransformer(use_idf=True)
countVectorizer = CountVectorizer(stop_words = german_stop_words)
wordCount = countVectorizer.fit_transform(dataset)
newTfIdf = tfIdfTransformer.fit_transform(wordCount)
dfidf = pd.DataFrame(newTfIdf[0].T.todense(), index=countVectorizer.get_feature_names(), columns=["TF-IDF"])
dfidf = dfidf.sort_values('TF-IDF', ascending=False)
print (dfidf.head(50))
# %%

import yake
kw_extractor = yake.KeywordExtractor()
keywords=[]
language = "de"
max_ngram_size = 5
deduplication_threshold = 0.5
numOfKeywords = 1
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
for i in dataset:
    temp= custom_kw_extractor.extract_keywords(i)
    for j in range(2):
        try:
            # print(temp[j][0])
            keywords.append(temp[j][0])
        except IndexError:
            pass



# %%
for kw in keywords:
    print(kw)
# %%
dict_keywords = []
for corps in keywords:
    values, counts = np.unique(corps, return_counts=True)
    print(values,counts)
    dicttemp = dict(zip(values, counts))
    dict_keywords.append(dicttemp)
# dict_corpus.append(dicttemp)
# %%
dict_keywords[2]
# %%
