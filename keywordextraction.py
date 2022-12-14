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
df = pd.read_csv(r"C:\Users\tmand\OneDrive\Dokumente\Github\MaKEathonPostFinance\data\tweets_hashtag_withsentiment.csv")
subset_tweets = df[(df["Negative"] >=.8)]
df = df[(df["language_prob"] =="de")]
df.head()
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
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords =1
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

wordcloud = WordCloud(width = 1000, height = 500, background_color="white", max_words=1000).generate_from_frequencies(dict_keywords)
# wordcloud = WordCloud(width = 1000, height = 500, background_color="white", max_words=1000, mask=mask).generate_from_frequencies(dict_keywords)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.savefig("wordcloud_twitter_hashtag")
#%%
df = pd.read_csv(r"C:\Users\tmand\OneDrive\Dokumente\Github\MaKEathonPostFinance\data\tweets_withsentiment.csv")
subset_tweets = df[(df["Negative"] >=.8)]
df = df[(df["language_prob"] =="de")]

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
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords =1
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

wordcloud = WordCloud(width = 1000, height = 500, background_color="white", max_words=1000).generate_from_frequencies(dict_keywords)
# wordcloud = WordCloud(width = 1000, height = 500, background_color="white", max_words=1000, mask=mask).generate_from_frequencies(dict_keywords)
plt.figure(figsize=(15,8))
plt.axis("off")
plt.imshow(wordcloud)
plt.savefig("wordcloud_twitter")

# %%

# # Daten einlesen
app_store = pkl.load(open("./data/app_store.pkl", "rb"))
app_store = pd.json_normalize(app_store)

play_store = pkl.load(open("./data/play_store.pkl", "rb"))
play_store = pd.json_normalize(play_store)
df = dataframe.GetDataframeStores(play_store,app_store)

#%%
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
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords =1
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

wordcloud = WordCloud(width = 1000, height = 500, background_color="white", max_words=1000).generate_from_frequencies(dict_keywords)
# wordcloud = WordCloud(width = 1000, height = 500, background_color="white", max_words=1000, mask=mask).generate_from_frequencies(dict_keywords)
plt.figure(figsize=(15,8))
plt.axis("off")
plt.imshow(wordcloud)
plt.savefig("wordcloud_stores")
# %%
## Hastags Keywords

df = pd.read_csv(r"C:\Users\tmand\OneDrive\Dokumente\Github\MaKEathonPostFinance\data\tweets_hashtag_withsentiment.csv")
subset_tweets = df[(df["Negative"] >=.8)]
df = df[(df["language_prob"] =="de")]


dict_keywords = []

# df = df[df['hashtags'].notna()]
# x = list(df["hashtags"].split(","))
# x

df_words = df["hashtags"].apply(pd.Series)
df_words = df_words[0].str.split(",",expand=True)
# df_words.fillna("",inplace=True)
# df = df.replace('old character','new character', regex=True)
for i in range(len(df_words.columns)):
    df_words[i] = df_words[i].str.replace('[','')
    df_words[i] = df_words[i].str.replace(']','')
    df_words[i] = df_words[i].str.replace('\'','')
    df_words[i] = df_words[i].str.replace(' ','')



df_words = df_words.values.tolist()

word_list = []
for i in df_words:
    print(i)
    for j in i:
        print(j)
        if j is not None:
            word_list.append(j)


word_list
values, counts = np.unique(word_list, return_counts=True)
print(values,counts)

dict_keywords = dict(zip(values, counts))

# %%
dict_keywords = {k: v for k, v in dict_keywords.items() if v >= len(df_words)/100}
dict_keywords = {k: v for k, v in dict_keywords.items() if k not in german_stop_words}
dict_keywords =  {k.lower(): v for k, v in dict_keywords.items()}
# %%
import matplotlib.pyplot as plt
from PIL import Image
mask = np.array(Image.open(r"C:\Users\tmand\OneDrive\Dokumente\Github\MaKEathonPostFinance\pics\twitter.png"))

wordcloud = WordCloud(width = 1000, height = 500, background_color="white", max_words=1000).generate_from_frequencies(dict_keywords)
plt.figure(figsize=(15,8))
plt.axis("off")
plt.imshow(wordcloud)
plt.savefig("wordcloud_tweets_hashtags")
    # %%
