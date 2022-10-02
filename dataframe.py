from langdetect import detect,LangDetectException
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax



def LanguagedetectTweets(df):
        ##
    n = len(df)
    lanlist = []
    for i in range(n):
        lanlist.append(detect(df.iloc[i]["content"]))
    ## add language to df
    df["language_prob"] = lanlist
    return df


def GetDataframeTweets(df):
    ##
    n = len(df)
    lanlist = []
    for i in range(n):
        lanlist.append(detect(df.iloc[i]["content"]))
    ## add language to df
    df["language_prob"] = lanlist

    # sentiment
    # load model and tokenizer in English
    roberta = "oliverguhr/german-sentiment-bert"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Positive', 'Negative', 'Neutral']
    german_tweets = df["renderedContent"].tolist()
    # print(german_tweets[:10])

    positive = []
    negative = []
    neutral = []

    count = 0

    for tweet in german_tweets:
        count += 1
        print(count)
        tweet_words = []
        for word in tweet.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
        
            elif word.startswith('http'):
                word = "http"
            tweet_words.append(word)

        tweet_proc = " ".join(tweet_words)

        encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')

        output = model(**encoded_tweet)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        positive.append(scores[0])
        negative.append(scores[1])
        neutral.append(scores[2])

    df_result = pd.DataFrame(list(zip(positive, negative, neutral)), columns =['Positive', 'Negative', 'Neutral'])
    df = pd.concat([df, df_result], axis=1)
    return df



def GetDataframeStores(df_play, df_app):
    ##
    app_liste = ["date", "title", "review", "rating"]
    df_app = df_app[app_liste]
    df_app["content"] = df_app["title"] + ": " + df_app["review"]
    df_app = df_app[["rating", "content", "date"]]

    # Playstore DatenFarme bearbeiten
    play_list = ['at', 'score', "content"]
    df_play = df_play[play_list]
    df_play = df_play.rename(columns={"score": "rating", "at": "date"})
    df = pd.concat([df_app, df_play], axis = 0)

    df["content"].fillna("",inplace=True)
    n = len(df)
    lanlist = []
    for i in range(n):
        try:
            # print(str(df.iloc[i]["content"]))
            # print(detect(str(df.iloc[i]["content"])))
            lanlist.append(str(detect(df.iloc[i]["content"])))
        except LangDetectException:
            lanlist.append(np.nan)
    ## add language to df
    df["language_prob"] = lanlist
    return df