from langdetect import detect,LangDetectException
import pandas as pd
import numpy as np

def GetDataframeTweets(df):
    ##
    n = len(df)
    lanlist = []
    for i in range(n):
        lanlist.append(detect(df.iloc[i]["content"]))
    ## add language to df
    df["language_prob"] = lanlist
    return df


def GetDataframeStores(df_play, df_app):
    ##
    app_liste = ["date", "title", "review", "rating"]
    df_app = df_app[app_liste]
    df_app["content"] = df_app["title"] + ": " + df_app["review"]
    df_app = df_app[["rating", "content"]]

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