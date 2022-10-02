#%%
from langdetect import detect_langs,detect
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
df = pd.read_csv(r"C:\Users\tmand\OneDrive\Dokumente\Github\MaKEathonPostFinance\data\tweets_ats.csv")
# %%
print(df.head())
#%%
n = len(df)
lanlist = []
for i in range(n):
    lanlist.append(detect(df.iloc[i]["content"]))

df["language_prob"] = lanlist
# %%

unique, counts = np.unique(df["language_prob"], return_counts=True)
count_sort_ind = np.argsort(-counts)
unique= unique[count_sort_ind]
counts = counts[count_sort_ind]
# color = ['blue',"green"]
plt.bar(unique,counts)
plt.xlabel("Count of Language")
plt.ylabel("Languages in Tweets")
plt.show()
plt.savefig('output.png')

# %%
