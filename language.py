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
plt.bar(unique,counts)
plt.show()
# %%
plt.hist(df["language_prob"])
for i, v in enumerate(counts):
    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold') 
plt.show()
# %%

