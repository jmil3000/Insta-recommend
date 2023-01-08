import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

jd = pd.read_csv("Instagram data.csv")
jd = jd[["Caption" , "Hashtags"]]
#print(jd.head())

captions = jd["Caption"].tolist()
tfidf = text.TfidfVectorizer(input=captions, stop_words="english")
matrix_ = tfidf.fit_transform(captions)
sim_ = cosine_similarity(matrix_)

def recommend_post(i):
  return ", ".join(jd["Caption"].loc[i.argsort()[-5:-1]])

jd["Recommended Post"] = [recommend_post(i) for i in sim_]
#print(jd.head())
selection = int(input("Enter associated number for instagram recommendation:"))
list1 = range(0,119,1)
if selection in list1:
    print(jd["Recommended Post"][int(selection)])
else:
    print("Please enter valid whole in 1-118 range")