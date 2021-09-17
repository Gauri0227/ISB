# -*- coding: utf-8 -*-
"""FP2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YTezDIUXfVnp_mERMBzZQTQet2yDSCGV
"""

#from google.colab import drive
#drive.mount('/content/drive')

import os

# Getting the current work directory (cwd)
thisdir = os.getcwd()
thisdir = "Resume_Supply_Chain"
#jddir = str(os.getcwd())+"/drive/MyDrive/Content/"
# r=root, d=directories, f = files
word_files = []
pdf_files = []
for r, d, f in os.walk(thisdir):
    for file in f:
        if file.endswith(".pdf"):
          pdf_files.append(file)
        if file.endswith(".docx"):
          word_files.append(file)

!pip install python-docx
!pip install tika
!pip install docx2python
!pip install docx2txt

from docx import Document
from tika import parser
from docx2python import docx2python
import re
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from nltk import FreqDist
from matplotlib.pyplot import figure
import pickle
import pandas as pd
import numpy as np

name=[]
content=[]
df=pd.DataFrame()
name.append("jd")
print(jddir)
content.append(parser.from_file("JD.docx")["content"])
for i in range(len(pdf_files)):
  name.append(re.sub(r'[^a-zA-Z]','',pdf_files[i].split(".")[0]))
  content.append(parser.from_file(thisdir+"/"+pdf_files[i])["content"])
for j in range(len(word_files)):
  name.append(re.sub(r'[^a-zA-Z]','',word_files[j].split(".")[0]))
  content.append(parser.from_file(thisdir+"/"+word_files[j])["content"])

df["Names"]=name
df["Resume"]=content

nltk.download("stopwords")
stop_words_l=stopwords.words('english')

df["Resume"]=df["Resume"].apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l))

tfidf=TfidfVectorizer(max_features=10)
tfidf.fit(df["Resume"])
tfidf_vectors=tfidf.transform(df["Resume"])

tfidf_vectors=tfidf_vectors.toarray()

df_test=pd.DataFrame(tfidf_vectors)

similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
pairwise_differences=euclidean_distances(tfidf_vectors)

output=similarities[0][:]

final_output=pd.DataFrame()

final_output['CName']=name
final_output['Score']=output

final_output.drop(final_output.head(1).index, inplace=True)

final_output.sort_values(by=['Score'],ascending=False)
