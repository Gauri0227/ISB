#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


# In[2]:


pwd


# In[3]:


#Import the resume from PDF to strings. Change the pdf file name to test it.

from tika import parser

resumes=['dummy','Zehai (Steve)Wang','MILI SHAH','Jay Bakshi','Aman','Raman','PM_Resume','Marketing_manager']


raw_jd = parser.from_file('JD.pdf')
jd=raw_jd['content']

raw_resume_ds1 = parser.from_file('ZehaiWang.pdf')
resume1=raw_resume_ds1['content']

raw_resume_ds2 = parser.from_file('MILISHAH.pdf')
resume2=raw_resume_ds2['content']

raw_resume_ds3 = parser.from_file('JayBakshi.pdf')
resume3=raw_resume_ds3['content']

raw_resume_ds4 = parser.from_file('Aman.pdf')
resume4=raw_resume_ds4['content']

raw_resume_ds4 = parser.from_file('Raman.pdf')
resume5=raw_resume_ds4['content']

raw_resume_PM = parser.from_file('PM.pdf')
resume_PM=raw_resume_PM['content']

raw_resume_Marketing_manager = parser.from_file('Marketing_manager.pdf')
resume_MM=raw_resume_Marketing_manager['content']


# In[4]:


documents=[jd,resume1,resume2,resume3,resume4,resume5,resume_PM,resume_MM]


# In[5]:


documents_df=pd.DataFrame(documents,columns=['documents'])


# ### Main Dataframe

# In[6]:


# removing special characters and stop words from the text
stop_words_l=stopwords.words('english')
documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )


# #### Tf-idf vectors

# In[7]:


tfidf=TfidfVectorizer(max_features=10)
tfidf.fit(documents_df.documents_cleaned)
tfidf_vectors=tfidf.transform(documents_df.documents_cleaned)


# In[8]:


tfidf_vectors=tfidf_vectors.toarray()


# In[9]:


df_test=pd.DataFrame(tfidf_vectors)


# In[10]:


similarities=np.dot(tfidf_vectors,tfidf_vectors.T)
pairwise_differences=euclidean_distances(tfidf_vectors)


# In[11]:


similarities


# In[13]:


output=similarities[0][:]


# In[14]:


# final_output=pd.DataFrame(output,resumes,columns=['Resume Score'])


# In[15]:


final_output=pd.DataFrame()

final_output['CName']=resumes
final_output['Score']=output


# In[16]:


final_output.drop(final_output.head(1).index, inplace=True)


# In[17]:


final_output.sort_values(by=['Score'],ascending=False)


# In[ ]:




