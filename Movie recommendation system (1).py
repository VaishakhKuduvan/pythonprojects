#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer - This is used to convert text data into numerical values
from sklearn.metrics.pairwise import cosine_similarity
import os


# In[2]:


os.chdir('X:\\Movie Recommendation system')


# In[3]:


display (os.getcwd())


# In[4]:


movies_data =pd.read_csv('movies.csv')
movies_data.head()



# In[5]:


display (movies_data.shape)


# In[6]:


selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[7]:


display (movies_data.info())


# In[8]:


display (movies_data.isna().sum())


# In[9]:


display (movies_data[selected_features].head())


# In[10]:


display (movies_data[selected_features].isna().sum())


# In[11]:


for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
display (movies_data.head())


# In[12]:


display (movies_data[selected_features].isna().sum())


# In[13]:


combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
display (combined_features)


# In[14]:


vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
display (feature_vectors.shape)
print (feature_vectors)


# In[15]:


similarity = cosine_similarity(feature_vectors)
print  (similarity )


# In[16]:


display(similarity.shape)


# In[17]:


pd.DataFrame(similarity).to_csv('20_April_Similarity.csv')


# In[18]:


movie_name = input(' Enter your favourite movie name : ')


# In[19]:


list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[20]:


len(list_of_all_titles)


# In[21]:


find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[22]:


close_match = find_close_match[0]
print(close_match)


# In[23]:


index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[24]:


similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[25]:


len(similarity_score)


# In[26]:


sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[27]:


print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[34]:


movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:




