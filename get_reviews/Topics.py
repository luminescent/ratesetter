
# coding: utf-8

# In[23]:

from pre_process import get_rate_setter_dictionary_corpus
from gensim import corpora, models, similarities
from gensim.models import LdaMulticore


# In[24]:

dictionary, corpus = get_rate_setter_dictionary_corpus()


# In[19]:

print(dictionary)


# In[20]:

lda = LdaMulticore(corpus=corpus, num_topics=5, id2word=dictionary, passes=100)


# In[21]:

lda.show_topics()


# In[22]:

lda.print_topics()


# In[10]:

dictionary[lda.get_topic_terms(0)[0][0]]


# In[ ]:



