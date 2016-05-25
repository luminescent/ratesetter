
# coding: utf-8

# In[4]:

from pre_process import get_rate_setter_dictionary_corpus
from gensim import corpora, models, similarities
from gensim.models import LdaMulticore
from gensim.ldamodel import ldamodel 
from pyLDAvis.gensim import prepare


# In[6]:

dictionary, corpus = get_rate_setter_dictionary_corpus()


# In[16]:

print(dictionary)
#print(corpus)


# In[8]:

lda = LdaMulticore(corpus=corpus, num_topics=5, id2word=dictionary, passes=100)


# In[10]:

lda.print_topics()


# In[11]:

dictionary[lda.get_topic_terms(0)[0][0]]


# In[12]:

prepared_data = prepare(lda, corpus, dictionary)


# In[14]:

import pyLDAvis 
# vis_data = pyLDAvis.prepare(**prepared_data)
pyLDAvis.display(prepared_data)


# In[ ]:



