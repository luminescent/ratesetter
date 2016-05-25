
# coding: utf-8

# # <p style="text-align: center;"> Title </p>

# In[1]:

# general imports 
from pre_process import get_all_reviews, get_rate_setter_dictionary_corpus, pretty_print_html 
from gensim import corpora, models, similarities
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
from pyLDAvis.gensim import prepare
from random import randint
import pyLDAvis 
from IPython.display import HTML
from pre_process import ReviewsRetriever, ReviewNormalizer


# #### We get the reviews by scraping TrustPilot's website and we take a look at a couple of them

# In[5]:

rr = ReviewsRetriever()
reviews = rr.get(cached=True)
pretty_print_html([reviews[randint(0, len(reviews))], reviews[randint(0, len(reviews))]])


# #### We now normalize the reviews: remove punctuation, make everything lowercase and stem. Then we take a look at another couple of them. 

# In[10]:

rn = ReviewNormalizer()
normalized_reviews = [rn.tokenize(r)
                      for r in reviews]
pretty_print_html([" ".join(normalized_reviews[randint(0, len(normalized_reviews))]), 
                   " ".join(normalized_reviews[randint(0, len(normalized_reviews))])])


# #### Training the model (this might take a while...)

# In[12]:

dictionary = corpora.Dictionary(normalized_reviews)
corpus = [dictionary.doc2bow(r)
          for r in normalized_reviews]
lda = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=100)


# #### Prepare data and visualize!

# In[14]:

prepared_data = prepare(lda, corpus, dictionary)
pyLDAvis.display(prepared_data)


# In[ ]:



