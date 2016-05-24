from bs4 import BeautifulSoup
import urllib.request as urlr
from bs4.element import NavigableString
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from gensim import corpora, models, similarities
from gensim.models.ldamodel import LdaModel


class ReviewNormalizer(object):
    def __init__(self):
        self._stemmer = PorterStemmer()
        self._stop_words = set(stopwords.words('english'))
        self._stop_words.add("ratesetter")

    def tokenize(self, sentence):
        words = re.sub("[^a-zA-Z]", " ", sentence)\
            .lower()\
            .split()
        return [self._stemmer.stem(w)
                for w in words
                if w not in self._stop_words]


def get_content_from_url(url):
    return BeautifulSoup(urlr.urlopen(url).read(), "lxml")


def get_reviews(page_content):
    # this preserves the order
    title_elements = page_content.findAll("a", {"class": "review-title-link"})
    titles = [str(text.string)
              for text in title_elements]
    review_text_elements = page_content.findAll("div", {"class": "review-body"})
    review_texts = [[str(paragraph)
                     for paragraph in review.contents
                     if type(paragraph) is NavigableString]
                    for review in review_text_elements]

    l = min(len(titles), len(review_texts))
    return [" ".join([titles[x]] + review_texts[x])
            for x in range(l)]


def get_rate_setter_dictionary_corpus():
    pages = ["https://uk.trustpilot.com/review/ratesetter.com"] +\
            ["https://uk.trustpilot.com/review/ratesetter.com?page={0}".format(x) for x in range(2, 69)]

    pages_content = [get_content_from_url(p)
                     for p in pages]
    reviews = [r
               for pc in pages_content
               for r in get_reviews(pc)]
    rn = ReviewNormalizer()
    normalized_reviews = [rn.tokenize(r)
                          for r in reviews]
    dictionary = corpora.Dictionary(normalized_reviews)
    corpus = [dictionary.doc2bow(r)
              for r in normalized_reviews]
    return dictionary, corpus











