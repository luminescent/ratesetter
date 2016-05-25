from bs4 import BeautifulSoup
import urllib.request as urlr
from bs4.element import NavigableString
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from IPython.display import HTML
import joblib as jb


class ReviewNormalizer(object):
    """
    Turns a sentence into its tokenized version: no punctuation, no stopwords and stemmed.
    """
    def __init__(self):
        self._stemmer = PorterStemmer()
        self._stop_words = set(stopwords.words('english'))
        self._stop_words.add("ratesetter")
        self._stop_words.add("setter")

    def tokenize(self, phrase):
        """
        Tokenizes a sentence
        :param phrase: string that contains a phrase (sentences separated by punctuation or space).
        :return: list of stemmed words
        """
        words = re.sub("[^a-zA-Z]", " ", phrase)\
            .lower()\
            .split()
        return [self._stemmer.stem(w)
                for w in words
                if w not in self._stop_words]


class ReviewsRetriever(object):
    def __init__(self):
        self._cache_file = "reviews.bin"

    def get(self, cached=False):
        if cached:
            return jb.load(self._cache_file)
        else:
            pages = ["https://uk.trustpilot.com/review/ratesetter.com"] +\
                    ["https://uk.trustpilot.com/review/ratesetter.com?page={0}".format(x) for x in range(2, 69)]

            pages_content = [get_content_from_url(p)
                             for p in pages]
            reviews = [r
                       for pc in pages_content
                       for r in self._parse_reviews(pc)]
            jb.dump(reviews, self._cache_file)
            return reviews

    def _parse_reviews(self, page_content):
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


def get_content_from_url(url):
    return BeautifulSoup(urlr.urlopen(url).read(), "lxml")


def pretty_print_html(items):
    return HTML('<div class="output_subarea output_html rendered_html output_result" style="padding-right: 100px">{0}</div>'
                    .format("</br></br>".join(items)))









