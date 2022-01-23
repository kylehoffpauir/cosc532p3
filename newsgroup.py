from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# converting raw text to bag of words
vectorizer = CountVectorizer(min_df= 1)

# feature selection --> distance measure --> clustering
#  --> data abstraction --> final evaluation
# https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04



