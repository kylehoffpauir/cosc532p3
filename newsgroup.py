
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups_vectorized
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt')


# feature selection --> distance measure --> clustering
#  --> data abstraction --> final evaluation
# https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
# https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04
# https://towardsdatascience.com/clustering-documents-with-python-97314ad6a78d
# https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
# https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/
# https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/


newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))


pprint(list(newsgroups_train.target_names))

#pprint(list(newsgroups_train.data))
# converting raw text to bag of words
"""
vectorizer = CountVectorizer(min_df= 1)

content = newsgroups_train.data
x = vectorizer.fit_transform(content)
print(x.shape)
print(vectorizer.get_feature_names_out())
"""

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

english_stemmer = nltk.stem.SnowballStemmer('english')
vectorizer = TfidfVectorizer(min_df=10, max_df=0.5, tokenizer=tokenize, stop_words={'english'})
#vectorizer = TfidfVectorizer(stop_words={'english'})

x = vectorizer.fit_transform(newsgroups_train.data)

#print(x)

Sum_of_squared_distances = []
K = range(2,40)
for k in K:
    km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    km = km.fit(x)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# 15 20 25
true_k = 15
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(x)
labels=model.labels_
forums=pd.DataFrame(list(zip(newsgroups_train.target_names,labels)),columns=['title','cluster'])
print(forums.sort_values(by=['cluster']))

"""


    Homogeniety: degree to which clusters contain element of the same class

    Completeness : degree to which all elements belonging to certain category are found in a cluster

    V-measure : mean of homogeniety and completeness

    Silhouette score : how similar an object is to its own cluster .
    
"""
print("Homogeneity: %0.3f" % metrics.homogeneity_score(newsgroups_train.target, labels))
print("Completeness: %0.3f" % metrics.completeness_score(newsgroups_train.target, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(newsgroups_train.target, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(x, labels, sample_size=1000))

"""
result={'cluster':labels,'forum':newsgroups_train.data}
result=pd.DataFrame(result)
for k in range(0,true_k):
    s=result[result.cluster==k]
    text=s['forum'].str.cat(sep=' ')
    text=text.lower()
    text=' '.join([word for word in text.split()])
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    print('Cluster: {}'.format(k))
    print('Titles')
    titles=forums[forums.cluster==k]['title']
    print(titles.to_string(index=False))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

"""


