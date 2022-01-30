from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups_vectorized
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
newsgroups_train = fetch_20newsgroups(subset='train')
# newsgroups_test = fetch_20newsgroups(subset='test')


# pprint(list(newsgroups_train.target_names))
# converting raw text to bag of words
"""
vectorizer = CountVectorizer(min_df= 1)

content = newsgroups_train.data
x = vectorizer.fit_transform(content)
print(x.shape)
print(vectorizer.get_feature_names_out())
"""
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words={'english'})
X = vectorizer.fit_transform(newsgroups_train.data)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Sum_of_squared_distances = []
"""
K = range(2,10)
for k in K:
    km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
"""

true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(X)
labels=model.labels_
forums=pd.DataFrame(list(zip(newsgroups_train.target_names,labels)),columns=['title','cluster'])
print(forums.sort_values(by=['cluster']))


# feature selection --> distance measure --> clustering
#  --> data abstraction --> final evaluation
# https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04
# https://towardsdatascience.com/clustering-documents-with-python-97314ad6a78d


from wordcloud import WordCloud
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




