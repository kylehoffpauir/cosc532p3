
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
import scipy as sp
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')


# feature selection --> distance measure --> clustering
#  --> data abstraction --> final evaluation

newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


#vectorizer = TfidfVectorizer(min_df=10, max_df=0.5, tokenizer=tokenize, stop_words={'english'})
vectorizer = TfidfVectorizer(stop_words={'english'})
x = vectorizer.fit_transform(newsgroups_train.data)

"""
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
"""
# sil = []
# K = range(2,40)
# for k in K:
#     km = KMeans(n_clusters=k, max_iter=200, n_init=10)
#     km.fit(x)
#     labels = km.labels_
#     sil.append(metrics.silhouette_score(x, labels, metric = 'euclidean'))
#     print(k)
# plt.plot(k, sil, 'bx-')
# plt.xlabel('k')
# plt.ylabel('silhouette score')
# plt.title('Silhouette Method For Optimal k')
# plt.show()

"""
    Metrics explained:

    Homogeniety: similarity of cluster's elements

    Completeness : degree to which all elements belonging to certain category are found in a cluster

    V-measure : mean of homogeniety and completeness

    Silhouette score : how similar an object is to its own cluster .
    
"""
def k_means(true_k, wordcloud):
    homo, comp, v, sil = 0, 0, 0, 0
    num_iter = 5
    for i in range(num_iter):
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
        model.fit(x)
        labels = model.labels_
        forums = pd.DataFrame(list(zip(newsgroups_train.target_names,labels)),columns=['title','cluster'])
        homo += metrics.homogeneity_score(newsgroups_train.target, labels)
        comp += metrics.completeness_score(newsgroups_train.target, labels)
        v += metrics.v_measure_score(newsgroups_train.target, labels)
        sil += metrics.silhouette_score(x, labels, sample_size=1000)
    print(forums.sort_values(by=['cluster']))
    print("METRICS FOR k = %i" % true_k)
    print("Homogeneity: %0.3f" % (homo / num_iter))
    print("Completeness: %0.3f" % (comp / num_iter))
    print("V-measure: %0.3f" % (v / num_iter))
    print("Silhouette Coefficient: %0.3f" % (sil / num_iter))

    if wordcloud:
        result = {'cluster':labels, 'forum':newsgroups_train.data}
        result = pd.DataFrame(result)
        for k in range(0,true_k):
            s = result[result.cluster==k]
            text = s['forum'].str.cat(sep=' ')
            text = text.lower()
            text = ' '.join([word for word in text.split()])
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
            print('Cluster: {}'.format(k))
            print('Titles')
            titles = forums[forums.cluster == k]['title']
            print(titles.to_string(index=False))
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()


#k_means(15, False)
#k_means(20, False)
k_means(25, False)
#k_means(35, False)

# making a model based on our best k
true_k = 25
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(x)
newPost = "Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically " \
          "now. I tried to format it, but now it doesn't boot any more. Any ideas? Thanks. "
newPostVec = vectorizer.fit_transform([newPost])
newPostLabel = model.predict(newPostVec)[0]
similar_indices = (model.labels_ == newPostLabel).nonzero()[0]

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((newPostVec - x[i]).toarray())
    similar.apppend((dist, newsgroups_train.data[i]))
similar = sorted(similar)
print(len(similar))

# show the most similar post to our newpost, our middle similar, and least similar post
print(similar[0])
print(similar[len(similar)/2])
print(similar[-1])

