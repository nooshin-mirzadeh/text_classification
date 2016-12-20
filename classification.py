import numpy as np
#import panda as pd
import sklearn as sk
#import gensim
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score


# train pos data
def loadTrainData(fn, emotion):
  x_in, y_lable = [], []
  with open(fn) as f:
    for line in f:
      x_in.append(line)
  x_in = np.array(x_in)
  y = np.full(x_in.shape, emotion)
  print(x_in.shape)
  return x_in, y



# feature extraction

## Averaging word vectors for all words in a text
class MeanEmbeddingVectorizer(object):
  def __init__(self, word2vec):
    self.word2vec = word2vec
    self.dim = word2vec.shape

  def fit(self, X, y):
    return self

  def transform(self, X):
    return np.array([
          np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                  or [np.zeros(self.dim)], axis=0 ) 
          for words in X
        ])


## tf-idf weight
class TfidfEmbeddingVectorizer(object):
  def __init__(self, word2vec):
    self.word2vec = word2vec
    self.word2weight = None
    self.dim = word2vec.shape

  def fit(self, X, y):
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit(X)
    max_idf = max(tfidf.idf_)
    self.word2weight = defaultdict(
      lambda: max_idf,
      [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    return self

  def transform(self, X):
    return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                      for w in words if w in self.word2vec] or 
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def main():
  
  X1, y1 = loadTrainData('twitter-datasets/train_pos.txt','1')
  print(X1)
  print(y1)
  
  X2,y2 = loadTrainData('twitter-datasets/train_neg.txt','-1')
  print(X2)
  print(y2)

  X = np.concatenate((X1, X2), axis=0)
  y = np.concatenate((y1, y2), axis=0)
  print(X,y)

  w2v = np.load('results/embeddings.npy')
  
  print(w2v)

  etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
  

  etree_w2v_tfidf = Pipeline([
    ("word2vex vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra tress", ExtraTreesClassifier(n_estimators=200))])
  
  score = cross_val_score(etree_w2v, X, y, cv=5).mean()
  print(score)
    
if __name__ == '__main__':
    main()


# let X
# model = gensim.models.Word2Vec(X, size=100)
# w2v = dict(zip(model.index2word, model.syn))
