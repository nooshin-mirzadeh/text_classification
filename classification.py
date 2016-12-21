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
  X, y = [], []
  with open(fn) as infile:
    for line in infile:
      text = line.split('\t')
      X.append(text.split())
  X = np.array(X)
  y = np.full(X.shape, emotion, dtype=string)
  print(X.shape)
  return X, y



def build_word_vector_matrix(vector_file, n_words):
  '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
  numpy_arrays = []
  labels_array = []
  with codecs.open(vector_file, 'r', 'utf-8') as f:
    for c, r in enumerate(f):
      sr = r.split()
      labels_array.append(sr[0])
      numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )

      if c == n_words:
        return numpy.array( numpy_arrays ), labels_array
  
  return numpy.array( numpy_arrays ), labels_array

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

  w2v = build_word_vector_matrix('', 1000000000)
  
  #print(w2v)


  mult_nb = Pipeline([ ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), ("multinomial nb", MultinomialNB())])

  etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
  

  etree_w2v_tfidf = Pipeline([
    ("word2vex vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra tress", ExtraTreesClassifier(n_estimators=200))])
  
  score = cross_val_score(etree_w2v, X, y, cv=5).mean()
  print(score)

  score2 = cross_val_score(mult_nb, X, y, cv=5).mean()
  print(score2)
    
if __name__ == '__main__':
    main()


# let X
# model = gensim.models.Word2Vec(X, size=100)
# w2v = dict(zip(model.index2word, model.syn))
