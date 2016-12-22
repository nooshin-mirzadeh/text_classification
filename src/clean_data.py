import numpy as np
import nltk
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

def loadTrainData(fn, emotion):
  X, y = [], []
  with open(fn) as infile:
    for line in infile:
      text = line.split()
      X.append(text)
      y.append(emotion)
  X = np.array(X)
  y = np.array(y)
  print(X.shape)
  return X, y


def stemize(words, stemmer):
  lemma_words = [WordNetLemmatizer(word) for word in words ]
  return [stemmer.stem(word) for word in lemma_words]

def main():

  #load pos and neg data
  print('loading pos data...')
  X1, y1 = loadTrainData('../twitter-datasets/train_pos_full.txt','1')
  print('loading neg data...')
  X2,y2 = loadTrainData('../twitter-datasets/train_neg_full.txt','-1')
  X = np.concatenate((X1, X2), axis=0)
  y = np.concatenate((y1, y2), axis=0)
  
  stops = set(stopwords.words("english"))
  stops.update(['<user>','<url>', '.'])
  print(stops)

  meaningful_X = [w for w in X if not w in stops]

  print(meaningful_X)

  porter_stemmer = PorterStemmer()
  lancaster_stemmer = LancasterStemmer()
  porter_X = stemize(meaningful_X, porter_stemmer)
  lancaster_X = stemize(meaningful_X, lancaster_stemmer)
  
  print(porter_X)
  print(lancaster_X)
  
  print('Saving the data...')
  np.save('./results_full/labels_training', y)
  np.save('./reulsts_full/porter_training', porter_X)
  np.save('./reulsts_full/lancaster_training', lancaster_X)

  with open('./clean_porter_tweets.txt') as f:
    for w in porter_X
      f.write((w + '\n'))

  with open('./clean_lancaster_tweets.txt') as f:
    for w in lancaster_X
      f.write((w + '\n'))

if __name__ == '__main__':
    main()

