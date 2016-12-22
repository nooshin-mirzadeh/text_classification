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
  lemma = WordNetLemmatizer()
  lemma_words = [lemma.lemmatize(word) for word in words ]
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

  meaningful_X = []
  for tweet in X:
  	meaningful_tweet = [w for w in tweet if not w in stops]
  	meaningful_X.append(meaningful_tweet)

  #print(meaningful_X)
  meaningful_X = np.array(meaningful_X)

  porter_X, lancaster_X = [],[]
  porter_stemmer = PorterStemmer()
  lancaster_stemmer = LancasterStemmer()
  for i, tweet in enumerate(meaningful_X):  
  	porter = stemize(tweet, porter_stemmer)
  	lancaster = stemize(tweet, lancaster_stemmer)
  	porter_X.append(porter)
  	lancaster_X.append(lancaster)
  	if i%1000 == 0:
  		print(i)
  
  #print(porter_X)
  #print(lancaster_X)
  
  print('Saving the data...')
  np.save('./results_full/clean_labels_training', y)
  np.save('./results_full/porter_training', np.array(porter_X))
  np.save('./results_full/lancaster_training', np.array(lancaster_X))

  with open('./data/clean_porter_tweets.txt', 'w') as f:
    for w in porter_X:
    	for item in w:
        	f.write((item + ' '))
    	f.write('\n')

  with open('./data/clean_lancaster_tweets.txt', 'w') as f:
    for w in lancaster_X:
    	for item in w:
                f.write((item + ' '))
    	f.write('\n')


if __name__ == '__main__':
    main()

