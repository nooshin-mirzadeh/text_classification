import numpy as np
import sklearn as sk
import codecs


def build_word_vector_matrix(vector_file, n_words):
  '''Read a GloVe array and return its vectors and labels as arrays'''
  numpy_arrays = []
  labels_array = []
  with codecs.open(vector_file, 'r', 'utf-8') as f:
    for c, r in enumerate(f):
      sr = r.split()
      labels_array.append(sr[0])
      numpy_arrays.append( np.array([float(i) for i in sr[1:]]) )

      if c == n_words:
        return np.array( numpy_arrays ), labels_array

  return np.array( numpy_arrays ), labels_array


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

def loadTestData(fn):
	X, tid = [], []
	with open(fn) as infile:
		for line in infile:
			i, sep, text = line.partition(',')
			tid.append(i)
			X.append(text.split())
		X = np.array(X)
		tid = np.array(tid)
		print(X.shape)
		return X, tid


def transform(w2v, dim, X):
	''' Return the bag of the words for an array of tweets'''
	return np.array([np.mean([w2v[w] for w in words if w in w2v] or [np.zeros(dim)], axis=0 ) for words in X ])



def main():
  #load pos and neg data
  #print('loading pos data...')
  #X1, y1 = loadTrainData('../twitter-datasets/train_pos_full.txt','1')
  #print('loading neg data...')
  #X2,y2 = loadTrainData('../twitter-datasets/train_neg_full.txt','-1')
  #X = np.concatenate((X1, X2), axis=0)
  #y = np.concatenate((y1, y2), axis=0)

  print('loading training data...')
  X = np.load('./results_full/porter_training.npy')
  y = np.load('./results_full/clean_labels_training.npy')
  print(X.shape)
  print(X[0])

  print('loading test data...')
  #Xt,tid = loadTestData('../twitter-datasets/test_data.txt')
	Xt = np.load('./results_full/porter_test.npy')
  tid = np.load('./results_full/clean_testID.npy')

  print('building the w2v...')
  #build bag of the word
  dim = input('choose vector dimension: 50, 200, 500\n')
  v, w = build_word_vector_matrix(('./data/vectors'+dim+'.txt'), 1000000000)
  w = np.array(w)
  w2v = dict(zip(w,v))

  print('Creating the bag of words for training data...')
  bag_of_words = transform(w2v, int(dim), X)
  print('Bag of the words shape: ')
  print(bag_of_words.shape)

  print('Creating the bag of words for test data...')
  bag_of_words_t = transform(w2v, int(dim), Xt)
  print('Bag of the words shape: ')
  print(bag_of_words_t.shape)

  print('Saving the data...')
  np.save('./results_full/labels_training', y)
  np.save(('./results_full/bagOfWord'+dim), bag_of_words)
  np.save('./results_full/testID', tid)
  np.save(('./results_full/test'+dim), bag_of_words_t)

if __name__ == '__main__':
    main()
