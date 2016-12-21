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

def transform(w2v, dim, X):
	return np.array([np.mean([w2v[w] for w in words if w in w2v] or [np.zeros(dim)], axis=0 ) for words in X ])


def main():
	
	#load pos and neg data
	X1, y1 = loadTrainData('twitter-datasets/train_pos_full.txt','1')
	X2,y2 = loadTrainData('twitter-datasets/train_neg_full.txt','-1')
	X = np.concatenate((X1, X2), axis=0)
  	y = np.concatenate((y1, y2), axis=0)
	
	#build bag of the word
	w2v = build_word_vector_matrix('./GloVe/res_full/vectors50.txt', 1000000000)

	bag_of_words = transform(w2v, 50, X)
	
if __name__ == '__main__':
    main()
