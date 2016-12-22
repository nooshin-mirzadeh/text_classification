import numpy as np
import sys
import pickle
import sklearn as sk
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score
import pandas as pd


#clf50 = ["SVC5000", "LinearSVC",
clf50 = ["RandomForest50",
         "GaussianBayes50", "BernoulliBayes50"]
clf200 = ["RandomForest200", "BernoulliBayes200"]
clf500 = ["RandomForest500", "GaussianBayes500", "BernoulliBayes500"]
classifiers = [
    SVC(kernel="linear", C=0.025, max_iter=5000),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    LinearSVC(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB()]

def getClassifier(clf_name):
    n2c = dict(zip(names, classifiers))
    return n2c[clf_name]


def main():
    ##name_clf = input('Classifier name')
    print('Load training data...')
    X50 = np.load('./results_full/bagOfWord.npy')
    X200 = np.load('./results_full/bagOfWord200.npy')
    X500 = np.load('./results_full/bagOfWord500.npy')
    y = np.load('./results_full/clean_labels_training.npy')
    
    scores = []
    for name_clf in clf50:
      clf = joblib.load(name_clf)
      print('cross validating ', name_clf)
      score = cross_val_score(clf, X50, y, cv=5).mean()
      print(name_clf, ' : ', score)
      scores.append((name_clf, score))

    for name_clf in clf200:
      clf = joblib.load(name_clf)
      print('cross validating ', name_clf)
      score = cross_val_score(clf, X200, y, cv=5).mean()
      print(name_clf, ' : ', score)
      scores.append((name_clf, score))

    for name_clf in clf500:
      clf = joblib.load(name_clf)
      print('cross validating ', name_clf)
      score = cross_val_score(clf, X50, y, cv=5).mean()
      print(name_clf, ' : ', score)
      scores.append((name_clf, score))	

    np.save(np.array(scores))
    print('Final results: ')
    print(scores)
    #with open(("./results_full/"+name_clf+"Test.csv"), 'w') as f:
    #  np.savetxt(f, out, delimiter=',', fmt="%s")
    print('done')
    

if __name__ == '__main__':
    main()

