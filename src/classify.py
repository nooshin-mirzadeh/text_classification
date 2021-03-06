import sys
import numpy as np
import pickle
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

names = ["LinearSVC",
         "RandomForest", 
         "GaussianBayes", "BernoulliBayes"]


classifiers = [
    LinearSVC(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    BernoulliNB()]

def getClassifier(clf_name):
    n2c = dict(zip(names, classifiers))
    return n2c[clf_name]


def main():
    name_clf = input('Classifier name\n')
    dim = input('choose the vector dimension: 50, 200, 500\n')
    X = np.load(('./results_full/bagOfWord'+ dim + '.npy'))
    y = np.load('./results_full/clean_labels_training.npy')

    clf = getClassifier(name_clf)
    clf.fit(X,y)
    joblib.dump(clf, (name_clf+dim))

if __name__ == '__main__':
    main()
