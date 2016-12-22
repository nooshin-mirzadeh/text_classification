import sys
import numpy as np
import pickle
import sklearn as sk
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

names = ["SVC5000", "SVC", "RBFSVC", "LinearSVC",
         "RandomForest", 
         "GaussianBayes", "NaiveBayes", "BernoulliBayes"]


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
    name_clf = input('Classifier name')
    X = np.load('./results_full/bagOfWord.npy')
    y = np.load('./results_full/labels_training.npy')

    clf = getClassifier(name_clf)
    clf.fit(X,y)
    joblib.dump(clf, name_clf)

if __name__ == '__main__':
    main()
