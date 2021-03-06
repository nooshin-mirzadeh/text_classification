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
import pandas as pd


names = ["SVC5000", "SVC", "RBFSVC", "LinearSVC",
         "RandomForest",
         "GaussianBayes", "NaiveBayes", "BernoulliBayes"]

def getClassifier(clf_name):
    n2c = dict(zip(names, classifiers))
    return n2c[clf_name]

def main():
    name_clf = input('Classifier name')
    dim = input('choose the dimension: 50, 200, 500')
    X = np.load(('./results_full/test'+ dim +'.npy'))
    tid = np.load('./results_full/testID.npy')

    clf = joblib.load((name_clf+dim))
    y = clf.predict(X) 
	
    print(y)
    out = np.column_stack((tid,np.array(y)))
    print(out.shape)
    out_df = pd.DataFrame(out, columns = ['Id', 'Prediction'])
    out_df.to_csv("./results_full/"+name_clf+dim+"CleanTest.csv", index=False)
    #with open(("./results_full/"+name_clf+"Test.csv"), 'w') as f:
    #  np.savetxt(f, out, delimiter=',', fmt="%s")
    print('done')
    

if __name__ == '__main__':
    main()

