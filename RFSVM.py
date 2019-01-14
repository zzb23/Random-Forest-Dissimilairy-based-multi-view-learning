"""
This is the code for two intermediate integration methods proposed in the work of:
@article{cao2019random,
  title={Random forest dissimilarity based multi-view learning for Radiomics application},
  author={Cao, Hongliu and Bernard, Simon and Sabourin, Robert and Heutte, Laurent},
  journal={Pattern Recognition},
  volume={88},
  pages={185--197},
  year={2019},
  publisher={Elsevier}
}
"""
import numpy as np
import pandas
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from collections import Counter
from math import floor
from joblib import Parallel, delayed


def floored_percentage(val, digits):
    #transform accuracy into percentage form
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}\%\pm '.format(digits, floor(val) / 10 ** digits)
def splitdata(X,Y,ratio,seed):
    '''This function is to split the data into train and test data randomly and preserve the class ratio'''
    n_samples = X.shape[0]
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    #fint the indices for each class
    indices = []
    print()
    for i in classes:
        indice = []
        for j in range(n_samples):
            if y[j] == i:
                indice.append(j)
        #print(len(indice))
        indices.append(indice)
    train_indices = []
    for i in indices:
        k = int(len(i)*ratio)
        train_indices += (random.Random(seed).sample(i,k=k))
    #find the unused indices
    s = np.bincount(train_indices,minlength=n_samples)
    mask = s==0
    test_indices = np.arange(n_samples)[mask]
    return train_indices,test_indices

def rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    # Measure the random forest similarity, returns the similarity matrix
    clf = RandomForestClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=1)
    clf = clf.fit(X[train_indices], Y[train_indices])
    pred = clf.predict(X[test_indices])
    weight = clf.score(X[test_indices], Y[test_indices])
    #print(1 - clf.oob_score_)
    n_samples = X.shape[0]
    dis = np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        dis[i][i] = 1
    res = clf.apply(X)
    for i in range(n_samples):
        for j in range(i+1,n_samples):
            a = np.ravel(res[i])
            b = np.ravel(res[j])
            score = a == b
            d = float(score.sum())/n_trees
            dis[i][j]  =dis[j][i] = d

    X_features1 = np.transpose(dis)
    X_features2 = X_features1[train_indices]
    X_features3 = np.transpose(X_features2)
    return X_features3[train_indices],X_features3[test_indices],weight,pred


def nLsvm_patatune(train_x,train_y,test_x, test_y):
    tuned_parameters = [
        {'kernel': ['precomputed'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    print(clf.score(test_x,test_y))
    return clf.best_params_['C']



def mcode(ite):
    R = 0.5
    ite = ite
    testfile = open("nus2%f_%f.txt" % (R,ite), 'w')

    #To make the results reproducible, the seed is set
    seed = 1000 + ite

    # data reading
    url = 'awa8_1.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X = array[1:, 1:]
    Y = pandas.read_csv('awa8_label.csv', header=None)
    Y = Y.values
    Y = Y[1:, 1:]
    Y = np.ravel(Y)
    # print(Y.shape

    for i in range(5):
        url = 'awa8_' + str(i + 2) + '.csv'
        dataframe = pandas.read_csv(url, header=None)
        array = dataframe.values
        X1 = array[1:, 1:]
        # print(X1.shape)
        X = np.concatenate((X, X1), axis=1)

    Xnew1 = X[:, 0:2688]
    Xnew2 = X[:, 2688:4688]
    Xnew3 = X[:, 4688:4940]
    Xnew4 = X[:, 4940:6940]
    Xnew5 = X[:, 6940:8940]
    Xnew6 = X[:, 8940:]
    X_all = [Xnew1,Xnew2,Xnew3,Xnew4,Xnew5,Xnew6]
    n_views = len(X_all)
    #n_features = X.shape[1]
    #n_samples = X.shape[0]
    #print("datasize")
    #print(X.shape)
    n_trees = 500
    #n_feat = selected_f(n_features)  # features selecleted
    train_indices, test_indices = splitdata(X=X, Y=Y, ratio=R, seed=seed)


    pre = [] #save the prediction results for Late RFDIS
    pred = [] #save the  prediction results  for Late RF
    X_features_train = [] # save the training RFD matrix
    X_features_test = [] #save the testing RFD matrix
    print("Start rest")
    #view1
    for nv in range(n_views):
        X_features_train1, X_features_test1,w1,pred1= rf_dis(n_trees=n_trees,  X=X_all[nv],Y=Y,  train_indices=train_indices,test_indices=test_indices,seed=seed)
        m12 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_train1, Y[train_indices])
        pre1 = m12.predict(X_features_test1)

        X_features_train.append(X_features_train1)
        X_features_test.append(X_features_test1)
        pred.append(pred1)
        pre.append(pre1)
        #e12.append(m12.score(X_features_test1, Y[test_indices]))
        #e11.append(w1)


    # multi view
    X_features_trainm = np.mean(X_features_train, axis=0)
    X_features_testm = np.mean(X_features_test,axis=0)
    #RFDIS
    mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_trainm, Y[train_indices])
    erfdis = (mv.score(X_features_testm, Y[test_indices]))

    #RFSVM
    c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                       test_y=Y[test_indices])

    clf = SVC(C=c, kernel='precomputed')
    clf.fit(X_features_trainm, Y[train_indices])
    erfsvm = (clf.score(X_features_testm, Y[test_indices]))

    #save the results for each ite
    testfile.write("RFSVM&%s" % (floored_percentage((erfsvm),2)) + '\n')
    testfile.write("RFDIS &%s  & " % (floored_percentage((erfdis),2)) + '\n')
    testfile.close()

if __name__ == '__main__':
    Parallel(n_jobs=10)(delayed(mcode)(ite=i) for i in range(10))