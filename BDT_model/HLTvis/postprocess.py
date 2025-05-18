import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def maskTarget(x, y):
    y = np.reshape(y, (-1,1))
    stacked = np.hstack((y,x))
    df = pd.DataFrame(data=stacked)
    mask = df[0] == 1
    dfSig = df[mask]
    dfBkg = df[~mask]
    dfSig = dfSig.drop([0],axis=1)
    dfBkg = dfBkg.drop([0],axis=1)

    return dfSig.values, dfBkg.values

def calROC(dTrainPredict, dTestPredict, y_train, y_test):
    fpr_Train, tpr_Train, thr_Train = roc_curve(y_train, dTrainPredict)
    AUC_Train = roc_auc_score(y_train, dTrainPredict)

    fpr_Test, tpr_Test, thr_Test = roc_curve(y_test, dTestPredict)
    AUC_Test = roc_auc_score(y_test, dTestPredict)

    return fpr_Train, tpr_Train, thr_Train, AUC_Train, fpr_Test, tpr_Test, thr_Test, AUC_Test

def confMat(y,pred):
    #confMat_     = confusion_matrix(y,pred,normalize='true',labels=[0,1,2,3])
    #confMat_abs_ = confusion_matrix(y,pred,labels=[0,1,2,3])
    confMat_     = confusion_matrix(y,pred,normalize='true',labels=[0,1])
    confMat_abs_ = confusion_matrix(y,pred,labels=[0,1])

    return confMat_, confMat_abs_

def sigmoid( pred_raw ):
    sigmoid_ = 1. / 1./(1.+np.exp(-1*pred_raw))

    return sigmoid_

def softmaxLabel(predict):
    return np.argmax(predict, axis=1)

def binaryLabel(predict):
    return np.around(predict)