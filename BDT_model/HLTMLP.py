import sys
import multiprocessing
import numpy as np
import pandas as pd
from HLTIO import IO
from HLTIO import preprocess
from HLTvis import vis
from HLTvis import postprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
gpu_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def doMLP(seed,seedname,runname):
    colname = list(seed[0].columns)
    x_train, x_test, y_train, y_test = preprocess.split(seed[0], seed[1])
    x_train, x_test = preprocess.stdTransform(x_train, x_test)
    y_wgtsTrain, y_wgtsTest, wgts = preprocess.computeClassWgt(y_train, y_test)
    wgts = {i : wgts[i] for i in range(wgts.size)}

    y_trainOneHot = to_categorical(y_train)
    y_testOneHot = to_categorical(y_test)

    print(seedname + r' C0: %d, C1: %d, C2: %d, C3: %d' % ( (seed[1]==0).sum(), (seed[1]==1).sum(), (seed[1]==2).sum(), (seed[1]==3).sum() ) )

    batchSize = 32
    opt = SGD(lr=0.01, decay=0.)

    model = keras.Sequential()
    model.add( keras.layers.Dense(10, input_dim=len(colname), activation='relu') )
    model.add( keras.layers.Dense(10, input_dim=10, activation='relu') )
    model.add( keras.layers.Dense(10, input_dim=10, activation='relu') )
    model.add( keras.layers.Dense(4, input_dim=10, activation='sigmoid') )

    model.compile(optimizer=opt, loss='categorical_crossentropy')

    history = model.fit(x_train, y_trainOneHot, batch_size=batchSize, epochs=50, validation_split=0.5, class_weight=wgts)

    predict_train = model.predict(x_train, batch_size=batchSize)
    predict_test = model.predict(x_test, batch_size=batchSize)

    labelTrain = postprocess.softmaxLabel(predict_train)
    labelTest = postprocess.softmaxLabel(predict_test)

    for cat in range(4):
        if ( np.asarray(y_train==cat,dtype=np.int).sum() < 2 ) or ( np.asarray(y_test==cat,dtype=np.int).sum() < 2 ): continue

        fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test = postprocess.calROC(predict_train[:,cat], predict_test[:,cat], np.asarray(y_train==cat,dtype=np.int), np.asarray(y_test==cat,dtype=np.int))
        vis.drawROC(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, runname+'_'+seedname+r'_cat%d.png' % cat)

    confMat = postprocess.confMat(labelTest,y_test)
    vis.drawConfMat(confMat,runname+'_'+seedname+'_testConfMat')

    confMatTrain = postprocess.confMat(labelTrain,y_train)
    vis.drawConfMat(confMatTrain,runname+'_'+seedname+'_trainConfMat')

    return

def run(seedname):
    seed = IO.readMinSeeds('/home/common/DY_seedNtuple_v20200510/ntuple_*.root','seedNtupler/'+seedname,0.,99999.,True)
    runname = 'PU180to200Barrel'
    # seed = IO.readMinSeeds('data/ntuple_1-3.root','seedNtupler/'+seedname,0.,99999.,True)
    doMLP(seed,seedname,runname)

seedlist = ['NThltIter2FromL1']

run('NThltIter2FromL1')

print('Finished')
