import sys
import numpy as np
from HLTIO import IO
from HLTvis import vis
from sklearn.decomposition import PCA

x, y = IO.loadsvm('./data/testTrain.svm')

pca = PCA(n_components=2)
pca.fit(x)
x = pca.transform(x)

vis.scatter2d(x,y,'PCA')
