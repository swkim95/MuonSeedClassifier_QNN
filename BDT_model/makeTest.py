import sys
import numpy as np
from HLTIO import IO
from HLTIO import preprocess

# Make test dataset
print("Making test dataset ...")

IO.maketest(1.,1.,"Sig")
IO.maketest(0.,1.,"Bkg")
mtxSig = IO.Read("./data/testSig.root",["v1","v2","v3","v4","v5"])
mtxBkg = IO.Read("./data/testBkg.root",["v1","v2","v3","v4","v5"])

# Preprocessing
print("Preprocessing ...")

x, y = preprocess.vconcat(mtxSig,mtxBkg)
x_train, x_test, y_train, y_test = preprocess.split(x, y)
IO.dumpsvm(x_train,y_train,'./data/testTrain.svm')
IO.dumpsvm(x_test,y_test,'./data/testTest.svm')

print("Finished making test dummy dataset.")
