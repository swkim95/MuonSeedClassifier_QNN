import sys
import numpy as np
import pandas as pd
from HLTIO import IO
from HLTIO import preprocess
from HLTvis import vis
from sklearn.manifold import TSNE
from pathlib import Path

def doTSNE(seed,seedname,filename):
    checkfile = Path('data/t-sne_'+filename+'_'+seedname+'.csv')
    try:
        checkfile.resolve()
    except FileNotFoundError:
        seed = pd.DataFrame(seed)
        seed, y = preprocess.dfSigBkg(seed)
        seed.drop(seed.columns[[1,2,3,23,24,26,27,28,29]],axis=1,inplace=True)

        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
        tsne_result = tsne.fit_transform(seed)
        seed['tsne-x'] = tsne_result[:,0]
        seed['tsne-y'] = tsne_result[:,1]

        seed = pd.concat([seed,y], axis=1,ignore_index=True)
        seed.to_csv(checkfile,index=None,header=True)
    else:
        seed = pd.read_csv(checkfile)
#        seed.drop(seed.columns['y'],axis=1,inplace=True)
        y = seed.iloc[:,-1]

    # vis.scatter2dSB(seed[~fake][['tsne-x','tsne-y']].values, seed[fake][['tsne-x','tsne-y']].values, 't-sne_'+seedname)
    fake0 = ( seed['y']==0. )
    fake1 = ( seed['y']==1. )
    sig0 = ( seed['y']==2. )
    sig1 = ( seed['y']==3. )
    vis.hist2d(2,seed[sig0][['tsne-x','tsne-y']].values,'t-sneSig0_'+filename+'_'+seedname)
    vis.hist2d(3,seed[sig1][['tsne-x','tsne-y']].values,'t-sneSig1_'+filename+'_'+seedname)
    vis.hist2d(0,seed[fake0][['tsne-x','tsne-y']].values,'t-sneBkg0_'+filename+'_'+seedname)
    vis.hist2d(1,seed[fake1][['tsne-x','tsne-y']].values,'t-sneBkg1_'+filename+'_'+seedname)
    vis.hist2dOverlay(seed[sig0][['tsne-x','tsne-y']].values,seed[sig1][['tsne-x','tsne-y']].values,seed[fake0][['tsne-x','tsne-y']].values,seed[fake1][['tsne-x','tsne-y']].values,'t-sneOverlay_'+filename+'_'+seedname)

    return

# seeds = IO.readSeed("./data/ntuple_PU50.root")
filename = 'Mu_FlatPt2to100_PU200'
seeds = IO.readSeedNp("./data/"+filename+".root")

doTSNE(seeds[0],"iterL3OISeedsFromL2Muons",filename)
doTSNE(seeds[1],"iter0IterL3MuonPixelSeedsFromPixelTracks",filename)
doTSNE(seeds[2],"iter2IterL3MuonPixelSeeds",filename)
doTSNE(seeds[3],"iter3IterL3MuonPixelSeeds",filename)
doTSNE(seeds[4],"iter0IterL3FromL1MuonPixelSeedsFromPixelTracks",filename)
doTSNE(seeds[5],"iter2IterL3FromL1MuonPixelSeeds",filename)
doTSNE(seeds[6],"iter3IterL3FromL1MuonPixelSeeds",filename)




# x, y = IO.loadsvm('./data/testTrain.svm')
#
# pca = PCA(n_components=2)
# pca.fit(x)
# x = pca.transform(x)
#
# vis.scatter2d(x,y,'PCA')
