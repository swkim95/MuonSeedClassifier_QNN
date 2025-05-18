import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from HLTvis import postprocess

def scatter2d(x, y, plotname, dirname="plot"):
    sig, bkg = postprocess.maskTarget(x, y)
    plt.figure(figsize=(6,4))
    plt.scatter(bkg[:,0],bkg[:,1],c='r',label='bkg',alpha=0.1)
    plt.scatter(sig[:,0],sig[:,1],c='b',label='sig',alpha=0.5)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def scatter2dSB(sig, bkg, plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    plt.scatter(bkg[:,0],bkg[:,1],c='r',label='bkg',alpha=0.1)
    plt.scatter(sig[:,0],sig[:,1],c='b',label='sig',alpha=0.5)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def hist2d(idx,data, plotname, dirname="plot"):
    if idx==0: cm=plt.cm.Reds
    if idx==1: cm=plt.cm.Oranges
    if idx==2: cm=plt.cm.Greens
    if idx==3: cm=plt.cm.Blues
    plt.figure(figsize=(6,4))
    plt.hist2d(data[:,0], data[:,1], bins=100, cmap=cm)
    plt.colorbar()
    plt.draw()
    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def hist2dOverlay(dataSig0,dataSig1,dataBkg0,dataBkg1,plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    cmapBlue = plt.cm.Blues(np.arange(plt.cm.Blues.N))
    cmapBlue[:,-1] = np.linspace(0, 0.8, plt.cm.Blues.N)
    cmapBlue = ListedColormap(cmapBlue)
    cmapRed = plt.cm.Reds(np.arange(plt.cm.Reds.N))
    cmapRed[:,-1] = np.linspace(0, 0.8, plt.cm.Reds.N)
    cmapRed = ListedColormap(cmapRed)
    cmapOrange = plt.cm.Oranges(np.arange(plt.cm.Oranges.N))
    cmapOrange[:,-1] = np.linspace(0, 0.8, plt.cm.Oranges.N)
    cmapOrange = ListedColormap(cmapOrange)
    cmapGreen = plt.cm.Greens(np.arange(plt.cm.Greens.N))
    cmapGreen[:,-1] = np.linspace(0, 0.8, plt.cm.Greens.N)
    cmapGreen = ListedColormap(cmapGreen)
    plt.hist2d(dataBkg0[:,0], dataBkg0[:,1], bins=100, cmap=cmapRed, normed=True)
    plt.hist2d(dataBkg1[:,0], dataBkg1[:,1], bins=100, cmap=cmapOrange, normed=True)
    # plt.colorbar()
    plt.hist2d(dataSig0[:,0], dataSig0[:,1], bins=100, cmap=cmapGreen, normed=True)
    plt.hist2d(dataSig1[:,0], dataSig1[:,1], bins=100, cmap=cmapBlue, normed=True)
    # plt.colorbar()
    plt.draw()
    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawROC(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    plt.plot(fpr_Train, 1-tpr_Train, color='r', label='Train ROC (AUC = %.4f)' % AUC_Train)
    plt.plot(fpr_Test, 1-tpr_Test, color='b', label='Test ROC (AUC = %.4f)' % AUC_Test)
    plt.xlabel('False Positive Rate')
    plt.ylabel('1 - True Positive Rate')
    plt.yscale('log')
    plt.ylim(ymin=1e-3, ymax=1.0)
    plt.title(plotname, fontsize=8)
    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawROC2(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    plt.plot(fpr_Train, tpr_Train, color='r', label='Train ROC (AUC = %.4f)' % AUC_Train)
    plt.plot(fpr_Test,  tpr_Test,  color='b', label='Test ROC (AUC = %.4f)' % AUC_Test)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.yscale('log')
    # plt.ylim(ymin=1e-3, ymax=1.0)
    plt.title(plotname, fontsize=8)
    plt.legend(loc='lower right')
    plt.grid()

    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawThr(thr_Train, tpr_Train, thr_Test, tpr_Test, plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    plt.plot(thr_Train, 1 - tpr_Train, color='r', label='Train thr')
    plt.plot(thr_Test,  1 - tpr_Test,  color='b', label='Test thr')
    plt.xlabel('Threshold')
    plt.ylabel('1 - True Positive Rate')
    plt.yscale('log')
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=1e-3, ymax=1.0)
    plt.title(plotname, fontsize=8)
    plt.legend(loc='lower left')
    plt.grid()

    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawThr2(thr_Train, tpr_Train, thr_Test, tpr_Test, plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    plt.plot(thr_Train, tpr_Train, color='r', label='Train thr')
    plt.plot(thr_Test,  tpr_Test,  color='b', label='Test thr')
    plt.xlabel('Threshold')
    plt.ylabel('True Positive Rate')
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.title(plotname, fontsize=8)
    plt.legend(loc='lower left')
    plt.grid()

    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawScore(dSigPredict, dBkgPredict, plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    plt.hist(dSigPredict, 100, density=True, alpha=0.5, label='Sig', range=(0,1), color='b')
    plt.hist(dBkgPredict, 100, density=True, alpha=0.5, label='Bkg', range=(0,1), color='r')
    plt.grid()
    plt.yscale('log')
    plt.xlim([0,1])
    # plt.ylim([0,100])
    plt.title(plotname, fontsize=8)
    plt.xlabel('Output')
    plt.ylabel('a.u.')
    plt.legend(loc='upper right')

    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawScoreOverlay(dSigPredictTrain, dBkgPredictTrain, dSigPredictTest, dBkgPredictTest, plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    plt.rc('font', size=12)
    plt.hist(dSigPredictTrain, 100, density=True, histtype=u'step', label='Train:Sig', range=(0,1), color='navy')
    plt.hist(dBkgPredictTrain, 100, density=True, histtype=u'step', label='Train:Bkg', range=(0,1), color='maroon')
    plt.hist(dSigPredictTest, 100, density=True, alpha=0.4, label='Test:Sig', range=(0,1), color='blue')
    plt.hist(dBkgPredictTest, 100, density=True, alpha=0.4, label='Test:Bkg', range=(0,1), color='red')
    plt.grid()
    plt.yscale('log')
    plt.xlim([0,1])
    # plt.ylim([0,100])
    # plt.title(plotname, fontsize=8)
    plt.xlabel('Score')
    plt.ylabel('a.u.')
    plt.legend(loc=(0.65,0.65))

    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawScoreRaw(dSigPredict, dBkgPredict, plotname, dirname="plot"):
    plt.figure(figsize=(6,4))
    plt.hist(dSigPredict, 100, density=True, alpha=0.5, label='Sig', color='b')
    plt.hist(dBkgPredict, 100, density=True, alpha=0.5, label='Bkg', color='r')
    plt.grid()
    plt.yscale('log')
    # plt.xlim([0,1])
    # plt.ylim([0,100])
    plt.title(plotname, fontsize=8)
    plt.xlabel('Output')
    plt.ylabel('a.u.')
    plt.legend(loc='upper right')

    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawConfMat(confMat, plotname, dirname="plot", doNorm = True):
    # plt.figure(figsize=(6,4))
    plt.rc('font', size=12)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=11)
    fig, ax = plt.subplots()
    #names = ['Failed-to-build','Combinatorial','Sim-matched','Muon-matched']
    names = ['Backgrounds','Muon-matched']

    if doNorm:
        mat = ax.imshow(confMat,cmap='viridis', vmin=0., vmax=1.)
    else:
        mat = ax.imshow(confMat,cmap='viridis')
    plt.title(plotname, fontsize=8)
    plt.xlabel('prediction')
    plt.ylabel('true')
    fig.colorbar(mat, ax=ax)

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    for i in range(len(names)):
        for j in range(len(names)):
            if doNorm:
                if confMat[i,j] > 0.8:
                    text = ax.text(j, i, r'{:.3f}'.format(confMat[i,j]), ha='center', va='center', color='navy')
                else:
                    text = ax.text(j, i, r'{:.3f}'.format(confMat[i,j]), ha='center', va='center', color='w')

            else:
                text = ax.text(j, i, r'{:.0f}'.format(confMat[i,j]), ha='center', va='center', color='w')

    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawImportance(gain, cover, colname_full, plotname, dirname="plot"):
    colname = [ col for col in colname_full if col in gain.keys() ]
    valGain     = np.asarray( [ gain[x]  for x in colname ] )
    sortedCover = np.asarray( [ cover[x] for x in colname ] )

    plt.figure(figsize=(6,4))
    barwidth = 0.4
    b1 = plt.barh(np.arange(len(gain)) -barwidth/2., 100.*valGain/np.sum(valGain),         barwidth, color='r', label='gain')
    b2 = plt.barh(np.arange(len(cover))+barwidth/2., 100.*sortedCover/np.sum(sortedCover), barwidth, color='b', label='cover')
    plt.yticks(range(len(gain)), colname, fontsize=5)
    plt.legend( (b1[0],b2[0]), ('gain','cover'), fontsize=5 )

    plt.savefig('./'+dirname+'/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.savefig('./'+dirname+'/'+plotname+'.pdf',dpi=300, bbox_inches='tight')
    plt.close()

    return
