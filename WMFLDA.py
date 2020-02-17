import os
import scipy.io
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing


print(os.listdir("./"))


def learnVector2(LabelMat,dim):
    Q=LabelMat
    U,S,V = np.linalg.svd(Q)
    line_S = S.reshape(S.shape[0],1)
    sum_S = line_S.sum(axis=0).reshape(line_S.sum(axis=0).shape[0],1)[0,0]
    U = U[:,0:dim]
    S = np.diag(S[0:dim])
    V = V.T[:,0:dim]
    percentage = S.sum(axis=0).sum(axis=0)/sum_S
    print("this dim:%d" %dim) 
    print("own percent is %.4f"%percentage)
    X = np.dot(U,np.sqrt(S))
    Z = np.dot(V,np.sqrt(S))
    X = preprocessing.minmax_scale(X, axis=1)
    minx = X.min(1).reshape(X.min(1).shape[0],1)
    maxx = X.max(1).reshape(X.max(1).shape[0],1)
    newX = X-np.tile(minx,(1,dim))
    Z = preprocessing.minmax_scale(Z, axis=1)
    return newX


def getOptimalWrWeights(Hs,alpha):
    K = len(Hs)
    sorted = np.sort(Hs,axis = 0)
    index = np.argsort(Hs,axis=0).reshape(np.argsort(Hs,axis=0).shape[0],)
    newHs = Hs[index]
    p = K
    bfind = 1
    gamma = 0
    while p>0 and bfind:
        gamma = (newHs[0:p].sum(axis=0)[0,]+2*alpha)/p
        if (gamma-newHs[p-1][0,])>0:
            bfind=0
        else:
            p=p-1
    print("gamma:%d\n" %gamma)
    newWs=np.zeros((K,1))
    for ii in range(0,p):
        newWs[ii]=(gamma-newHs[ii])/(2*alpha)
    Ws=np.zeros((K,1))
    Ws[index]=newWs
    return Ws


def getOptimalWhWeights(Ks,beta):
    r = Ks.shape[0]
    c = Ks.shape[1]
    Kss = Ks.flatten(order ='F')
    sorted = np.sort(Kss,axis = 0)
    index = np.argsort(Kss,axis=0,kind='stable')
    index_zero = np.nonzero(Kss==0)[0]
    index2 = np.setdiff1d(index,index_zero,assume_unique=True)
    newKss = Kss[index2]
    K=len(newKss)
    p=K
    bfind=1
    miu=0
    while p>0 and bfind:
        miu=(newKss[0:p].sum(axis=0)+2*beta)/p
        if (miu-newKss[p-1])>0:
            bfind=0
        else:
            p=p-1
    print("miu:%d\n" %miu)
    newWt=np.zeros((K,1))
    for ii in range(0,p):
        newWt[ii]=(miu-newKss[ii])/(2*beta)
    Wt = np.zeros((r*c,1))
    Wt[index2] = newWt
    Wt = Wt.reshape(r,c,order='F')
    return Wt


def WMFLDA_demo(nTypes,instanseIdx,Gcell,Rcell,thetaCell):
    max_iter = 200
    alpha = 10**7
    beta = 10**6
    threshold = 0.00001
    r_thetacell = thetaCell.shape[0]
    c_thetacell = thetaCell.shape[1]
    theta_p = [[[] for _ in range(c_thetacell)] for _ in range(r_thetacell)]
    theta_n = [[[] for _ in range(c_thetacell)] for _ in range(r_thetacell)]
    G_enum = [0] * 6
    G_denom = [0] * 6
    Scell = [[] for _ in range(len(Rcell))]
    for ii in range(0,r_thetacell):
        for jj in range(0,c_thetacell):
            theta = thetaCell[ii,jj]
            t = abs(theta)
            theta_p[ii][jj] = (t+theta)/2
            theta_n[ii][jj] = (t-theta)/2
    for iter in range(0,max_iter):
        print("iter:%d\n" %iter)
        # get mus and kmus
        mus = np.zeros((len(Rcell),1))
        for rr in range(0,len(instanseIdx)):
            i = int(instanseIdx[rr]/nTypes)+1
            j = instanseIdx[rr] % nTypes
            if j == 0:
                i = i-1
                j = 6
            Gmatii = Gcell[i-1]
            Gmatjj = Gcell[j-1]
            Rmat = Rcell[rr]
            Smat = np.dot(np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(Gmatii.T,Gmatii)),Gmatii.T),Rmat),Gmatjj),np.linalg.pinv(np.dot(Gmatjj.T,Gmatjj)))
            Smat[np.isnan(Smat)]=0
            Scell[rr]=Smat
            result = Rcell[rr]-np.dot(np.dot(Gmatii,Scell[rr]),Gmatjj.T)
            R = np.square(result).sum(axis=0).sum(axis=0)
            mus[rr,0]=R
        kmus = np.zeros((r_thetacell,c_thetacell))
        for ii in range(r_thetacell):
            for jj in range(c_thetacell):
                if thetaCell[ii,jj].shape[0]:
                    Gmat = Gcell[ii]
                    result2 = np.dot(np.dot(Gmat.T,thetaCell[ii,jj].toarray()),Gmat)
                    R2 = np.trace(result2)
                    kmus[ii,jj] = R2
        # get Ws and Wt
        Ws = getOptimalWrWeights(mus,alpha)
        Wt = getOptimalWhWeights(kmus,beta)
        # update G with relation matrices and constraint matrices
        for rr in range(0,len(instanseIdx)):
            i = int(instanseIdx[rr]/nTypes)+1
            j = instanseIdx[rr] % nTypes
            if j == 0:
                i = i-1
                j = 6
            temp1 = np.dot(np.dot(Rcell[rr],Gcell[j-1]),Scell[rr].T)
            temp1[np.isnan(temp1)]=0
            t = abs(temp1)
            temp1p = (t+temp1)/2
            temp1n = (t-temp1)/2
            temp2 = np.dot(np.dot(np.dot(Scell[rr],Gcell[j-1].T),Gcell[j-1]),Scell[rr].T)
            temp2[np.isnan(temp2)]=0
            t = abs(temp2)
            temp2p = (t+temp2)/2
            temp2n = (t-temp2)/2
            temp3 = np.dot(np.dot(Rcell[rr].T,Gcell[i-1]),Scell[rr])
            temp3[np.isnan(temp3)]=0
            t = abs(temp3)
            t = np.where(t < 0, 0, t)
            temp3p = (t+temp3)/2
            temp3n = (t-temp3)/2
            temp4 = np.dot(np.dot(np.dot(Scell[rr].T,Gcell[i-1].T),Gcell[i-1]),Scell[rr])
            temp4[np.isnan(temp4)]=0
            t = abs(temp4)
            t = np.where(t < 0, 0, t)
            temp4p = (t+temp4)/2
            temp4n = (t-temp4)/2
            G_enum[i-1] = G_enum[i-1]+Ws[rr][0]*temp1p+np.dot(Ws[rr][0]*Gcell[i-1],temp2n)
            G_denom[i-1]= G_denom[i-1]+Ws[rr][0]*temp1n+np.dot(Ws[rr][0]*Gcell[i-1],temp2p)
            G_enum[j-1] = G_enum[j-1]+ Ws[rr][0]*temp3p+np.dot(Ws[rr][0]*Gcell[j-1],temp4n)
            G_denom[j-1]= G_denom[j-1]+Ws[rr][0]*temp3n+np.dot(Ws[rr][0]*Gcell[j-1],temp4p)
        for ii in range(0,r_thetacell):
            for jj in range(0,c_thetacell):
                if thetaCell[ii,jj].shape[0]:
                    G_enum[ii] = G_enum[ii]+np.dot(Wt[ii,jj]*(theta_n[ii][jj].toarray()),Gcell[ii])
                    G_denom[ii] = G_denom[ii]+np.dot(Wt[ii,jj]*(theta_p[ii][jj].toarray()),Gcell[ii])
        for ii in range(0,len(Gcell)):
            G_denom[ii]=G_denom[ii]+np.spacing(1)
            factor = np.sqrt(G_enum[ii]/G_denom[ii])
            Gcell[ii]=Gcell[ii]*factor
            Gcell[ii][np.isnan(Gcell[ii])]=0
            Gcell[ii][np.isinf(Gcell[ii])]=0
        #compare the target approximation (||R15-G1S15G5'||^2) with threshold
        result = Rcell[3]-np.dot(np.dot(Gcell[0],Scell[3]),Gcell[4].T)
        R = np.square(result).sum(axis=0).sum(axis=0)
        print("R:%d\n" %R)
        if R<threshold:
            break
    newF = np.dot(np.dot(Gcell[0],Scell[3]),Gcell[4].T)
    return newF


lncRNAMiA = scipy.io.loadmat( 'lncRNAMiA.mat' )
R12 = lncRNAMiA['lncMI'].astype(np.float64)
nlnc = lncRNAMiA['lncMI'].shape[0]
nmi = lncRNAMiA['lncMI'].shape[1]
lncRNAGene = scipy.io.loadmat( 'lncRNAGene.mat' )
R13 = lncRNAGene['LGasso'].astype(np.float64)
lncRNAGOs = scipy.io.loadmat( 'lncRNAGOs.mat' )
R14 = np.hstack((lncRNAGOs['lncBPs'],lncRNAGOs['lncCCs'],lncRNAGOs['lncMFs'])).astype(np.float64)
LncDOs = scipy.io.loadmat( 'LncDOs.mat' )
lncDisease2 = scipy.io.loadmat( 'lncDisease2.mat' )
LncDO = LncDOs['lncDisease'] + lncDisease2['LncCancer']
R15 = np.where(LncDO > 1, 1, LncDO).astype(np.float64)
MiDOs = scipy.io.loadmat( 'MiDOs.mat' )
R25 = MiDOs['miDOs'].astype(np.float64)
miRNAGene = scipy.io.loadmat( 'miRNAGene.mat' )
R23 = miRNAGene['MGasso'].astype(np.float64)
GeneDisease = scipy.io.loadmat( 'GeneDisease.mat' )
R35 = GeneDisease['GDasso'].astype(np.float64)
npro = GeneDisease['GDasso'].shape[0]
ndi = GeneDisease['GDasso'].shape[1]
HumanGOAs = scipy.io.loadmat( 'HumanGOAs.mat' )
R34 = sp.hstack((HumanGOAs['bpLabels'],HumanGOAs['ccLabels'],HumanGOAs['mfLabels'])).toarray()
nGO = R34.shape[1]
GeneDrug = scipy.io.loadmat( 'GeneDrug.mat' )
R63 = GeneDrug['GDrgasso'].astype(np.float64)
nDrug = R63.shape[0]
thetaCell = scipy.io.loadmat( 'thetaCell.mat' )
thetaCell = thetaCell['thetaCell']
print("load data successfully!\n")
## filter the no instanse disease.
fun_stat = R15.sum(axis=0)
fun_stat = (fun_stat.reshape(fun_stat.shape[0],1)).T
sel_do_idx = np.where(fun_stat>0)[1]
R15 = R15[:,sel_do_idx]
R25 = R25[:,sel_do_idx]
R35 = R35[:,sel_do_idx]
ndi = len(sel_do_idx)
Rcell=[R12,R13,R14,R15,R23,R25,R34,R35,R63]
## Initialize the G with PNMF
k1=220
k2=220
k3=160
k4=20
k5=50
k6=50
k = [k1,k2,k3,k4,k5,k6]
R1 = np.hstack((R12,R13,R14))
R5 = np.hstack((R25.T,R35.T))
G1 = learnVector2(R1,k1)
G2 = learnVector2(R25,k2)
G3 = learnVector2(R35,k3)
G4 = learnVector2(R14.T,k4)
G5 = learnVector2(R5,k5)
G6 = learnVector2(R63,k6)
Gcell=[G1,G2,G3,G4,G5,G6]
## instanse index for the corresponding position in the whole relation matrix R
instanseIdx = [2,3,4,5,9,11,16,17,33]
nTypes=6
newF = WMFLDA_demo(nTypes,instanseIdx,Gcell,Rcell,thetaCell)
