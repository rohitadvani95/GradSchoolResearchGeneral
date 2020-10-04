# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:37:03 2020

@author: kirby
"""


# PLEASE NOTE: LINES 49-175 ARE INTELLECTUAL PROPERTY OF ERIK NORDQUIST
import pandas as pd
import numpy as np


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold,StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
from sklearn import utils

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc
import csv

#with open (r'C:\Users\kirby\OneDrive\Documents\rotationproj\data\data\X_test.npy', 'rU') as infile:




#useCats = True

#geometryv0 = pd.read_csv('C:\Users\kirby\OneDrive\Documents\RotationProject\geometryv0.csv', skiprows=0)



#temp = pd.read_csv('C:\Users\kirby\OneDrive\Documents\rotationproj\py\py\temp.csv', skiprows=0)



#X = pd.DataFrame(data, columns = ['SECS', 'WATX', 'MEMBX', 'PORX', 'BURIED', 'BORDER', 'SASA', 'CHARGE'])

#df = pd.DataFrame(data, columns = ['MW'])


#print(df)


X_test = np.load('C:/Users/kirby/OneDrive/Documents/rotationproj/data/data/X_test.npy')
X_train = np.load('C:/Users/kirby/OneDrive/Documents/rotationproj/data/data/X_train.npy')
y_test = np.load('C:/Users/kirby/OneDrive/Documents/rotationproj/data/data/y_test.npy')
y_train = np.load('C:/Users/kirby/OneDrive/Documents/rotationproj/data/data/y_train.npy')

print(X_test.shape)
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)


mslo1df = pd.read_csv('C:/Users/kirby/OneDrive/Documents/RotationProject/bk/model/mutations/mslo1_singleMT.csv')
mslo1df = mslo1df[mslo1df.Mutation.notna()]
#print('\n  read in mslo1 mutations database\n')
cols=['Mutation','MTV1/2[Ca0]']
BK1 = mslo1df[cols]
#print(BK1.head())
  
# add WT V1/2 to mslo dataframe
V = 184

BK1.loc[:,'WTV1/2[Ca0]'] = V
 
#print(BK1)
 
def f(x):
  try:
    return np.float(x)
  except:
    return np.nan 


BK = BK1
BK.loc[:,'MTV1/2[Ca0]'] = BK.loc[:,'MTV1/2[Ca0]'].apply(f)
BK.loc[:,'WTV1/2[Ca0]'] = BK.loc[:,'WTV1/2[Ca0]'].apply(f)
BK.loc[:,'dV']   = BK.loc[:,'MTV1/2[Ca0]'] - BK.loc[:,'WTV1/2[Ca0]']

residueDict = {'L':0, 'I':1, 'V':2, 'F':3, 'M':4,
               'W':5, 'A':6, 'C':7, 'G':8, 'P':9,
               'Y':10,'T':11,'S':12,'H':13,'Q':14,
               'K':15,'N':16,'E':17,'D':18,'R':19}

def wt(x):
  return residueDict[x[0]]
def mt(x):
  return residueDict[x[-1]]
def resN(x):
  return x[1:-1]

# get the residueNumber, WT residueID of WT and MT --> then put them in as numbers
BK.loc[:,'WT'] = BK.loc[:,'Mutation'].apply(wt)
BK.loc[:,'MT'] = BK.loc[:,'Mutation'].apply(mt)
BK.loc[:,'resN'] = BK.loc[:,'Mutation'].apply(resN)


hyb = pd.read_csv('C:/Users/kirby/OneDrive/Documents/RotationProject/bk/model/mutations/solvFE.dat',index_col='id')

def g(x):
  return np.float(hyb.loc[x,'solvFE'])

# convert MT and WT into hydrophobicity (solvFE or dG_transfer)
BK.loc[:,'WThyd']  = BK.loc[:,'WT'].apply(g)
BK.loc[:,'MThyd']  = BK.loc[:,'MT'].apply(g)
BK.loc[:,'dHyd']   = BK.loc[:,'MThyd'] - BK.loc[:,'WThyd']

BK.dropna(inplace=True)

geometryv0 = pd.read_csv('C:\Users\kirby\OneDrive\Documents\RotationProject\geometryv0.csv',sep=',',index_col='RESID')
residueDict = {'LEU':0, 'ILE':1, 'VAL':2, 'PHE':3, 'MET':4,
               'TRP':5, 'ALA':6, 'CYS':7, 'GLY':8, 'PRO':9,
               'TYR':10,'THR':11,'SER':12,'HIS':13,'GLN':14,
               'LYS':15,'ASN':16,'GLU':17,'ASP':18,'ARG':19}


#print(geometryv0)


def geom(x):
  return geometryv0.loc[int(x),['SECS', 'WATX', 'MEMBX', 'PORX', 'BURIED', 'BORDER', 'MODULE']] #leaving out the sasa for the coils not in there
# adds the geometrical data as-is
BK = pd.concat([BK,BK.loc[:,'resN'].apply(geom)],axis=1)


xlabels = ['resN','dHyd','SECS','WATX','MEMBX','PORX','BURIED','BORDER'] # no categories (discrete but not ordered)-- binary is ok
xcats =   ['MODULE']

#Residue number, 

ylabels = ['dV']

#print(type(xlabels))
#print(type(xcats))
#print(type(ylabels))


Xarray = BK.loc[:,xlabels].values # no categories yet
yarray = BK.loc[:,ylabels].values

#print(Xarray)
#print(yarray)
#print("3")



#if not useCats:
  ## K-NN appears to benefit from categorical data
  # ANNs and RF look like they need one-hots
  
  #Xcat = BK.loc[:,xcats].values
  #print("3")
  #enc = OneHotEncoder(categories='auto') # just to silence warning, shouldn't need this now
  #print("3")
  #enc.fit(Xcat)
  #print("3")
  #Xcat = enc.transform(Xcat).toarray()
  #print("3")
  #print(Xcat.shape) # Ncol -> entries, Nrows -> k1 + k2 + k3 + ... (ki = Ncategories in feature i)
  #print(Xarray.shape,Xcat.shape,np.concatenate((Xarray, Xcat),axis=1).shape)
  #Xarray = np.concatenate((Xarray, Xcat), axis=1)
  #print(Xarray)
print(Xarray.shape,yarray.shape)



A = Xarray

B = yarray



dfA = pd.DataFrame(A)
#dfB = pd.DataFrame(B)

with open(r'C:\Users\kirby\OneDrive\Desktop\dfB.csv', 'rU') as infile:
  
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

print(infile)


data = pd.read_csv(r'C:\Users\kirby\OneDrive\Desktop\dfB.csv', encoding='utf-8')
                    
dfB = pd.DataFrame(data, columns = ['BinThreshold'])

#dfB.drop(["BinThreshold"], axis = 1, inplace = True)
#print(dfB)



#print(df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(dfA)
    #print(dfB)

NUM_TRIALS = 10


non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)


for i in range(NUM_TRIALS):

    sss_outer = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=10)
    sss_inner = StratifiedShuffleSplit(n_splits=2, test_size=0.10, random_state=10)

    param_grid = {"max_depth": [3, None], 
                  "min_samples_split": [2, 5]} 
                  #"classifier__min_samples_leaf": [1, 3, 10]}
                  #"bootstrap": [True, False],
                  #"classifier__criterion": ["gini", "entropy"]}
                  #"max_features": [1, 3, 10],
              
                  #"min_samples_split": [1, 3, 10]}
              

    RFC = RandomForestClassifier(random_state = 10)

    estimator = RFC
    clf = GridSearchCV(estimator, param_grid=param_grid, cv=sss_inner)
    #print(estimator.get_params().keys())
    #clf.fit(dfA, dfB)
    #non_nested_scores[i] = clf.best_score_
    nested_score = cross_val_score(clf, dfA, dfB, cv = sss_outer)
    nested_scores[i] = nested_score.mean()
    
#score_difference = non_nested_scores - nested_scores
#print(non_nested_scores.mean())
print(nested_score)
print(nested_scores.mean())
print(nested_scores.std())
#print(confusion_matrix())




#probas_ = GridSearchCV.fit(dfA[train], dfB[train]).predict_proba(dfA[test])
#fpr, tpr, thresholds = roc_curve(dfB[test], probas_[:, 1])
#roc_auc = auc(fpr, tpr)
#y_pred = clf.predict(nested_score)
#print(y_pred)
