#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:06:15 2020

@author: jcosme
"""
import requests
import zipfile
import pandas as pd
import pickle
import numpy as np
import sqlalchemy
import tensorflow as tf
from  tensorflow import keras as kr
from sklearn.linear_model import LogisticRegression as skLogReg
from sklearn.preprocessing import LabelEncoder as skLabelEncoder
from sklearn.metrics import confusion_matrix
from umap import UMAP
from sklearn.cluster import KMeans
from itertools import compress
import itertools
import gc
import sys
projectDBName = 'postgresql+psycopg2://postgres:postgres@localhost/argus'
rawDataUrl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
colNames = ['id', 
'diagnosis',
'radius_mean',
'texture_mean',
'perimeter_mean',
'area_mean',
'smoothness_mean', 
'compactness_mean', 
'concavity_mean',
'concave_points_mean',
'symmetry_mean', 
'fractal_dimension_mean', 
'radius_se',
'texture_se', 
'perimeter_se', 
'area_se',
'smoothness_se', 
'compactness_se',
'concavity_se',
'concave_points_se',
'symmetry_se',
'fractal_dimension_se',
'radius_worst', 
'texture_worst',
'perimeter_worst', 
'area_worst', 
'smoothness_worst',
'compactness_worst',
'concavity_worst', 
'concave_points_worst',
'symmetry_worst', 
'fractal_dimension_worst', 
]

def getRawInSQL():
    engine = sqlalchemy.create_engine(projectDBName, echo=True)
    conn = engine.connect()
    rawDataDf = pd.read_csv(rawDataUrl, header=None, names=colNames, index_col=0)
    rawDataDf.to_sql(name='rawdata', con=conn, if_exists='replace')
    return None

def labelTrainDevTest(x):
    if x < .70:
        return 'train'
    elif x >= .85:
        return 'test'
    else:
        return 'dev'

splitVec = np.vectorize(labelTrainDevTest)

def splitDataRando(x):
    mDf = x[ x['diagnosis'] == 'M' ].copy()
    bDf = x[ x['diagnosis'] == 'B' ].copy()
    mRando = np.random.uniform(size=mDf.shape[0])
    bRando = np.random.uniform(size=bDf.shape[0])
    mDf['split'] = splitVec(mRando)
    bDf['split'] = splitVec(bRando)
    x = pd.concat([mDf, bDf])
    trainDf = x[ x['split'] == 'train' ].copy()
    devDf = x[ x['split'] == 'dev' ].copy()
    testDf = x[ x['split'] == 'test' ].copy()
    del x
    del trainDf['split'], devDf['split'], testDf['split']
    return (trainDf, devDf, testDf)



def labelTrainDevTest2(x_in):
    split = x_in[1]
    x = x_in[0]
    split = np.array(split)
    assert len(split) == 3, "The length of the split list must be 3"
    assert split.sum() == 1.0, "The sum of the split list must equal 1"
    train = split[0]
    test = split[0] + split[1]
    if x < train:
        return 'train'
    elif x >= test:
        return 'test'
    else:
        return 'dev'

splitVec2 = np.vectorize(labelTrainDevTest2)


def loadRawFromSQL():
    engine = sqlalchemy.create_engine(projectDBName, echo=False)
    conn = engine.connect()
    sqlString = "SELECT * FROM rawdata"
    out = pd.read_sql(sql=sqlString, con=conn, index_col='id')
    return out

def initialSplitSQL():
    engine = sqlalchemy.create_engine(projectDBName, echo=True)
    conn = engine.connect()
    rawBCancerDF = loadRawFromSQL()
    initialTrain, initialDev, initialTest = splitDataRando(rawBCancerDF)
    initialTrain.to_sql(name='initialtrain', con=conn, if_exists='replace')
    initialDev.to_sql(name='initialdev', con=conn, if_exists='replace')
    initialTest.to_sql(name='initialtest', con=conn, if_exists='replace')
    return None

def loadInitialDataSplitsSQL():
    engine = sqlalchemy.create_engine(projectDBName, echo=False)
    conn = engine.connect()
    sqlString = "SELECT * FROM initialtrain"
    initialTrain = pd.read_sql(sql=sqlString, con=conn, index_col='id')
    initialTrain = initialTrain.reset_index()
    del initialTrain['id']
    sqlString = "SELECT * FROM initialdev"
    initialDev = pd.read_sql(sql=sqlString, con=conn, index_col='id')
    initialDev = initialDev.reset_index()
    del initialDev['id']
    sqlString = "SELECT * FROM initialtest"
    initialTest = pd.read_sql(sql=sqlString, con=conn, index_col='id')
    initialTest = initialTest.reset_index()
    del initialTest['id']
    return (initialTrain, initialDev, initialTest)

def trainDimRdcNN(x):
    x_train = x.copy()
    del x_train['diagnosis']
    y_train = x_train.copy()[['x','y']]
    x_train.drop(['x','y'], axis=1, inplace=True)
    x_train = x_train.values
    y_train = y_train.values
    x_train.shape
    model = kr.Sequential([
    kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(240, activation=tf.nn.swish),
        kr.layers.Dense(2, activation=tf.nn.swish),
    ])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001, amsgrad=True),loss='Huber',)
    model.fit( x_train, y_train, epochs=1000)
    return model










def nnDimRdcPred(aModel, data_x):
    # x_dev = theData.copy()
    # del x_dev['diagnosis']
    # del x_dev['diagnosis_encoded']
    # x_dev.drop(['x','y'], axis=1, inplace=True)
    data_x = data_x.values
    preds = aModel.predict(data_x)
    return preds


def trainSubModels(NNModel, KMModel, trainData):
    tempDf = trainData.copy()
    NNPreds = nnDimRdcPred(NNModel, tempDf.drop(['diagnosis'], axis=1))
    tempDf['diagnosis_encoded'] = skLabelEncoder().fit(tempDf['diagnosis']).transform(tempDf['diagnosis'])
    KMPreds = KMModel.predict(NNPreds)
    tempDf['clustMem'] = KMPreds
    nClusts = KMModel.get_params()['n_clusters']
    negs = tempDf[ tempDf['diagnosis_encoded'] == 0 ]
    poss = tempDf[ tempDf['diagnosis_encoded'] == 1 ]
    negClusts = poss.groupby(['clustMem']).count().reset_index()['clustMem'].unique()
    posClusts = negs.groupby(['clustMem']).count().reset_index()['clustMem'].unique()
    negClust = list(set(posClusts).difference(set(negClusts)))
    posClust = list(set(negClusts).difference(set(posClusts)))
    clustModelNames = list(set(posClusts) & set(negClusts))
    clustModels = {}
    for curModName in clustModelNames:
        clustModels[curModName] = skLogReg(max_iter=3000)
        curData = tempDf[ tempDf['clustMem'] == curModName ].drop(['clustMem'], axis=1)
        clustModels[curModName].fit(curData.drop(['diagnosis','diagnosis_encoded'], axis=1), curData['diagnosis_encoded'])
    return (clustModels, negClust, posClust)

def predFromModels(NNModel, KMModel, subModels, theData):
    out = theData.copy()
    out['diagnosis_encoded'] = skLabelEncoder().fit(out['diagnosis']).transform(out['diagnosis'])
    negClust = subModels[1][0]
    posClust = subModels[2][0]
    clustModels = subModels[0]
    otherClusts = list(clustModels.keys())
    nnPreds = nnDimRdcPred(NNModel, theData.drop(['diagnosis'], axis=1))
    out['clustMem'] = KMModel.predict(nnPreds)
    out.loc[out['clustMem'] == negClust, 'diagnosis_pred'] = 0
    out.loc[out['clustMem'] == posClust, 'diagnosis_pred'] = 1
    for key in otherClusts:
        tempDf = theData[out['clustMem'] == key].drop(['diagnosis'], axis=1)
        tempPred = clustModels[key].predict(tempDf)
        out.loc[out['clustMem'] == key, 'diagnosis_pred'] = tempPred
    return out['diagnosis_encoded'].values

####################################################################################################################################


def encodeData(trainDf, devDf, testDf, nameOfColToEncode='diagnosis'):
    print('******************************************************')
    print('\nencoding data')
    trainDfout = trainDf.copy()
    devDfout = devDf.copy()
    testDfout = testDf.copy()
    rawLabels = pd.concat([trainDfout[nameOfColToEncode], devDfout[nameOfColToEncode], testDfout[nameOfColToEncode]], ignore_index=True).values.tolist()
    rawLabels = pd.unique(rawLabels)
    print('these are the labels we are encoding:\n{}'.format(rawLabels))
    labelEncodings = skLabelEncoder()
    labelEncodings.fit(rawLabels) # create label encodings
    trainDfout[nameOfColToEncode] = labelEncodings.transform(trainDfout[nameOfColToEncode]) # replace the original column with encodings
    devDfout[nameOfColToEncode] = labelEncodings.transform(devDfout[nameOfColToEncode])
    testDfout[nameOfColToEncode] = labelEncodings.transform(testDfout[nameOfColToEncode])
    print('this is what they were encoded to:\n{}'.format(trainDfout[nameOfColToEncode].unique()))
    return (trainDfout, devDfout, testDfout, labelEncodings)

def getUMAPResults(xTrain, params={'metric':'manhattan','min_dist':0.0}):
    print(".............")
    print('going to do a UMAP on this data:')
    print('{}\nthe shape of the data is: {}'.format(xTrain.columns, xTrain.shape))
    xTrainCopy = xTrain.copy()
    print('we are using these parameters:\n{}'.format(params))
    umap  = UMAP(metric=params['metric'], min_dist=params['min_dist'])
    umap = umap.fit_transform(xTrainCopy)
    umap[:,0] = umap[:,0] + umap[:,0].max()
    umap[:,1] = umap[:,1] + umap[:,1].max()
    umapResults = pd.DataFrame()
    umapResults['umapX'] = umap[:,0]
    umapResults['umapY'] = umap[:,1]
    del umap
    gc.collect()
    print('this is the head of the UMAP Results:\n{}'.format(umapResults.head()))
    return umapResults

testingParams = [{'metric':'euclidean','min_dist':0.0}, 
                 {'metric':'manhattan','min_dist':0.0}, 
                 {'metric':'chebyshev','min_dist':0.0}, 
                 {'metric':'minkowski','min_dist':0.0},]
    
def calcManyUMAPResults(trainData, targetColName='diagnosis', nTrials=60, testingParams=testingParams):
    print('******************************************************')
    print('\ncalculating many UMAP Results...')
    yTrain = trainData[targetColName].copy()
    print('this is the target set (will not be used):')
    print('{}\nthe shape of the data is: {}'.format(yTrain.name, yTrain.shape))
    xTrain = trainData.drop([targetColName], axis=1).copy()
    print('this is the train set (will be used):')
    print('{}\nthe shape of the data is: {}'.format(xTrain.columns, xTrain.shape))
    print('we will be doing {} trial on each of {} parameter sets ({} total)'.format(nTrials, len(testingParams), len(testingParams)*nTrials))
    allResults = {}
    for paramSet_i in np.arange(0,len(testingParams)):
        allResults[paramSet_i] = []
    for paramSet_i in np.arange(0,len(testingParams)):
        for trial_i in np.arange(0,nTrials):
            print("---------------------------")
            print('currently doing trials on param set {}'.format(testingParams[paramSet_i]))
            print('conducting trial {} of {}'.format(trial_i+1, nTrials))
            tempUMAPResults = getUMAPResults(xTrain, params=testingParams[paramSet_i])
            allResults[paramSet_i].append(tempUMAPResults)
            del tempUMAPResults
            gc.collect()
    return (allResults, testingParams, yTrain)


def getTopClusts(xTrainUMAPResults, trainedKMeansModel, yTrain):
    predictedKMeansClusters = trainedKMeansModel.predict(xTrainUMAPResults)
    nClusters = trainedKMeansModel.get_params()['n_clusters']
    clustID = np.arange(0, nClusters)
    totalNeg = []
    totalPos = []
    for clusterN in clustID:
        clusterFilterBool = predictedKMeansClusters == clusterN
        clusterTargetVals = yTrain[clusterFilterBool]
        negBool = clusterTargetVals == 0
        negCount = len(clusterTargetVals[negBool])
        posBool = clusterTargetVals == 1
        posCount = len(clusterTargetVals[posBool])
        totalNeg.append(negCount)
        totalPos.append(posCount)
    totals = list([ x + y for x,y in zip(totalPos, totalNeg)])
    negProp = list([ x / y for x,y in zip(totalNeg, totals)])
    posProp = list([ x / y for x,y in zip(totalPos, totals)])
    maxNeg = max(negProp)
    maxPos = max(posProp)
    negSelectBool = np.array(negProp) == maxNeg
    posSelectBool = np.array(posProp) == maxPos
    negClust = list(compress(clustID, negSelectBool))[0]
    posClust = list(compress(clustID, posSelectBool))[0]
    return (negClust, posClust)
    
    

def getManyUMAPResultsCMs(allResults, testingParams, yTrain):
    n_clusters=4
    print('******************************************************')
    print("fitting KMeans to all the UMAP results, and evaluating each")
    allCMs = {}
    for key in allResults.keys():
        allCMs[key] = []
    for key in allResults.keys():
        for value in allResults[key]:
            print("---------------------------")
            print('currently working on this parameter set:\n{}'.format(key))
            print('currently working on this UMAP result:\n{}'.format(value.head()))
            print('fitting a KMeans...')
            kmeansModel = KMeans(n_clusters=n_clusters)
            kmeansModel.fit(value)
            predictedKMeansClusters = kmeansModel.predict(value)    
            print('here are some predicted KMeans cluster memberships:\n{}'.format(predictedKMeansClusters[:5]))
            negClust, posClust = getTopClusts(value, kmeansModel, yTrain)
            print('the cluster number with the highest proportion of NEGATIVE targets is: {}'.format(negClust))
            print('the cluster number with the highest proportion of POSITIVE targets is: {}'.format(posClust))
            print('we will only be evaluating these two cluster.')
            print('cluster {} will always predict NEGATIVE'.format(negClust))
            print('cluster {} will always predict POSITIVE'.format(posClust))
            negBool = predictedKMeansClusters == negClust
            posBool = predictedKMeansClusters == posClust          
            negTar = yTrain[negBool].values
            posTar = yTrain[posBool].values
            allTar = np.concatenate((negTar, posTar), axis=None)         
            negPred = np.zeros(negTar.shape)
            posPred = np.ones(posTar.shape)
            allPreds = np.concatenate((negPred, posPred), axis=None) 
            print("the shape of the targets is: {}".format(allTar.shape))
            print("the shape of the predictions is: {}".format(allPreds.shape))
            tempCM = confusion_matrix(allTar, allPreds)
            print("here is the confusion matrix:\n{}".format(tempCM))
            allCMs[key].append(tempCM)
    return (allCMs, testingParams)

def evaluateManyCMs(allCMs, testingParams):
    print('******************************************************')
    print("going to average out the confusion matrices for each param set")
    avgCMs = {}
    for key in allCMs.keys():
        print('currently working on this parameter set:\n{}'.format(testingParams[key]))
        curCMs = np.array(allCMs[key])
        totalRows = curCMs.shape[0]
        curCMs = curCMs.sum(axis=0) / totalRows
        avgCMs[key] = curCMs
        print('the average of the confusion matrices is:\n{}'.format(curCMs))
    return (avgCMs, testingParams)

criteria = 'min'
metric = 'fn'

def returnUMAPOptParams(avgCMs, testingParams, criteria=criteria, metric=metric):
    assert metric in ('tn','fp','fn','tp'), "metric must be one of: 'tn', 'fp' ,'fn', or 'tp'"
    assert criteria in ('min','max'), "criteria must be one of: 'min', or 'max'"
    print('******************************************************')
    print('selecting the optimal parameter set')
    print('we will select the set that corresponse to the confusion matric with the {} {}'.format(criteria, metric))
    if metric == 'tn':
        cmPosition = 0
    elif metric == 'fp':
        cmPosition = 1
    elif metric == 'fn':
        cmPosition = 2
    else:
        cmPosition = 3
    CMsList = list(avgCMs.values())
    print('these are our parameter sets:\n{}'.format(testingParams))
    print('these are the confusion matrices:\n{}'.format(avgCMs))
    valsToCompare = [ x.ravel()[cmPosition] for x in CMsList]
    print('we will be comparing these values:\n{}'.format(valsToCompare))
    if criteria == 'min':
        valWanted = min(valsToCompare)
    else:
        valWanted = max(valsToCompare)
    print('we want to select this value:\n{}'.format(valWanted))
    boolFilter = valsToCompare == valWanted
    optParamsID = np.array(list(avgCMs.keys()))[boolFilter][0]
    print('the optimal parameter set ID is: {}'.format(optParamsID))
    optParamSet = testingParams[optParamsID]
    print('the optimal parameter set is:\n{}'.format(optParamSet))
    return optParamSet

optUMAPFileName = {'testingParams': './modelOptimization/UMAPtestingParams.pkl',
                           'allResults': './modelOptimization/UMAPallResults.pkl',
                           'yTrain': './modelOptimization/UMAPyTrain.pkl',
                           'allCMs': './modelOptimization/UMAPallCMs.pkl',
                           'avgCMs': './modelOptimization/UMAPavgCMs.pkl',
                           'optParamSet': './modelOptimization/UMAPoptParamSet.pkl'}


def runUMAPParamSearch(trainDf, 
                       targetColName='diagnosis', 
                       nTrials=30, 
                       testingParams=testingParams, 
                       criteria=criteria,
                       metric=metric,
                       saveOutputFiles=optUMAPFileName,):
    sys.stdout = open("./data/UMAPOptSearchConsoleOutput.txt", "w")
    allResults, testingParams, yTrain = calcManyUMAPResults(trainDf, targetColName=targetColName, nTrials=nTrials, testingParams=testingParams)
    allCMs, testingParams = getManyUMAPResultsCMs(allResults, testingParams, yTrain)
    avgCMs, testingParams = evaluateManyCMs(allCMs, testingParams)
    optParamSet = returnUMAPOptParams(avgCMs, testingParams, criteria=criteria, metric=metric)
    if saveOutputFiles:
        print("saving output files...")
        with open(optUMAPFileName['testingParams'], 'wb') as handle:
            pickle.dump(testingParams, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
        with open(optUMAPFileName['allResults'], 'wb') as handle:
            pickle.dump(allResults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
        with open(optUMAPFileName['allCMs'], 'wb') as handle:
            pickle.dump(allCMs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
        with open(optUMAPFileName['avgCMs'], 'wb') as handle:
            pickle.dump(avgCMs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
        with open(optUMAPFileName['optParamSet'], 'wb') as handle:
            pickle.dump(optParamSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
        with open(optUMAPFileName['yTrain'], 'wb') as handle:
            pickle.dump(yTrain, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
        print("done!")
    sys.stdout.close()
    return optParamSet

optNeurNetNames = {'selectedUMAPResults': './modelOptimization/NeurNetSelectedUMAPResults.pkl',
                   'selectedKMeansModel': './modelOptimization/NeurNetSelectedKMeansModel.pkl'}


def getSelectedUMAPAndKMeansModels(xTrainDf, params, optNeurNetNames=optNeurNetNames):
    selectedUMAPResults = getUMAPResults(xTrainDf, params)
    selectedKMeansModel = KMeans(n_clusters=4).fit(selectedUMAPResults)
    if optNeurNetNames:
        with open(optNeurNetNames['selectedUMAPResults'], 'wb') as handle:
            pickle.dump(selectedUMAPResults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
        with open(optNeurNetNames['selectedKMeansModel'], 'wb') as handle:
            pickle.dump(selectedKMeansModel, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del handle
    return (selectedUMAPResults, selectedKMeansModel)

def subSplitDataRando(x, UMAPResultDf):
    x = x.copy()
    x['umapX'] = UMAPResultDf['umapX']
    x['umapY'] = UMAPResultDf['umapY']
    mDf = x[ x['diagnosis'] == 1 ].copy()
    bDf = x[ x['diagnosis'] == 0 ].copy()
    mRando = np.random.uniform(size=mDf.shape[0])
    bRando = np.random.uniform(size=bDf.shape[0])
    mDf['split'] = splitVec(mRando)
    bDf['split'] = splitVec(bRando)
    x = pd.concat([mDf, bDf])
    trainDf = x[ x['split'] == 'train' ].copy()
    devDf = x[ x['split'] == 'dev' ].copy()
    testDf = x[ x['split'] == 'test' ].copy()
    del x
    del trainDf['split'], devDf['split'], testDf['split']
    return (trainDf, devDf, testDf)

def formatDfForNN(trainDfL2):
    xTrainL2 = trainDfL2.copy().drop(['diagnosis', 'umapX', 'umapY'], axis=1).values
    yTrainL2 = trainDfL2.copy()[['umapX', 'umapY']].values
    return (xTrainL2, yTrainL2)



neurNetParams = {'learning_rate': 0.0001, 'loss': 'Huber', 'activation': tf.nn.swish}

def trainNeurNetModel(xTrainL2, yTrainL2, neurNetParams=neurNetParams):
    learnRate = neurNetParams['learning_rate']
    lossFunc = neurNetParams['loss']
    actvFunc = neurNetParams['activation']

    model = kr.Sequential([
    kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(2, activation=actvFunc),
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learnRate, amsgrad=True),loss=lossFunc,)
    model.fit( xTrainL2, yTrainL2, epochs=1000)
    return model

def trainNeurNetModelBigBatch(xTrainL2, yTrainL2, neurNetParams=neurNetParams):
    learnRate = neurNetParams['learning_rate']
    lossFunc = neurNetParams['loss']
    actvFunc = neurNetParams['activation']

    model = kr.Sequential([
    kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(2, activation=actvFunc),
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learnRate, amsgrad=True),loss=lossFunc,)
    model.fit( xTrainL2, yTrainL2, epochs=1000, batch_size=2048)
    return model


def neurNetPred(neurNetModel, xDataL2):
    predictions = neurNetModel.predict(xDataL2)
    predictedDf = pd.DataFrame(predictions, columns=['umapXpredicted','umapYpredicted'])
    return predictedDf

def combineNeurNetPredictedAndTarget(predictedDf, yDataDfL2):
    preds = predictedDf.copy()
    preds.columns = ['umapX','umapY']
    preds['Type'] = 'Predicted'
    yDataDfL2 = pd.DataFrame(yDataDfL2, columns=['umapX','umapY'])
    yDataDfL2['Type'] = 'Target'
    combinedDf = pd.concat([yDataDfL2, preds ], ignore_index=True)
    return combinedDf

# paramMap = {'learning_rate': [0.001, 0.0001, 0.00001],
#                   'loss': ['MSE', 'MAPE', 'Huber'],
#                   'nNeurons': [120, 240, 480]}

def paramMapToParamList(paramMap):
    allKeys = list(paramMap.keys())
    allVals = list(paramMap.values())
    paramCombos = list(itertools.product(*allVals))
    paramsList = []
    for aParam in paramCombos:
        tempParamDict = {}
        for i in np.arange(0,len(aParam)):
            tempParamDict[allKeys[i]] = aParam[i]
        paramsList.append(tempParamDict)
    return paramsList

# neurNetParamsList = paramMapToParamList(paramMap)

optNeurNetNames = {'selectedUMAPResults': './modelOptimization/NeurNetSelectedUMAPResults.pkl',
                   'selectedKMeansModel': './modelOptimization/NeurNetSelectedKMeansModel.pkl',
                   'trainL2Results': './modelOptimization/NeurNetTrainL2Results.pkl',
                   'devL2Results': './modelOptimization/NeurNetDevL2Results.pkl'}

def calcManyNeurNetPredics(trainDfL2, devDfL2, neurNetParamsList, optNeurNetNames):
    print('******************************************************')
    print('going to train neural nets, using trainDfL2, for {} sets of parameters'.format(len(neurNetParamsList)))
    xTrainL2, yTrainL2 = formatDfForNN(trainDfL2)
    xDevL2, yDevL2 = formatDfForNN(devDfL2)
    neurNetResultsTrain = {}
    neurNetResultsDev = {}
    print('we will be using the the trainDfL2 data for training, and predicting devDfL2 data')
    for paramSet_i in np.arange(0, len(neurNetParamsList)):
        curNeurNetModel = trainNeurNetModel(xTrainL2, yTrainL2, neurNetParams=neurNetParamsList[paramSet_i])
        curPredsTrain = neurNetPred(curNeurNetModel, xTrainL2)
        curPredsDev = neurNetPred(curNeurNetModel, xDevL2)
        neurNetResultsTrain[paramSet_i] = curPredsTrain
        neurNetResultsDev[paramSet_i] = curPredsDev
        del curNeurNetModel
        gc.collect()
    with open(optNeurNetNames['trainL2Results'], 'wb') as handle:
        pickle.dump(neurNetResultsTrain, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    with open(optNeurNetNames['devL2Results'], 'wb') as handle:
        pickle.dump(neurNetResultsDev, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    return (neurNetResultsTrain, neurNetResultsDev, neurNetParamsList, yTrainL2, yDevL2)

def getGetOneNeurNetCM(neurNetPrediction, kMeansModel, yTargets):
    predictedKMeansClusters = kMeansModel.predict(neurNetPrediction)
    negClust, posClust = getTopClusts(neurNetPrediction, kMeansModel, yTargets)
    negBool = predictedKMeansClusters == negClust
    posBool = predictedKMeansClusters == posClust          
    negTar = yTargets[negBool].values
    posTar = yTargets[posBool].values
    allTar = np.concatenate((negTar, posTar), axis=None)         
    negPred = np.zeros(negTar.shape)
    posPred = np.ones(posTar.shape)
    allPreds = np.concatenate((negPred, posPred), axis=None) 
    print("the shape of the targets is: {}".format(allTar.shape))
    print("the shape of the predictions is: {}".format(allPreds.shape))
    tempCM = confusion_matrix(allTar, allPreds)
    return tempCM

def getAllNeurNetCMs(neurNetPredictions, kMeansModel, yTargets, neurNetParamsList):
    allNeurNetCMs = {}
    for key in neurNetPredictions.keys():
        result = neurNetPredictions[key]
        tempCM = getGetOneNeurNetCM(result, kMeansModel, yTargets)
        allNeurNetCMs[key] = tempCM
    return (allNeurNetCMs, neurNetParamsList)

def selectNNParams(neurNetResultsTrain, neurNetResultsDev, neurNetParamsList, xTrainDfL2, yDevDfL2, selectedKMeansModel):
	allCMs = {}
	for key in neurNetResultsTrain.keys():
		predictedKMeansClusters = selectedKMeansModel.predict(neurNetResultsDev[key])  
		negClust, posClust = getTopClusts(neurNetResultsTrain[key], selectedKMeansModel, xTrainDfL2)
		negBool = predictedKMeansClusters == negClust
		posBool = predictedKMeansClusters == posClust 
		negTar = yDevDfL2[negBool].values
		posTar = yDevDfL2[posBool].values
		allTar = np.concatenate((negTar, posTar), axis=None)
		negPred = np.zeros(negTar.shape)
		posPred = np.ones(posTar.shape)
		allPreds = np.concatenate((negPred, posPred), axis=None) 
		tempCM = confusion_matrix(allTar, allPreds)
		allCMs[key] = tempCM
	return (allCMs, neurNetParamsList)

def trainLogRegOnClusts(trainedNeurNet, trainedKMeans, xTrainData, yTrainData, topClusts):
    negClust = topClusts[0]
    posClust = topClusts[1]
    reducedDimsData = trainedNeurNet.predict(xTrainData)
    clustPred = trainedKMeans.predict(reducedDimsData)
    logRegModels = {}
    nClusters = trainedKMeans.get_params()['n_clusters']
    for clus_i in np.arange(0, nClusters):
        if (clus_i != negClust) & (clus_i != posClust):
        	isClustBool = clustPred == clus_i
        	xTrainSub = xTrainData[isClustBool]
        	yTrainSub = yTrainData[isClustBool]
        	tempLogReg = skLogReg(max_iter=3000)
        	tempLogReg.fit(xTrainSub, yTrainSub)
        	logRegModels[clus_i] = tempLogReg
    return logRegModels


def makePred(oneObs, trainedNeurNet, trainedKMeans, topClusts, logRegModels):
    oneObs = np.array([oneObs])
    negClust = topClusts[0]
    posClust = topClusts[1]
    reducedDims = trainedNeurNet.predict(oneObs)
    clustPred = trainedKMeans.predict(reducedDims)[0]
    if clustPred == negClust:
        return 0
    elif clustPred == posClust:
        return 1
    else:
        return logRegModels[clustPred].predict(oneObs)[0]


neurNetParams = {'learning_rate': 0.0001, 'loss': 'MSLE', 'activation': tf.nn.swish, 'metrics': 'FalseNegatives'}


def trainSubNeurNetModel(xTrainL2, yTrainL2, neurNetParams=neurNetParams):
    learnRate = neurNetParams['learning_rate']
    lossFunc = neurNetParams['loss']
    actvFunc = neurNetParams['activation']
    theMetric = neurNetParams['metrics']

    model = kr.Sequential([
    kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(240, activation=actvFunc),
        kr.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learnRate, amsgrad=True),loss=lossFunc, metrics=theMetric)
    model.fit( xTrainL2, yTrainL2, epochs=500, batch_size=2048)
    return model

neurNetParams = {'learning_rate': 0.001, 'loss': 'binary_crossentropy', 'activation': tf.nn.swish, 'metrics': 'FalseNegatives'}

def trainNeurNetOnClusts(trainedNeurNet, trainedKMeans, xTrainData, yTrainData, neurNetParams=neurNetParams):
    

    reducedDimsData = trainedNeurNet.predict(xTrainData)
    clustPred = trainedKMeans.predict(reducedDimsData)
    NeurNetModels = {}
    nClusters = trainedKMeans.get_params()['n_clusters']
    for clus_i in np.arange(0, nClusters):
        isClustBool = clustPred == clus_i
        xTrainSub = xTrainData[isClustBool]
        yTrainSub = yTrainData[isClustBool]
        tempNN = trainSubNeurNetModel(xTrainSub, yTrainSub, neurNetParams)
        NeurNetModels[clus_i] = tempNN
        del tempNN
    return NeurNetModels

def makePred(oneObs, trainedNeurNet, trainedKMeans, NeurNetModels):
    oneObs = np.array([oneObs])
    reducedDims = trainedNeurNet.predict(oneObs)
    clustPred = trainedKMeans.predict(reducedDims)[0]
    return NeurNetModels[clustPred].predict(oneObs)[0]

def subSplitTrainData(x, UMAPResultDf):
    x = x.copy()
    x['x'] = UMAPResultDf['x']
    x['y'] = UMAPResultDf['y']
    x['cluster'] = UMAPResultDf['cluster']
    mDf = x[ x['diagnosis'] == 'M' ].copy()
    bDf = x[ x['diagnosis'] == 'B' ].copy()
    mRando = np.random.uniform(size=mDf.shape[0])
    bRando = np.random.uniform(size=bDf.shape[0])
    mDf['split'] = splitVec(mRando)
    bDf['split'] = splitVec(bRando)
    x = pd.concat([mDf, bDf])
    trainDf = x[ x['split'] == 'train' ].copy()
    devDf = x[ x['split'] == 'dev' ].copy()
    testDf = x[ x['split'] == 'test' ].copy()
    del x
    del trainDf['split'], devDf['split'], testDf['split']
    return (trainDf, devDf, testDf)


def labelTrainDevTest(x):
    if x < .70:
        return 'train'
    elif x >= .85:
        return 'test'
    else:
        return 'dev'

splitVec = np.vectorize(labelTrainDevTest)

def splitDataRando(x):
    mDf = x[ x['diagnosis'] == 'M' ].copy()
    bDf = x[ x['diagnosis'] == 'B' ].copy()
    mRando = np.random.uniform(size=mDf.shape[0])
    bRando = np.random.uniform(size=bDf.shape[0])
    mDf['split'] = splitVec(mRando)
    bDf['split'] = splitVec(bRando)
    x = pd.concat([mDf, bDf])
    trainDf = x[ x['split'] == 'train' ].copy()
    devDf = x[ x['split'] == 'dev' ].copy()
    testDf = x[ x['split'] == 'test' ].copy()
    del x
    del trainDf['split'], devDf['split'], testDf['split']
    return (trainDf, devDf, testDf)


def loadRawFromSQL():
    engine = sqlalchemy.create_engine(projectDBName, echo=False)
    conn = engine.connect()
    sqlString = "SELECT * FROM rawdata"
    out = pd.read_sql(sql=sqlString, con=conn, index_col='id')
    return out

def initialSplitSQL2():
    engine = sqlalchemy.create_engine(projectDBName, echo=False)
    conn = engine.connect()
    rawBCancerDF = loadRawFromSQL()
    initialTrain, initialDev, initialTest = splitDataRando(rawBCancerDF)
    return (initialTrain, initialDev, initialTest)


def trainLogRegOnClusts(trainedNeurNet, trainedKMeans, xTrainData, yTrainData):
    reducedDimsData = trainedNeurNet.predict(xTrainData)
    clustPred = trainedKMeans.predict(reducedDimsData)
    logRegModels = {}
    nClusters = trainedKMeans.get_params()['n_clusters']
    for clus_i in np.arange(0, nClusters):
        isClustBool = clustPred == clus_i
        xTrainSub = xTrainData[isClustBool]
        yTrainSub = yTrainData[isClustBool]
        tempLogReg = skLogReg(max_iter=3000)
        tempLogReg.fit(xTrainSub, yTrainSub)
        logRegModels[clus_i] = tempLogReg
    return logRegModels

def trainLogRegOnClusts2(umapResults, trainedKMeans, xTrainData, yTrainData):
    reducedDimsData = umapResults
    clustPred = trainedKMeans.predict(reducedDimsData)
    logRegModels = {}
    nClusters = trainedKMeans.get_params()['n_clusters']
    for clus_i in np.arange(0, nClusters):
        isClustBool = clustPred == clus_i
        xTrainSub = xTrainData[isClustBool]
        yTrainSub = yTrainData[isClustBool]
        tempLogReg = skLogReg(max_iter=3000)
        tempLogReg.fit(xTrainSub, yTrainSub)
        logRegModels[clus_i] = tempLogReg
    return logRegModels

def trainNeurNetOnClusts2(umapResults, trainedKMeans, xTrainData, yTrainData, neurNetParams=neurNetParams):
    reducedDimsData = umapResults
    clustPred = trainedKMeans.predict(reducedDimsData)
    NeurNetModels = {}
    nClusters = trainedKMeans.get_params()['n_clusters']
    for clus_i in np.arange(0, nClusters):
        isClustBool = clustPred == clus_i
        xTrainSub = xTrainData[isClustBool]
        yTrainSub = yTrainData[isClustBool]
        tempNN = trainSubNeurNetModel(xTrainSub, yTrainSub, neurNetParams)
        NeurNetModels[clus_i] = tempNN
        del tempNN
    return NeurNetModels

def makePred2(oneObs, trainedNeurNet, trainedKMeans, NeurNetModels, umapResults):
    oneObs = np.array([oneObs])
    reducedDims = umapResults
    clustPred = trainedKMeans.predict(reducedDims.reshape(1,-1))[0]
    return NeurNetModels[clustPred].predict(oneObs)[0]