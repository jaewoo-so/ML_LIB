from sklearn.model_selection import train_test_split , KFold , StratifiedKFold
import numpy as np
from time import time
import datetime
from collections import OrderedDict
import pandas as pd
import copy

# 이경우는 early stopping 안쓴다. 최적 iter는 미리 구해놔야 한다.
# 각 stacking에 사용되는 모델은 저장이 안된다. 
# 테스트 세트 고정인 경우 
def training_blending_fixedTest(mode, 
                        model_generator, 
                        model_params, 
                        training_params, 
                        metric_func, 
                        X, y, X_test,
                        nfold , nradom=7, verbose=False):
    '''
    def metric_func(y , predict_proba) -> obj
    return -> train_pred_new_feature, test_new_feature , fold_metric_score
    '''
    starttime = time()
    if verbose:
        print('-'*100)
        print('Training Model : {}'.format( model_generator.__class__.__name__))
  
    if mode == 'classification':
        train_pred = np.zeros( ( X.shape[0]  , len(np.unique(y))) , dtype = np.float)
    else:
        train_pred = np.zeros( (X.shape[0] ) , dtype = np.float)

    kfold = fold_splitter(mode, X, y, nfold, nradom)
    
    # result container
    
    test_pred = np.empty( (X_test.shape[0] , 1 ) )
    fold_metric = OrderedDict()
    for i, (train_index, val_index) in enumerate(kfold.split(X, y)):
        if type(X) == pd.core.frame.DataFrame:
            xtrain, xval = X.iloc[train_index], X.iloc[val_index]
            ytrain, yval = y.iloc[train_index], y.iloc[val_index]
        else:
            xtrain, xval = X[train_index], X[val_index]
            ytrain, yval = y[train_index], y[val_index]

        model = copy.deepcopy(model_generator).make(model_params)
        model.fit(xtrain, ytrain, training_params)  # train셋으로 훈련
        pred_on_val = model.predict_proba(xval)
        # 저장
        if mode == 'classification':
            train_pred[val_index , : ] = pred_on_val
        else:
            train_pred[val_index] = pred_on_val
            
        
        test_pred = np.append(test_pred, model.predict_proba(X_test), axis=1)

        # 폴드별 스코어
        fold_metric['fold' + str(i)] = metric_func( yval , train_pred[val_index])

    test_pred = test_pred.mean(axis=1)
    return train_pred , test_pred , fold_metric



def training_blending_Testfold_noVal( mode, 
                       model_generator, 
                       model_params, 
                       training_params, 
                       metric_func, 
                       X, y,  
                       test_nfold , nradom = 7 , verbose = False) :

    starttime = time()
    if verbose:
        print('-'*100)
        print('-'*100)
        print('** Test Fold {} on Model : {} **'.format( test_nfold , model_generator.__class__.__name__))

    

    if mode == 'classification':
        train_pred = np.zeros( ( X.shape[0]  , len(np.unique(y))) , dtype = np.float)
    else:
        train_pred = np.zeros( (X.shape[0] ) , dtype = np.float)

    test_fold_index = OrderedDict()
    fold_train_pred = OrderedDict()
    fold_test_pred = OrderedDict()
    mean_fold_score = OrderedDict()
    kfold = fold_splitter(mode , X , y , test_nfold , nradom)

    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        if verbose:
            print()
            print('* Test Fold {} *'.format(i))
        if type(X) == pd.core.frame.DataFrame:
            xtrain, xtest = X.loc[train_index], X.loc[test_index]
            ytrain, ytest = y.loc[train_index], y.loc[test_index]
            
        else:
            xtrain, xtest = X[train_index], X[test_index]
            ytrain, ytest = y[train_index], y[test_index]
            

        # test_fold 인덱스 저장 : 문제 없음
        test_fold_index['fold'+str(i)] = test_index

        # 폴드 실행
        train_pred , test_pred , fold_metric = training_blending_fixedTest(mode, model_generator, 
                                                                model_params, 
                                                                training_params, 
                                                                metric_func, 
                                                                xtrain, ytrain, xtest, verbose=verbose)
        
        fold_train_pred['fold' + str(i)] = train_pred
        fold_test_pred['fold' + str(i)] = test_pred
        
        mean_fold_score['fold'+str(i)] = fold_metric.mean()

    return test_fold_index , fold_train_pred , fold_test_pred, mean_fold_score







def fold_splitter(mode , X,y , nfold , nrandom):
    if mode == 'regression':
        kfold = KFold(n_splits=nfold, random_state=nrandom, shuffle=True)
    elif (mode == 'classification') or (mode == 'binary'):
        kfold = StratifiedKFold(n_splits=nfold, random_state=nrandom, shuffle=False)
    return kfold