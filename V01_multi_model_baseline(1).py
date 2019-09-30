import sys
sys.path.append(r'D:\00_work\ML_LIB')

import Lib_sjw.training as tr
import Lib_sjw.model_interface as mi
import Lib_sjw.model_parmas as mp
import Lib_sjw.evaluator as ev
import Lib_sjw.classification_util as cu
import Lib_sjw.ensemble as es
import Lib_sjw.analysis as anl
import Lib_sjw.visual_result as vr

from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score , average_precision_score , accuracy_score , mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm_notebook as tqdm

from collections import OrderedDict



lgbm_params = {'num_leaves': 546,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.1797454081646243,
          'bagging_fraction': 0.2181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.005883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3299927210061127,
          'reg_lambda': 0.3885237330340494,
          'random_state': 42,
          'use_gpu' : True
}
lgbm_params['is_unbalance'] = True


xgb_params = {'bagging_fraction': 0.8993155305338455, 
              'colsample_bytree': 0.7463058454739352, 
              'feature_fraction': 0.7989765808988153, 
              'gamma': 0.6665437467229817, 
              'learning_rate': 0.013887824598276186, 
              'max_depth': 16.0, 
              'min_child_samples': 170, 
              'num_leaves': 220, 
              'reg_alpha': 0.39871702770778467,
              'reg_lambda': 0.24309304355829786,
              'subsample': 0.7}

xgb_params['tree_method'] = 'gpu_hist'
xgb_params['predictor']    = 'gpu_predictor'
xgb_params['objective']    = 'binary:logistic'



def Test_Binary(xtrain , xtest , ytrain , nfold = 5 , verbose = False):
    # name list 
    name_list = ['xgb',
                 'lgb',
                 'cat',
                 'rfc',
                 'svm',
                 'gpc',
                 'lda',
                 'qda']
    # model_list , param_list , 
    model_list = OrderedDict()    
    model_list = OrderedDict()    
    model_list['xgb']  = mi.myXGBBinary()
    model_list['lgb']  = mi.myLGBMBinary()
    model_list['cat']  = mi.myCatBoostBinary()
    model_list['rfc']  = mi.myRandomForestBinary()
    model_list['svm']  = mi.mySVMBinary()
    model_list['gpc']  = mi.myGPBinary()
    model_list['lda']  = mi.myLDABinary()
    model_list['qda']  = mi.myQDABinary()
    
    param_list = OrderedDict ( )
    #param_list['xgb'] = mp .param_xgb ('binary' , len(np.unique(y)) , use_gpu= True )
    param_list['xgb'] = xgb_params
    #param_list['lgb'] = mp .param_lgbm('binary' , len(np.unique(y)) , use_gpu= False )
    param_list['lgb'] = lgbm_params
    param_list['cat'] = mp .param_cat ('binary' , use_gpu= True , is_unbalance= True )
    param_list['rfc'] = mp .param_rf ('binary' )
    param_list['svm'] = mp .param_svm ('binary' )
    param_list['gpc'] = mp .param_gpc ('binary' )
    param_list['lda'] = mp .param_lda ( )
    param_list['qda'] = mp .param_qda ( )
    
    #fitting parmas
    fitpm_list = OrderedDict()

    for name in name_list:
            fitpm_list[name] = {}
    fitpm_list['lgb'] = {'early_stopping_rounds' : 12 , 'verbose' : -1}
    #fit_cat = {}
    #fit_xgb = {}
    
    # metric func
    metric_func = roc_auc_score
    
    # Result
    result_list = OrderedDict()

    # Training 
    for name in name_list:
        print(name)
        fold_predict , fold_oof  , fold_metric , fold_model = tr.training_fixedTest('binary' , model_list[name] , param_list[name] , fitpm_list[name] ,  
                                                                                    metric_func , xtrain , ytrain , xtest , nfold , verbose ) 
        result_list[name] = [fold_predict , fold_oof , fold_metric , fold_model]
    return result_list

def Test_Binary_TestFold(X , y , nfold_test , nfold_val , verbose = True):
    # name list 
    name_list = ['xgb',
                 'lgb',
                 'cat',
                 'rfc',
                 'svm',
                 'gpc',            
                 'lda',
                 'qda']
    # model_list , param_list , 
    model_list = OrderedDict()    
    model_list['xgb']  = mi.myXGBBinary()
    model_list['lgb']  = mi.myLGBMBinary()
    model_list['cat']  = mi.myCatBoostBinary()
    model_list['rfc']  = mi.myRandomForestBinary()
    model_list['svm']  = mi.mySVMBinary()
    model_list['gpc']  = mi.myGPBinary()
    model_list['lda']  = mi.myLDABinary()
    model_list['qda']  = mi.myQDABinary()
    
    param_list = OrderedDict ( )
    #param_list['xgb'] = mp .param_xgb ('binary' , len(np.unique(y)) , use_gpu= True )
    param_list['xgb'] = xgb_params
    #param_list['lgb'] = mp .param_lgbm('binary' , len(np.unique(y)) , use_gpu= False )
    param_list['lgb'] = lgbm_params
    param_list['cat'] = mp .param_cat ('binary' , use_gpu= True , is_unbalance= True )
    param_list['rfc'] = mp .param_rf ('binary' )
    param_list['svm'] = mp .param_svm ('binary' )
    param_list['gpc'] = mp .param_gpc ('binary' )
    param_list['lda'] = mp .param_lda ( )
    param_list['qda'] = mp .param_qda ( )
    
    #fitting parmas
    fitpm_list = OrderedDict()
    for name in name_list:
            fitpm_list[name] = {}
    fitpm_list['lgb'] = {'early_stopping_rounds' : 12 , 'verbose' : -1}
    #fit_cat = {}
    #fit_xgb = {}
    
    # metric func
    metric_func = roc_auc_score
    
    # Result
    result_list = OrderedDict()
    auc_score_list = OrderedDict()
    for name in name_list:
        print(name)
        test_fold_index , oof, model_list = tr.training_Testfold('binary' , model_list[name] , param_list[name] , fitpm_list[name] ,  metric_func , 
                                                                     X , y , nfold_test , nfold_val , verbose ) 
        result_list[name] = [test_fold_index , oof, model_list]
        auc_score_list[name] = roc_auc_score(y, oof.mean(axis = 1) )
    print('Test_Classification_TestFold Compelte')    
    return result_list




input_folder = r'D:/04_kaggle/IEEE_CIS_Fraud_Detection/data/'
X = pd.read_hdf(input_folder + 'train_v01.h5')
X_test = pd.read_hdf( input_folder + 'test_v01.h5')
y = pd.read_hdf(input_folder + 'train_transaction.h5')['isFraud'].copy()



df = pd.concat([X,y] , axis = 1)
df = df.sample(frac=1 , random_state  = 7 ).sample(frac=1 , random_state  = 77 ).reset_index(drop=True)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]




res = Test_Binary(X,y,X_test , 10)





