import sys
sys.path.append('../')
sys.path.append('../mllib/')
import mllib

import mllib.training as tr
import mllib.model_interface as mi
import mllib.model_parmas as mp
import mllib.evaluator as ev
import mllib.classification_util as cu

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
    
from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score , roc_curve
from sklearn.model_selection import train_test_split


from mllib.data_class import data_fixed_test , data_fold_test
from mllib.report import report_binary_fixed_test , report_binary_fold_test

import numpy as np
import pandas as pd
from dotmap import DotMap
from collections import OrderedDict
from dotwiz import DotWiz

data = load_breast_cancer()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data['feature_names'])
df['target'] = y
df = df.reset_index(drop=True)
X = df.iloc[:, :-1] #.astype('float16')
y = df.iloc[:, -1]
y=y.reset_index(drop=True)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


def Test_Binary_FixedTest(xtrain , ytrain , xtest  , nfold = 5 , verbose = False):
    # name list 
    name_list = ['xgb',
                 'lgb',]

    model_list = OrderedDict()    
    model_list = OrderedDict()    
    model_list['xgb']  = mi.myXGBBinary()
    model_list['lgb']  = mi.myLGBMBinary()

    param_list = OrderedDict ( )
    param_list['xgb'] = mp .param_xgb ('binary' , len(np.unique(ytrain)) , use_gpu= False )
    param_list['lgb'] = mp .param_lgbm('binary' , len(np.unique(ytrain)) , use_gpu= False )

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

    result_all = DotWiz()

    for name in name_list:
        print(name)
        fold_predict , fold_oof  , fold_metric , fold_model = tr.training_fixedTest('binary' , model_list[name] , param_list[name] , fitpm_list[name] ,  
                                                                                    metric_func , xtrain , ytrain , xtest , nfold , verbose ) 

        
        result_all[name] = data_fixed_test.FixedTestData(fold_predict , fold_oof  , fold_metric , fold_model)
    test_result = data_fixed_test.FixedTestResult(result_all)
    return test_result

result = Test_Binary_FixedTest(xtrain , ytrain , xtest , 3,False)
report_gen = report_binary_fixed_test.ReportBinaryFixedTest()
report_gen.oof_custom_score_each_model(result, ytrain ,  precision_score )