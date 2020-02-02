import os
os.chdir(os.path.dirname(__file__))
import sys
sys.path.append(r'E:\01_PProj\ML_LIB')
sys.path.append(r'D:\00_work\ML_LIB')


import Lib_sjw.training as tr
import Lib_sjw.model_interface as mi
import Lib_sjw.model_parmas as mp
import Lib_sjw.evaluator as ev
import Lib_sjw.classification_util as cu


from sklearn.datasets import load_boston , load_iris , load_breast_cancer
from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from collections import OrderedDict


'''
1. 데이터 만들기
2. 모델 제너레이터 만들기 
3. 모델 생성 파라미터, 피팅 파라미터 만들기 
4. 성능평가 메트릭 정의하기 :  f : y , pred -> value
5. training_fixedTest 실행하기 
6. 유틸에 테스트 데이터에 대한 성능평가, 또는 저장 등을 하기 
'''
def Test_Regression(xtrain , ytrain , xtest  , nfold = 5):
    # name list 
    name_list = ['xgb',
                 'lgb',
                 'cat',
                 'rfc',
                 'elt',
                 'svm',
                 'gpc']
    # model_list , param_list , 
    model_list = OrderedDict()      
    model_list['xgb']  = mi.myXGBRegressor()
    model_list['lgb']  = mi.myLGBMRegressor()
    model_list['cat']  = mi.myCatBoostRegressor()
    model_list['rfc']  = mi.myRandomForestRegressor()
    model_list['elt']  = mi.myElasticNetRegressor()
    model_list['svm']  = mi.mySVMRegressor()
    model_list['gpc']  = mi.myGPRegressor() 
    
    param_list = OrderedDict()
    param_list['xgb'] = mp.param_xgb('regression' , use_gpu= False)
    param_list['lgb'] = mp.param_lgbm('regression' , use_gpu= False)
    param_list['cat']  = mp.param_cat('regression' , use_gpu= True , is_unbalance= False )
    param_list['rfc']  = mp.param_rf('regression')
    param_list['elt']  = mp.param_elst('regression')
    param_list['svm']  = mp.param_svm('regression')
    param_list['gpc']  = mp.param_gpc('regression')    
    #fitting parmas
    fitpm_list = OrderedDict()
    for name in name_list:
            fitpm_list[name] = {}
    fitpm_list['lgb'] = {'early_stopping_rounds' : 12 , 'verbose' : -1}
    
    # metric func
    metric_func = mean_squared_error
    result_list = OrderedDict()   
    for name in name_list:
        print(name)
        fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest('regression' , model_list[name] , param_list[name] , fitpm_list[name] ,  metric_func , xtrain , ytrain , xtest , 5  ) 
        result_list[name] = [fold_predict, fold_oof, fold_metric, fold_models]
        
        print(fold_models['fold1'].predict_proba(xtrain)[10], fold_models['fold4'].predict_proba(xtrain)[10])
        print()

    print('Test_Regression Complete')
    return result_list
    

def Test_Regression_TestFold(X , y , nfold_test , nfold_val , verbose = True):
    # name list 
    name_list = ['xgb',
                 'lgb',
                 'cat',
                 'rfc',
                 'elt',
                 'svm',
                 'gpc']
    # model_list , param_list , 
    model_dict = OrderedDict()      
    model_dict['xgb']  = mi.myXGBRegressor()
    model_dict['lgb']  = mi.myLGBMRegressor()
    model_dict['cat']  = mi.myCatBoostRegressor()
    model_dict['rfc']  = mi.myRandomForestRegressor()
    model_dict['elt']  = mi.myElasticNetRegressor()
    model_dict['svm']  = mi.mySVMRegressor()
    model_dict['gpc']  = mi.myGPRegressor() 

    param_list = OrderedDict()
    param_list['xgb'] = mp.param_xgb('regression' , use_gpu= False)
    param_list['lgb'] = mp.param_lgbm('regression' , use_gpu= False)
    param_list['cat']  = mp.param_cat('regression' , use_gpu= True , is_unbalance= False )
    param_list['rfc']  = mp.param_rf('regression')
    param_list['elt']  = mp.param_elst('regression')
    param_list['svm']  = mp.param_svm('regression')
    param_list['gpc'] = mp.param_gpc('regression')
    #fitting parmas
    fitpm_list = OrderedDict()

    for name in name_list:
            fitpm_list[name] = {}
    fitpm_list['lgb'] = {'early_stopping_rounds' : 12 , 'verbose' : -1}
    
    # metric func
    metric_func = mean_squared_error

    auc_score_list = OrderedDict()
    result_list = OrderedDict()
    for name in name_list:
        print(name)
        print('Model : {}'.format(name))
        test_fold_index , oof, model_list = tr.training_Testfold('regression' , model_dict[name] , param_list[name] , fitpm_list[name] ,  metric_func , X , y , nfold_test , nfold_val ) 
        result_list[name] = [test_fold_index , oof, model_list] # 모든 데이터에 대해 예측값이 oof에 저장되어 있다. 
        auc_score_list[name] = roc_auc_score(np.where(y > 25 , 1 ,0 ) , oof.mean(axis = 1))
    return result_list
    

if __name__ == '__main__':
    data = load_boston()
    X = data.data
    y = data.target

    df = pd.DataFrame(X, columns=data.feature_names)
    df['target'] = y
    X = df.iloc[:,:-1]
    y = df.iloc[:, -1]

    xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = 0.2 )
    print(xtrain.shape,ytrain.values.shape)
    Test_Regression(xtrain , ytrain , xtest , 5)
    Test_Regression_TestFold(X,y,5,5)

    print('done')