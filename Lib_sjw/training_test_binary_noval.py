import Lib_sjw.training as tr
import Lib_sjw.model_interface as mi
import Lib_sjw.model_parmas as mp
import Lib_sjw.evaluator as ev
import Lib_sjw.classification_util as cu


from sklearn.datasets import load_boston , load_iris , load_breast_cancer
from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score
from sklearn.model_selection import train_test_split
import numpy as np

from collections import OrderedDict

# 이 코드를 참고해서 사용한다. 

'''
1. 데이터 만들기
2. 모델 제너레이터 만들기 
3. 모델 생성 파라미터, 피팅 파라미터 만들기 
4. 성능평가 메트릭 정의하기 :  f : y , pred -> value
5. training_fixedTest 실행하기 
6. 유틸에 테스트 데이터에 대한 성능평가, 또는 저장 등을 하기 
'''
def Test_Binary(xtrain , ytrain , xtest  , nfold = 5 , verbose = False):
    # name list 
    name_list = ['xgb',
                 'lgb',
                 #'cat',
                 'rfc',
                 'svm',
                 #'gpc',
                 'lda',
                 'qda',
                 'rdg',
                 'lso',
                 'ann']
    # model_dicts , param_list , 
    model_dicts = OrderedDict()    
    model_dicts['xgb']  = mi.myXGBBinary()
    model_dicts['lgb']  = mi.myLGBMBinary()
    model_dicts['cat']  = mi.myCatBoostBinary()
    model_dicts['rfc']  = mi.myRandomForestBinary()
    model_dicts['svm']  = mi.mySVMBinary()
    model_dicts['gpc']  = mi.myGPBinary()
    model_dicts['lda']  = mi.myLDABinary()
    model_dicts['qda']  = mi.myQDABinary()
    model_dicts['rdg']  = mi.myRidgeBinary()
    model_dicts['lso'] = mi.myLassoBinary()
    model_dicts['ann']  = mi.myANNBinary()
    
    param_list = OrderedDict ( )
    param_list['xgb'] = mp .param_xgb ('binary' , len(np.unique(ytrain)) , use_gpu= False )
    param_list['lgb'] = mp .param_lgbm('binary' , len(np.unique(ytrain)) , use_gpu= False )
    param_list['cat'] = mp .param_cat ('binary' , use_gpu= True , is_unbalance= False )
    param_list['rfc'] = mp .param_rf ('binary' )
    param_list['svm'] = mp .param_svm ('binary' )
    param_list['gpc'] = mp .param_gpc ('binary' )
    param_list['lda'] = mp .param_lda ( )
    param_list['qda'] = mp .param_qda ( )
    param_list['rdg'] = mp .param_ridge ( 'binary')
    param_list['lso'] = mp .param_lasso ( 'binary')
    param_list['ann']  = mp.param_ANN()
    
    
    params_xgb = {'colsample_bytree': 0.018359345409703118,
                  'max_delta_step': 10.0,
                  'max_depth': 100,
                  'min_child_weight': 0.0,
                  'n_estimators': 800,
                  'reg_alpha': 2.0,
                  'reg_lambda': 10.0,
                  'subsample': 1.0}
                     
    params_lgb =  {'bagging_fraction': 0.9146615380853989,
                    'colsample_bytree': 0.7384250232683872,
                    'feature_fraction': 0.2892361777710602,
                    'lambda_l1': 6.11807950735429,
                    'lambda_l2': 9.779990080293718,
                    'learning_rate': 0.001,
                    'max_depth': 9,
                    'min_child_weight': 0.6385281864950193,
                    'min_data_in_leaf': 1,
                    'min_split_gain': 0.5944870633301388,
                    'num_leaves': 969,
                    'reg_alpha': 18.045166839320736,
                    'reg_lambda': 8.490946187426754,
                    'subsample': 0.16006631386138065}
    '''
    params_cat = {'bagging_temperature': 36.514154289873396,
                  'depth': 7,
                  'iterations': 1884,
                  'l2_leaf_reg': 5,
                  'learning_rate': 0.8494130280301052,
                  'random_strength': 39.83926219359324}
    '''
    params_rfc = {'max_depth': 3,
              'max_features': 0.26390005062522226,
              'n_estimators': 89}
    params_rdg = {'normalize': False}
    params_lso = {'alpha': 0.03,
              'normalize': False}

    param_list['xgb']= {**param_list['xgb'], **params_xgb} # 뒤의 딕셔너리가 우선이 된다. 
    param_list['lgb']= {**param_list['lgb'], **params_lgb}
    #param_list['cat']= {**param_list['cat'], **params_cat}
    param_list['rfc']= {**param_list['rfc'], **params_rfc}
    param_list['rdg']= {**param_list['rdg'], **params_rdg}
    param_list['lso']= {**param_list['lso'], **params_lso}
    
    #fitting parmas
    fitpm_list = OrderedDict()

    for name in name_list:
            fitpm_list[name] = {}
    #fitpm_list['lgb'] = {'early_stopping_rounds' : 12 , 'verbose' : -1}
    #fit_cat = {}
    #fit_xgb = {}
    
    # metric func
    metric_func = roc_auc_score
    
    # Result
    result_list = OrderedDict()

    # Training 
    for name in name_list:
        print(name)
        prediction ,  model = tr.training_fixedTest_noVal('binary' , model_dicts[name] , param_list[name] , fitpm_list[name] ,  
                                                                                    metric_func, xtrain, ytrain, xtest, verbose)
        result_list[name] = [prediction  , model]
    return result_list


def Test_Binary_TestFold(X , y , nfold_test , nfold_val , verbose = True):
    # name list 
    name_list = ['xgb',
                 'lgb',
                 #'cat',
                 'rfc',
                 'svm',
                 #'gpc',            
                 'lda',
                 'qda',
                 'rdg',
                 'lso',
                 'ann']
    # model_dicts , param_list , 
    model_dicts = OrderedDict()    
    model_dicts['xgb']  = mi.myXGBBinary()
    model_dicts['lgb']  = mi.myLGBMBinary()
    model_dicts['cat']  = mi.myCatBoostBinary()
    model_dicts['rfc']  = mi.myRandomForestBinary()
    model_dicts['svm']  = mi.mySVMBinary()
    model_dicts['gpc']  = mi.myGPBinary()
    model_dicts['lda']  = mi.myLDABinary()
    model_dicts['qda']  = mi.myQDABinary()
    model_dicts['rdg']  = mi.myRidgeBinary()
    model_dicts['lso']  = mi.myLassoBinary()
    model_dicts['ann'] = mi.myANNBinary()
    
    param_list = OrderedDict ( )
    param_list['xgb'] = mp .param_xgb ('binary' , len(np.unique(y)) , use_gpu= False )
    param_list['lgb'] = mp .param_lgbm('binary' , len(np.unique(y)) , use_gpu= False )
    param_list['cat'] = mp .param_cat ('binary' , use_gpu= True , is_unbalance= False )
    param_list['rfc'] = mp .param_rf ('binary' )
    param_list['svm'] = mp .param_svm ('binary' )
    param_list['gpc'] = mp .param_gpc ('binary' )
    param_list['lda'] = mp .param_lda ( )
    param_list['qda'] = mp .param_qda ( )
    param_list['rdg'] = mp .param_ridge ('binary' )
    param_list['lso'] = mp .param_lasso ('binary' )
    param_list['ann']  = mp.param_ANN()
    
    
    params_xgb = {'colsample_bytree': 0.018359345409703118,
                  'max_delta_step': 10.0,
                  'max_depth': 100,
                  'min_child_weight': 0.0,
                  'n_estimators': 800,
                  'reg_alpha': 2.0,
                  'reg_lambda': 10.0,
                  'subsample': 1.0}
                     
    params_lgb =  {'bagging_fraction': 0.9146615380853989,
                    'colsample_bytree': 0.7384250232683872,
                    'feature_fraction': 0.2892361777710602,
                    'lambda_l1': 6.11807950735429,
                    'lambda_l2': 9.779990080293718,
                    'learning_rate': 0.001,
                    'max_depth': 9,
                    'min_child_weight': 0.6385281864950193,
                    'min_data_in_leaf': 1,
                    'min_split_gain': 0.5944870633301388,
                    'num_leaves': 969,
                    'reg_alpha': 18.045166839320736,
                    'reg_lambda': 8.490946187426754,
                    'subsample': 0.16006631386138065}
    '''
    params_cat = {'bagging_temperature': 36.514154289873396,
                  'depth': 7,
                  'iterations': 1884,
                  'l2_leaf_reg': 5,
                  'learning_rate': 0.8494130280301052,
                  'random_strength': 39.83926219359324}
    '''
   
    params_rfc = {'max_depth': 3,
              'max_features': 0.26390005062522226,
              'n_estimators': 89}
    params_rdg = {'normalize': False}
    params_lso = {'alpha': 0.03,
              'normalize': False}
    
    param_list['xgb']= {**param_list['xgb'], **params_xgb} # 뒤의 딕셔너리가 우선이 된다. 
    param_list['lgb']= {**param_list['lgb'], **params_lgb}
    #param_list['cat']= {**param_list['cat'], **params_cat}
    param_list['rfc']= {**param_list['rfc'], **params_rfc}
    param_list['rdg']= {**param_list['rdg'], **params_rdg}
    param_list['lso']= {**param_list['lso'], **params_lso}
    
    
    
    
    #fitting parmas
    fitpm_list = OrderedDict()
    for name in name_list:
            fitpm_list[name] = {}
    #fitpm_list['lgb'] = {'early_stopping_rounds' : 12 , 'verbose' : -1}
    #fit_cat = {}
    #fit_xgb = {}
    
    # metric func
    metric_func = roc_auc_score
    
    # Result
    result_list = OrderedDict()
    auc_score_list = OrderedDict()
    for name in name_list:
        if verbose : print(name)
        test_fold_index , oof, model_lists = tr.training_Testfold_noVal('binary' , model_dicts[name] , param_list[name] , fitpm_list[name] ,  metric_func , X , y , nfold_test , 517 , verbose ) 
        result_list[name] = [test_fold_index , oof, model_lists]
        auc_score_list[name] = roc_auc_score(y, oof )
    print('Test_Classification_TestFold Compelte')    
    return result_list


from sklearn.neural_network import MLPClassifier

import pandas as pd
if __name__ == '__main__':

    def df_test(): # 인덱스로 나눌때 iloc를 써야한다. loc쓰면 경우에 따라서 nan이 리턴되는 경우도 있었다. 
        data = load_breast_cancer()
        X = data.data
        y = data.target
        df = pd.DataFrame(X, columns=data['feature_names'])
        df['target'] = y
        df = df.reset_index(drop=True)
        X = df.iloc[:, :-1].astype('float16')
        y = df.iloc[:, -1]
        y=y.reset_index(drop=True)
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
        '''
        #확인 완료 
        print(xtrain.isna().any())
        print(xtest.isna().any())
        print(ytrain.isna().any())
        print(ytest.isna().any())
        '''
        param_ANN = dict()
        param_ANN['hidden_layer_sizes'] = (44,3)
        param_ANN['solver'] = 'adam'
        param_ANN['learning_rate_init'] = 0.01
        param_ANN['max_iter'] = 180
        mm = MLPClassifier(**param_ANN)
        mm.fit(xtrain,ytrain)

        Test_Binary(xtrain , ytrain , xtest , 5,False)
        Test_Binary_TestFold(X,y,2,2)

    def np_test():
        data = load_breast_cancer()
        X = data.data
        y = data.target
        df = pd.DataFrame()
        xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = 0.2 )
        #Test_Binary(xtrain , ytrain , xtest , 5,False)
        Test_Binary_TestFold(X, y, 5, 5)
        

    df_test()