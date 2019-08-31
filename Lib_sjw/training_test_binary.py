import Lib_CDSS.training as tr
import Lib_CDSS.model_interface as mi
import Lib_CDSS.model_parmas as mp
import Lib_CDSS.evaluator as ev
import Lib_CDSS.classification_util as cu


from sklearn.datasets import load_boston , load_iris , load_breast_cancer
from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score
from sklearn.model_selection import train_test_split
import numpy as np

from collections import OrderedDict


'''
1. 데이터 만들기
2. 모델 제너레이터 만들기 
3. 모델 생성 파라미터, 피팅 파라미터 만들기 
4. 성능평가 메트릭 정의하기 :  f : y , pred -> value
5. training_fixedTest 실행하기 
6. 유틸에 테스트 데이터에 대한 성능평가, 또는 저장 등을 하기 
'''

def Test_Classification_TestFold(verbose = True):
    data = load_iris()
    X = data.data
    y = data.target

    # name list 
    name_list = ['xgb',
                 'lgb',
                 'cat',
                 'rfc',
                 'elt',
                 'svm',
                 'gpc',
                 'elm',
                 'lda',
                 'qda']
    # model_list , param_list , 
    model_list = OrderedDict()    
    model_list['xgb']  = mi.myXGBClassifier()
    model_list['lgb']  = mi.myLGBMClassifier()
    model_list['cat']  = mi.myCatBoostClassifier()
    model_list['rfc']  = mi.myRandomForestClassifier()
    model_list['elt']  = mi.myElasticNetClassifier()
    model_list['svm']  = mi.mySVMClassifier()
    model_list['gpc']  = mi.myGPClassifier()
    model_list['elm']  = mi.myELMClassifier()
    model_list['lda']  = mi.myLDAClassifier()
    model_list['qda']  = mi.myQDAClassifier()
    
    param_list = OrderedDict ( )
    param_list['xgb'] = mp .param_xgb ('classification' , len(np.unique(y)) , use_gpu= False )
    param_list['lgb'] = mp .param_lgbm('classification' , len(np.unique(y)) , use_gpu= False )
    param_list['cat'] = mp .param_cat ('classification' , use_gpu= True , is_unbalance= False )
    param_list['rfc'] = mp .param_rf ('classification' )
    param_list['elt'] = mp .param_elst('classification' )
    param_list['svm'] = mp .param_svm ('classification' )
    param_list['gpc'] = mp .param_gpc ('classification' )
    param_list['elm'] = mp .param_elm ( )
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
    metric_func = cu.aucpr
    
    # REsult
    result_list = OrderedDict()
    # Training 
    auc_score_list = OrderedDict()
    for name in name_list:
        print(name)
        fold_predict , fold_oof , fold_metric = tr.training_Testfold('classification' , model_list[name] , param_list[name] , fitpm_list[name] ,  metric_func , X , y , 5 , 5 ) 
        result_list[name] = [fold_predict , fold_oof , fold_metric]
        auc_score_list[name] = roc_auc_score(np.where(y > 25 , 1 ,0 ) , fold_oof.mean(axis = 1))
    print('Test_Classification_TestFold Compelte')    


def Test_Classification():
    data = load_iris()
    X = data.data
    y = data.target
    
    xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = 0.2 )
    
    # name list 
    name_list = ['xgb',
                 'lgb',
                 'cat',
                 'rfc',
                 'elt',
                 'svm',
                 'gpc',
                 'elm',
                 'lda',
                 'qda']
    # model_list , param_list , 
    model_list = OrderedDict()    
    model_list['xgb']  = mi.myXGBClassifier()
    model_list['lgb']  = mi.myLGBMClassifier()
    model_list['cat']  = mi.myCatBoostClassifier()
    model_list['rfc']  = mi.myRandomForestClassifier()
    model_list['elt']  = mi.myElasticNetClassifier()
    model_list['svm']  = mi.mySVMClassifier()
    model_list['gpc']  = mi.myGPClassifier()
    model_list['elm']  = mi.myELMClassifier()
    model_list['lda']  = mi.myLDAClassifier()
    model_list['qda']  = mi.myQDAClassifier()
    
    param_list = OrderedDict()
    param_list['xgb'] = mp.param_xgb('classification' , len(np.unique(y)), use_gpu= False)
    param_list['lgb'] = mp.param_lgbm('classification' , len(np.unique(y)), use_gpu= False)
    param_list['cat']  = mp.param_cat('classification' , use_gpu= True , is_unbalance= False )
    param_list['rfc']  = mp.param_rf('classification')
    param_list['elt']  = mp.param_elst('classification')
    param_list['svm']  = mp.param_svm('classification')
    param_list['gpc']  = mp.param_gpc('classification')
    param_list['elm']  = mp.param_elm()
    param_list['lda']  = mp.param_lda()
    param_list['qda']  = mp.param_qda()
    
    #fitting parmas
    fitpm_list = OrderedDict()

    for name in name_list:
            fitpm_list[name] = {}
    fitpm_list['lgb'] = {'early_stopping_rounds' : 12 , 'verbose' : -1}
    #fit_cat = {}
    #fit_xgb = {}
    
    # metric func
    metric_func = cu.aucpr
    
    # REsult
    result_list = OrderedDict()
    # Training 
    for name in name_list:
        print(name)
        fold_predict , fold_oof , fold_metric , fold_model = tr.training_fixedTest('classification' , model_list[name] , param_list[name] , fitpm_list[name] ,  metric_func , xtrain , ytrain , xtest , 5  ) 
        result_list[name] = [fold_predict , fold_oof , fold_metric]
    print('Test_Classification Complete')


def Test_Regression_TestFold(verbose = True):
    data = load_boston()
    X = data.data
    y = data.target
    
    # name list 
    name_list = ['xgb',
                 'lgb',
                 'cat',
                 'rfc',
                 'elt',
                 'svm',
                 'gpc',
                 'elm',
                 'lda',
                 'qda']
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

    auc_score_list = OrderedDict()
    for name in name_list:
        print(name)
        fold_predict , fold_oof , fold_metric = tr.training_Testfold('regression' , model_list[name] , param_list[name] , fitpm_list[name] ,  metric_func , X , y , 5 , 5 ) 
        result_list[name] = [fold_predict , fold_oof , fold_metric]
        auc_score_list[name] = roc_auc_score(np.where(y > 25 , 1 ,0 ) , fold_oof.mean(axis = 1))
    print('Test_Regression_TestFold Complete')

def Test_Regression():
    data = load_boston()
    X = data.data
    y = data.target
    
    xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = 0.2 )

    
    # name list 
    name_list = ['xgb',
                 'lgb',
                 'cat',
                 'rfc',
                 'elt',
                 'svm',
                 'gpc',
                 'elm',
                 'lda',
                 'qda']
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
    param_list['xgb']['num_class'] = 3
    param_list['lgb'] = mp.param_lgbm('regression' , use_gpu= False)
    param_list['lgb']['num_class'] = 3
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

    auc_score_list = OrderedDict()
    for name in name_list:
        print(name)
        fold_predict , fold_oof , fold_metric = tr.training_fixedTest('regression' , model_list[name] , param_list[name] , fitpm_list[name] ,  metric_func , xtrain , ytrain , xtest , 5  ) 
        result_list[name] = [fold_predict , fold_oof , fold_metric]
    print('Test_Regression Complete')
    
Test_Classification()
Test_Classification_TestFold(False)
Test_Regression()
Test_Regression_TestFold(False)
print('done')