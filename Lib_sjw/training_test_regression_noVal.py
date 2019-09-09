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

def Test_Regression_noVal(xtrain , xtest , ytrain ):
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
        res_pred , model = tr.training_fixedTest_noVal('regression' , model_list[name] , param_list[name] , fitpm_list[name] ,  metric_func , xtrain , ytrain , xtest ) 
        result_list[name] = [res_pred , model]
    print('Test_Regression Complete')
    

def Test_Regression_TestFold_noVal(X , y , nfold_test , verbose = True):
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
        test_fold_index , oof, model_list = tr.training_Testfold_noVal('regression' , model_dict[name] , param_list[name] , fitpm_list[name] ,  metric_func , X , y , nfold_test ) 
        result_list[name] = [test_fold_index , oof, model_list] # 모든 데이터에 대해 예측값이 oof에 저장되어 있다. 
        auc_score_list[name] = roc_auc_score(np.where(y > 25 , 1 ,0 ) , oof)
    return result_list
    

if __name__ == '__main__':
    data = load_boston()
    X = data.data[:20]
    y = data.target[:20]
    
    xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = 0.2 )
    
    res1 = Test_Regression_noVal(xtrain , xtest , ytrain , ytest)
    res2 = Test_Regression_TestFold_noVal(X,y,5)

    print('done')