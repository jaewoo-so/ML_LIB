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
    param_list['xgb'] = mp .param_xgb ('binary' , len(np.unique(ytrain)) , use_gpu= False )
    param_list['lgb'] = mp .param_lgbm('binary' , len(np.unique(ytrain)) , use_gpu= False )
    param_list['cat'] = mp .param_cat ('binary' , use_gpu= True , is_unbalance= False )
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
    model_dict = OrderedDict()    
    model_dict['xgb']  = mi.myXGBBinary()
    model_dict['lgb']  = mi.myLGBMBinary()
    model_dict['cat']  = mi.myCatBoostBinary()
    model_dict['rfc']  = mi.myRandomForestBinary()
    model_dict['svm']  = mi.mySVMBinary()
    model_dict['gpc']  = mi.myGPBinary()
    model_dict['lda']  = mi.myLDABinary()
    model_dict['qda']  = mi.myQDABinary()
    
    param_list = OrderedDict ( )
    param_list['xgb'] = mp .param_xgb ('binary' , len(np.unique(y)) , use_gpu= False )
    param_list['lgb'] = mp .param_lgbm('binary' , len(np.unique(y)) , use_gpu= False )
    param_list['cat'] = mp .param_cat ('binary' , use_gpu= True , is_unbalance= False )
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
        test_fold_index , oof, model_list = tr.training_Testfold('binary' , model_dict[name] , param_list[name] , fitpm_list[name] ,  metric_func , 
                                                                     X , y , nfold_test , nfold_val , verbose ) 
        result_list[name] = [test_fold_index , oof, model_list]
        auc_score_list[name] = roc_auc_score(y, oof.mean(axis = 1) )
    print('Test_Classification_TestFold Compelte')    
    return result_list
#endregion

if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    y = data.target

    xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = 0.2 )

    Test_Binary(xtrain , ytrain , xtest , 5,False)
    Test_Binary_TestFold(X,y,5,5)