import os
os.chdir(os.path.dirname(__file__))
print(os.getcwd()) # 이걸로 보고, 이거 기준으로 

import sys
sys.path.append('../') # 이런식으로 설정하면 된다.
sys.path.append(os.path.dirname(__file__))


import mllib.training as tr
import mllib.model_interface as mi
import mllib.model_parmas as mp
import mllib.evaluator as ev
import mllib.classification_util as cu

from sklearn.datasets import fetch_california_housing , load_iris , load_breast_cancer
from sklearn.datasets import fetch_openml
    
from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score , roc_curve, recall_score
from sklearn.model_selection import train_test_split
import numpy as np

from collections import OrderedDict

from mllib.data_class import data_fixed_test , data_fold_test
from mllib.report import report_binary_fixed_test , report_binary_fold_test
from dotwiz import DotWiz
import pandas as pd

## metric_func = roc_auc_score 이외
def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

def optimal_cutoff(y_val , pred):
    fpr,tpr,thresholds = roc_curve(y_val , pred)
    best_thres = cutoff_youdens_j(fpr,tpr,thresholds)
    return best_thres

def precision(y,ypred):
    thres = optimal_cutoff(y,ypred)
    ypred_binary = np.where( ypred > thres , 1 , 0 )
    return precision_score(y , ypred_binary)

def recall(y,ypred):
    thres = optimal_cutoff(y,ypred)
    ypred_binary = np.where( ypred > thres , 1 , 0 )
    return recall_score(y , ypred_binary)


'''
1. 데이터 만들기
2. 모델 제너레이터 만들기 
3. 모델 생성 파라미터, 피팅 파라미터 만들기 
4. 성능평가 메트릭 정의하기 :  f : y , pred -> value
5. training_fixedTest 실행하기 
6. 유틸에 테스트 데이터에 대한 성능평가, 또는 저장 등을 하기 
'''
def Test_Binary_FixedTest(xtrain , ytrain , xtest  , nfold = 5 , verbose = False):
    # name list 
    name_list = ['xgb',
                 'lgb',]
                 #'cat',
                 #'rfc',
                 #'svm',
                 #'gpc',
                 #'lda',
                 #'qda']
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

    result_all = DotWiz()

    for name in name_list:
        print(name)
        fold_predict , fold_oof  , fold_metric , fold_model = tr.training_fixedTest('binary' , model_list[name] , param_list[name] , fitpm_list[name] ,  
                                                                                    metric_func , xtrain , ytrain , xtest , nfold , verbose ) 

        
        result_all[name] = data_fixed_test.FixedTestData(fold_predict , fold_oof  , fold_metric , fold_model)
       
    return result_all


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
        auc_score_list[name] = roc_auc_score(y, oof.mean(axis=1))
        print('done')
    print('Test_Classification_TestFold Compelte')    
    return result_list
 

def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print(' Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

if __name__ == '__main__':

    import result_report_binary
    from sklearn.metrics import roc_auc_score , precision_score, accuracy_score

    def df_test(): # 인덱스로 나눌때 iloc를 써야한다. loc쓰면 경우에 따라서 nan이 리턴되는 경우도 있었다. 
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
        '''
        #확인 완료 
        print(xtrain.isna().any())
        print(xtest.isna().any())
        print(ytrain.isna().any())
        print(ytest.isna().any())
        '''
        result = Test_Binary_FixedTest(xtrain , ytrain , xtest , 3,False)
        resultall = data_fixed_test.FixedTestResult(result)

        #print(result_report_binary.result2df_rowmodel_colfold_oof(result , ytrain , accuracy_score))
        #result = Test_Binary_TestFold(X,y,3,4)

    def np_test():
        data = load_breast_cancer()
        X = data.data
        y = data.target
        df = pd.DataFrame()
        xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = 0.2 )
        result = Test_Binary_FixedTest(xtrain , ytrain , xtest , 3,False)
        

        
        #print(result_report_binary.result2df_rowmodel_colfold_oof(result , ytrain , accuracy_score))
        #Test_Binary_TestFold(X, y, 5, 5)
        

    df_test()
    np_test()
    print('test pass')