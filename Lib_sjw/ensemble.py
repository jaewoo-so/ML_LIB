import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

#region use training_fixedTest
def esemble_binary(df_res , ytest_binary, lower , upper  , base_value):
    '''
    <ytest format>
    ytest = [1,1,0,0,1,0]

    <res format>
    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]

    df_res = result2df(res)
    ->
            xgb0       xgb1       xgb2       xgb3       xgb4
    0  20.492434  20.970526  20.399235  20.600239  20.491152
    1  21.900711  22.657307  22.999657  21.506317  22.652929
    2  22.384945  21.956985  22.209883  22.431749  21.806103

    '''
    # lower ~ upper 사이의 상관관계의 결과만 가져오기
    temp_cor = df_res.corr()

    temp = temp_cor.applymap(lambda x : x if x > lower else 0 )
    cor_filtered = temp.applymap(lambda x : 0 if x > upper else x )
    
    

    cor_selected = cor_filtered.stack()
    cor_selected = cor_selected[(cor_selected != 0) ] # s[(s!=0) & (s>0.5)]
    
    # 페어의 이름만 가져오기
    pair_pred = {}
    pair_auc = {}
    pair_pr = {}
    for i in range(cor_selected.index.shape[0]):
        pair = cor_selected.index[i]
    
        df_pair = df_res[list(pair)] # df_res는 원본 데이터
        avg_val = df_pair.mean(axis = 1).values
        pair_pred[pair[0]+pair[1]] = avg_val 

        # 이제 원래 레이블 ytest와 비교해본다. 
        pred_norm = MinMaxScaler().fit_transform(avg_val.reshape(-1, 1))  # [0,1]노멀라이즈
        temp1 = pair[0] + pair[1]
        temp2 = roc_auc_score(ytest_binary , pred_norm) 
        pair_auc[pair[0]+pair[1]] = roc_auc_score(ytest_binary , pred_norm) 
        pair_pr[pair[0]+pair[1]] = average_precision_score( ytest_binary , pred_norm)

    pair_auc = sorted(pair_auc.items(), key=lambda x: x[1], reverse=True)
    pair_pr  = sorted(pair_pr.items(), key=lambda x: x[1] , reverse = True)

    return pair_pred, pair_auc , pair_pr
    


def result2df(res_all):
    '''
    res_all format
    fold_predict , fold_oof , fold_metric , fold_models = tr.training_fixedTest( )
    res_all[name] = [ fold_predict , fold_oof , fold_metric , fold_models ]
    '''
    res = {}
    for model_name in res_all:
        pred_res = res_all[model_name][0]
        for i ,nfold in enumerate(pred_res):
            res[model_name+str(i)] = pred_res[nfold]
    df_res = pd.DataFrame(res)
    return df_res


def find_max_score(res_all , ytest_binary, verbose = False):
    single_res_auc = []
    single_res_pr = []
    for model_name in res_all:
        pred_res = res_all[model_name][0]
        print(model_name)
        for nfold in pred_res:
            binary = ytest_binary
            pred_norm = MinMaxScaler().fit_transform(pred_res[nfold].reshape(-1,1)) 
            if verbose : print('{} : {:.4f}/{:.4f}'.format(nfold, roc_auc_score(binary , pred_norm) , average_precision_score( binary , pred_norm)))
            single_res_auc.append( roc_auc_score(binary , pred_norm)  )
            single_res_pr.append( average_precision_score( binary , pred_norm) )
        
        
    print('auc max = {} , pr max = {}'.format(np.array(single_res_auc).max(), np.array(single_res_pr).max()))
    
if __name__ == '__main__':
    import sys
    sys.path.append(r'E:\01_PProj\ML_LIB')

    import Lib_sjw.training as tr
    import Lib_sjw.model_interface as mi
    import Lib_sjw.model_parmas as mp
    import Lib_sjw.evaluator as ev
    import Lib_sjw.classification_util as cu
    import Lib_sjw.ensemble as es




    from sklearn.datasets import load_boston , load_iris , load_breast_cancer
    from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score , average_precision_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler 
    import numpy as np
    import pandas as pd
    import os

    import seaborn as sns
    from matplotlib import pyplot as plt
    import pickle

    from collections import OrderedDict

    data_folder = r'E:/00_proj/genestart/code_final/inputs'
    dftr = pd.read_csv( r'E:/00_proj/genestart/inputs/dataset_mean_v01_norm.BMI2binning_v03.csv')
    dfte = pd.read_csv( r'E:/00_proj/genestart/inputs/testset_mean_v01_norm.BMI2binning_v05.csv')

    Xtrain = dftr.iloc[:, 1:-2].values
    ytrain = dftr.iloc[:,-2].values

    Xtest = dfte.iloc[:, 1:-2].values
    ytest = dfte.iloc[:,-2].values
    ytest_binary = np.where(ytest > 25, 1, 0)
    
    res_all = None
    with open(r'E:/00_proj/genestart/code_final/res_01_baseline.pickle' , 'rb') as f:
        res_all = pickle.load(f)

    df_v1 = es.result2df(res_all)
    pair_pred_v1, temp1, temp2 = es.esemble_binary(df_v1, ytest_binary, 0.97, 0.997, 0) ## 작업 : 결과가 이상하게 나오는데 이것을 확인 해야 한다. 
    print('done')