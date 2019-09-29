from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.metrics import precision_recall_curve , average_precision_score
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt


#region Helpr
def aucpr(y_true, y_scores):
    aucpr = average_precision_score(y_true, y_scores)
    return aucpr
#endregion

#region Pack
def create_mean_pred_df(result , axis = 0):
    res = {}
    #correlation
    for model_name in res:
        pred_res = res[model_name][0]
        pred_mean = np.array(list(pred_res.values())).mean(axis = axis)
        res[model_name] = pred_mean
    return pd.DataFrame(res)

def mean_score_to_df(model_score):
    return pd.DataFrame.from_dict(model_score)
    
def model_each_score_to_flat_df(model_each_score):
    for k , v in model_each_score.items():
        model_each_score[k] = v.flatten()
    return pd.DataFrame.from_dict(model_each_score)
    
def mean_score_each_valset(y , res, score_func = roc_auc_score):
    '''
    res : result of Test_Binary_TestFold
    -> model_score , model_each_score
    '''
    model_score      = OrderedDict()
    model_each_score = OrderedDict()

    for model_name in res:
        model_res = res[model_name]

        prediction   = model_res[0]
       
        # 각 테스트 셋의 평균 저장변수 
        test_score_list = []

        for k , idx in test_idx.items():
            test_fold_pred = prediction[idx,:] # 각각의 테스트 셋에대한 validation갯수만큼의 예측 값이 들어있다. 

            # 한개의 테스트 세트에 대해서 validation 갯수 만큼의 스코어를 저장
            score_each_val = []

            for val_fold in range(test_fold_pred.shape[1]):
                pred_of_valfold = test_fold_pred[:,val_fold] # validation셋 하나의 예측값만 가져온 것 

                #여기서 하나의 validation에 대한 예측 값을 가지고 스코어링 
                val_score = score_func(y[idx] , pred_of_valfold)
                score_each_val.append(val_score)

            test_fold_score = np.array(score_each_val) # 한개의 테스트 세트에 대한 각 validation에 대해 예측값들
            test_score_list.append(test_fold_score)

        model_all_score = np.array(test_score_list)
        print( f'{model_name} mean : {model_all_score.mean()*100:0.2f}')

        # save
        model_score[model_name] = [model_all_score.mean()]
        model_each_score[model_name] = model_all_score
    return model_score , model_each_score

def mean_score_each_testset(y , res, score_func = roc_auc_score):
    '''
    res : result of Test_Binary_TestFold
    -> model_score , model_each_score
    '''
    model_score      = OrderedDict()
    model_each_score = OrderedDict()

    for model_name in res:
        model_res = res[model_name]

        test_idx   = model_res[0]
        prediction = model_res[1]
        model_file = model_res[2]

        # 각 테스트 셋의 평균 저장변수 
        test_score_list = []

        for k , idx in test_idx.items():
            test_fold_pred = prediction[idx,:] # 각각의 테스트 셋에대한 validation갯수만큼의 예측 값이 들어있다. 

            # 한개의 테스트 세트에 대해서 validation 갯수 만큼의 스코어를 저장
            score_each_val = []

            for val_fold in range(test_fold_pred.shape[1]):
                pred_of_valfold = test_fold_pred[:,val_fold] # validation셋 하나의 예측값만 가져온 것 

                #여기서 하나의 validation에 대한 예측 값을 가지고 스코어링 
                val_score = score_func(y[idx] , pred_of_valfold)
                score_each_val.append(val_score)

            test_fold_score = np.array(score_each_val) # 한개의 테스트 세트에 대한 각 validation에 대해 예측값들
            test_score_list.append(test_fold_score)

        model_all_score = np.array(test_score_list)
        print( f'{model_name} mean : {model_all_score.mean()*100:0.2f}')

        # save
        model_score[model_name] = [model_all_score.mean()]
        model_each_score[model_name] = model_all_score
    return model_score , model_each_score


#endregion

def Create_ttest_data(df_res , model_count ):
    '''
    df_res -> n_fold X models 
          |model1 | model2
    fold1 |0.2    | 0.3
    fold2 |0.4    | 0.5
    '''
    init_data = np.ones((model_count,model_count))
    idx_j , idx_i = np.tril_indices(model_count)
    for j , i in zip(idx_j, idx_i):
        init_data[j,i] = ttest_ind( df_res.iloc[:,i].values , df_res.iloc[:,j].values)[1]
    return init_data

def t_test_models(src, names, cmap=plt.cm.Greys, title='', fig_size=(6, 6), save_path=None):
    '''
          |model1 | model2
    fold1 |0.2    | 0.3
    fold2 |0.4    | 0.5

    ttest_data  = Create_ttest_data(df_ttest,df_ttest.shape[1])
    t_test_models(ttest_data , model_names, plt.cm.Blues , 'T-test ', save_path='./result/t-test_table.png')
    '''
    fig, ax = plt.subplots( 1,1, figsize = fig_size)
    im = ax.imshow(src, interpolation='nearest', cmap=plt.cm.Blues)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(src.shape[1]),
           yticks=np.arange(src.shape[0]),
           ylabel='Model Names',
           xlabel='Model Names',
          xticklabels=names, yticklabels=names)
    ax.set_title(title , fontsize = 23)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.4f' 
    thresh = src.max() / 2.
    
    idx_j , idx_i = np.tril_indices(len(names))

    for j , i in zip(idx_j, idx_i):
        if j == i:
            continue
        ax.text(i, j, format(src[j, i], fmt), ha="center", va="center", color="white" if src[j, i] > thresh else "black")
            
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax)
    
    fig.tight_layout()
    if save_path != None:
        plt.savefig(save_path)
    plt.show()

# regression
# regression에서 aucpr구하기
from sklearn.preprocessing import MinMaxScaler

def regression_aucpr(res , ytest , threshold):
    '''
    res = Test_Regression(Xtrain,ytrain,Xtest , 10)
    '''
    for model_name in res:
        pred_res = res[model_name][0]
        print(model_name)
        for nfold in pred_res:
            binary = np.where( ytest > threshold , 1 , 0 )
            pred_norm = MinMaxScaler().fit_transform(pred_res[nfold].reshape(-1,1)) 
            print('{} : {:.4f}'.format(nfold,an.aucpr( binary , pred_norm)))
        

#endregion
# 결과 로딩
import pickle 
import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score
if __name__ == '__main__':
    with open(r'E:\\00_proj\\CDSS\\code_final_v02/result/result_list_on_normalized_dataset1234.pickle' , 'rb') as f:
        result_list = pickle.load(f)

    # 사용 데이터 세트 로딩
    df_list = []
    for i in range(4):
        res = pd.read_csv(r'E:\\00_proj\\CDSS\\code_final_v02/input/data_norm_imputed{}.csv'.format(i) )
        df_list.append(res)
    print('done')
    score_list = []
    each_score_list = []
    for i , result in enumerate(result_list):
        print(i)
        model_score , model_each_score = mean_score_each_testset( df_list[i].iloc[:,-1] , result , roc_auc_score)
        score_list.append(model_score)
        each_score_list.append(model_each_score)


