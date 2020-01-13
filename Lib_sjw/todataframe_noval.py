import pandas as pd
from sklearn.metrics import mean_squared_error , roc_auc_score , precision_score , accuracy_score , average_precision_score

def result2df_rowmodel_colscore_noval(res_all, ytest , score_func):
    '''
    res_pred , model = tr.training_fixedTest_noVal
    '''
    models = []
    scores = []
    for model_name in res_all:
            pred_res = res_all[model_name][0]
            score = score_func(ytest , pred_res)
            models.append(model_name)
            scores.append(score)
    df_res = pd.DataFrame()
    df_res['model']  = models
    df_res['score'] = scores
    return df_res

def result2df_rowmodel_colfold_noval(res_all , ytest , score_func):
    '''
    res_all format : 
    test_fold_index , oof, model_list = tr.training_Testfold_noVal
    result_list[name] = [test_fold_index , oof, model_list]
    '''
    model_name_first = list(res_all.items())[0][0]
    fold_names  = list(res_all[model_name_first][0].keys())
    result = pd.DataFrame( columns = ['model'] + fold_names)
    model_name_list = []

    for model_name in res_all:
        model_res = res_all[model_name]
        test_idx = model_res[0]
        score = {}
        score['model'] = model_name
        for i , (nfold , idx ) in enumerate(test_idx.items()):    
            pred = model_res[1][idx]
            score[str(nfold)] = score_func(ytest, pred)
        result = result.append(score, ignore_index=True)
    result['mean'] = result.mean(axis = 1)
    return result