
# 멀티 클래스일 경우는 num_class 설정 해줘야 한다. 
def param_xgb( mode , num_class = 2 , use_gpu = False, random_num = 7):
    '''
    멀티 클래스일 경우는 num_class 설정 해줘야 한다. 
    '''
    params_xgb = dict()
    
    params_xgb['booster']      = 'gbtree'
    params_xgb['verbosity']    = 0

    params_xgb['learning_rate']    = 0.02
    params_xgb['bagging_fraction'] = 0.8
    params_xgb['feature_fraction'] = 0.8
    params_xgb['lambda_l1']    = 0.3
    params_xgb['lambda_l2']    = 0.4
    params_xgb['max_depth']    = 9
 
    # gpu , cpu
    if use_gpu:
        params_xgb['tree_method'] = 'gpu_hist'
        params_xgb['predictor']    = 'gpu_predictor'
    else:
        params_xgb['tree_method'] = 'hist'
        params_xgb['predictor']    = 'cpu_predictor'

        if mode == 'regression':
            params_xgb['objective']    = 'reg:squarederror'
        elif mode == 'binary':
            params_xgb['objective']    = 'binary:logistic'
        elif mode == 'classification':
            params_xgb['objective']    = 'multi:softmax'
            params_xgb['num_class']    = num_class

    return params_xgb

def param_lgbm(mode , num_class = 2, use_gpu = False, is_unbalance = False, random_num = 7):
    '''
    멀티 클래스일 경우는 num_class 설정 해줘야 한다. 
    '''
    params_lgb = {'colsample_bytree': 0.1, # feature_fraction 
        'max_depth': 15,
        'min_child_weight': 0.0,
        'min_data_in_leaf': 0,
        'min_split_gain': 0.0,
        'num_leaves': 100}

    params_lgb['task'] = 'train'
    params_lgb['boosting_type'] = 'gbdt'
    params_lgb['is_unbalance'] = is_unbalance
    params_lgb["learning_rate"] = 0.02
    params_lgb["num_iterations"] = 2000
    params_lgb['seed']=random_num
    params_lgb['bagging_seed']=random_num
    params_lgb['bagging_fraction'] = 0.8
    params_lgb['bagging_fraction'] = 0.8
    params_lgb['feature_fraction_seed'] = random_num
    params_lgb['lambda_l1'] = 0.3
    params_lgb['lambda_l2'] = 0.4
    params_lgb['verbose'] = 0
    
    # gpu , cpu
    if use_gpu:
        params_lgb['device'] = 'gpu'
    else:
        params_lgb['device'] = 'cpu'

    # metric , objective
    if mode == 'regression':
        params_lgb['objective'] = 'regression'
        params_lgb['metric'] = 'l2_root'
    elif mode == 'binary':
        params_lgb['objective'] = 'binary'
        params_lgb['metric'] = 'auc'
    elif mode == 'classification':
        params_lgb['objective'] = 'multiclass'
        params_lgb['metric'] = 'multi_logloss'
        params_lgb['num_class'] = num_class

    return params_lgb

def param_cat(mode , use_gpu = False, is_unbalance = False, random_num = 7):
    params_cat = {}
    params_cat['iterations']            = 2000
    params_cat['learning_rate']         = 0.02  
    params_cat['od_type']               =  'Iter'
    params_cat['early_stopping_rounds'] = 12
  
     # gpu , cpu
    if use_gpu:
        params_cat['task_type'] = "GPU"
    else:
        params_cat['task_type'] = "CPU"

    # metric , objective
    # https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    if mode == 'regression':
        params_cat['objective'] = 'RMSE'
        params_cat['eval_metric'] = 'RMSE'
    elif mode == 'binary':
        params_cat['objective'] = 'CrossEntropy'
        params_cat['eval_metric'] = 'Accuracy'
    elif mode == 'classification':
        params_cat['objective'] = 'MultiClass'
        params_cat['eval_metric'] = 'MultiClass'
    return params_cat


def param_rf(mode , njobs = -1, random_num = 7):
    params_rf = dict()
    params_rf['n_estimators'    ] = 80
    params_rf['max_depth'       ] = 15
    params_rf['n_jobs'          ] = njobs
    params_rf['min_samples_leaf'] = 1
    params_rf['max_features'    ] = 'sqrt' # None
    params_rf['random_state'    ] = random_num
    return params_rf


def param_svm(mode , njobs = -1, random_num = 7):
    params_svm = dict()
    params_svm['C'           ] = 0.9
    params_svm['kernel'      ] = 'rbf' # 'linear', 'poly'
    if mode =='classification':
         params_svm['probability'] = True
    return params_svm
    
def param_elst(mode , njobs = -1, random_num = 7):
    params_elst = dict()
    params_elst['alpha'       ] = 0.7
    params_elst['l1_ratio'    ] = 0.3
    params_elst['random_state'] = random_num
    return params_elst

from sklearn.gaussian_process.kernels import *
def param_gpc(mode , njobs = -1 , random_num = 7):
    
    '''
    kernel list
    gaussian_process.kernels.CompoundKernel (kernels)	Kernel which is composed of a set of other kernels .
    gaussian_process.kernels.ConstantKernel ([…]) Constant kernel .
    gaussian_process.kernels.DotProduct ([…]) Dot - Product kernel .
    gaussian_process.kernels.ExpSineSquared ([…]) Exp - Sine -Squared kernel .
    gaussian_process.kernels.Exponentiation (…) Exponentiate kernel by given exponent .
    gaussian_process.kernels.Hyperparameter A kernel hyperparameter’s specification in form of a namedtuple.
    gaussian_process.kernels.Kernel Base class for all kernels .
    gaussian_process.kernels.Matern ([…]) Matern kernel .
    gaussian_process.kernels.PairwiseKernel ([…]) Wrapper for kernels in sklearn .metrics.pairwise.
    gaussian_process.kernels.Product ( k1 , k2 )	Product-kernel k1 * k2 of two kernels k1 and k2 .
    gaussian_process.kernels.RBF ([ length_scale, …]) Radial -basis function kernel (aka squared-exponential kernel).
    gaussian_process.kernels.RationalQuadratic([…]) Rational Quadratic kernel .
    gaussian_process.kernels.Sum ( k1 , k2 )	Sum-kernel k1 + k2 of two kernels k1 and k2 .
    gaussian_process.kernels.WhiteKernel ([…]) White kernel .
    '''
    param_gpc = dict()
    param_gpc['kernel'      ] = RBF(1.0)
    if mode      == 'regression'  :
        param_gpc['alpha'       ] = 1e-10
    if mode == 'classification':
        param_gpc[ 'n_jobs'       ] = njobs
    param_gpc[ 'optimizer'    ] = 'fmin_l_bfgs_b'
    param_gpc[ 'random_state' ] = random_num
    return param_gpc

def param_lda(solver = 'svd' , n_components = None , priors = None):
    param_lda = dict()
    param_lda['solver'] = solver #'svd' 'lsqr , eigen
    param_lda['n_components'] = n_components # 차원축소 갯수
    param_lda['priors']      = priors
    return param_lda

def param_qda(priors = None):
    param_qda = dict()
    param_qda['priors'] = priors
    return param_qda