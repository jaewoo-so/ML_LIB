from abc import *
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore") 
'''
make
fit 
predict
predict_prob
param
 공통으로 하기 위해서 
 make는 쓰던 안쓰던 validation 값도 넣자
'''
# abstract class for model
class myModel(metaclass=ABCMeta):

    @abstractmethod
    def make(self , make_params , rand_num):
        pass

    @abstractmethod
    def fit(self , xtrain , ytrain , fit_params ):
        pass

    @abstractmethod
    def predict(self , xtrain ):
        pass

    @abstractmethod
    def predict_proba(self , xtrain ):
        pass

#xgboost
from xgboost import XGBClassifier , XGBRegressor
class myXGBClassifier(myModel):
    def make(self , make_params , rand_num = 7 ):
        self.model = XGBClassifier(**make_params , random_state = rand_num )
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model.fit( xtrain , ytrain , verbose = False , **fit_params )
        else:
            self.model.fit( xtrain , ytrain , eval_set=[(xtest,ytest)] , verbose = False ,**fit_params )
        
    def predict(self , xs ):
        return self.model.predict(xs) 

    def predict_proba(self , xs ):
        return self.model.predict_proba(xs)

class myXGBBinary(myModel):
    def make(self , make_params , rand_num = 7 ):
        self.model = XGBClassifier(**make_params , random_state = rand_num )
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model.fit( xtrain , ytrain , verbose = False , **fit_params )
        else:
            self.model.fit( xtrain , ytrain , eval_set=[(xtest,ytest)] , verbose = False ,**fit_params )
        
    def predict(self , xs ):
        return self.model.predict(xs) 

    def predict_proba(self, xs):
       
        return self.model.predict_proba(xs)[:,1]

class myXGBRegressor(myModel):
    def make(self , make_params , rand_num = 7 ):
        self.model = XGBRegressor(**make_params , random_state = rand_num )
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model.fit( xtrain , ytrain , verbose = False , **fit_params )
        else:
            self.model.fit( xtrain , ytrain , eval_set=[(xtest,ytest)] , verbose = False ,**fit_params )
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
                    
    def predict_proba(self , xs ):
        return self.model.predict(xs)


#lightgbm
'''
early_stopping_rounds 옵션을 fit함수에서 넣어준다.
'''
from lightgbm import LGBMClassifier , LGBMRegressor
import lightgbm as lgb

class myLGBMClassifier:
    def make(self , params , rand_num = 7 ):
        self.params = params
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        params_all = {**self.params, **fit_params}
        lgb_train = lgb.Dataset(xtrain, ytrain, params={'verbose': -1}, free_raw_data=False)
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model = lgb.train(params_all, lgb_train)
        else:
            lgb_eval = lgb.Dataset(xtest, ytest, params={'verbose': -1},free_raw_data=False)
            self.model = lgb.train(params_all, lgb_train, valid_sets=lgb_eval, verbose_eval=False)
        
    def predict(self , xs ):
        return self.model.predict(xs) 
        
    def predict_proba(self , xs ):
        #return self.model.predict_proba(xs)[:,1] # sklearn version
        return self.model.predict(xs)

class myLGBMBinary:
    def make(self , params , rand_num = 7 ):
        self.params = params
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        params_all = {**self.params, **fit_params}
        lgb_train = lgb.Dataset(xtrain, ytrain, params={'verbose': -1}, free_raw_data=False)
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model = lgb.train(params_all, lgb_train)
        else:
            lgb_eval = lgb.Dataset(xtest, ytest, params={'verbose': -1},free_raw_data=False)
            self.model = lgb.train(params_all, lgb_train, valid_sets=lgb_eval, verbose_eval=False)
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
        
    def predict_proba(self, xs):
        return self.model.predict(xs)

class myLGBMRegressor:
    def make(self , params , rand_num = 7 ):
        #self.model =  LGBMRegressor(**params , random_state = rand_num ) # sklearn version
        self.params = params
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        params_all = {**self.params, **fit_params}
        lgb_train = lgb.Dataset(xtrain, ytrain, params={'verbose': -1}, free_raw_data=False)
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model = lgb.train(params_all, lgb_train)
        else:
            lgb_eval = lgb.Dataset(xtest, ytest, params={'verbose': -1},free_raw_data=False)
            self.model = lgb.train(params_all, lgb_train, valid_sets=lgb_eval, verbose_eval=False)
        ''' sklearn version
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model.fit( xtrain , ytrain , verbose = -1 , **fit_params )
        else:
            self.model.fit( xtrain , ytrain , eval_set=[(xtest,ytest)] , verbose = -1 , free_raw_data = False,**fit_params )
        '''
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
        
    def predict_proba(self , xs ):
        return self.model.predict(xs)
    
#catboost
'''
early_stopping_rounds 옵션을 모델 만들때 넣어준다. 
'''
from catboost import CatBoostClassifier , CatBoostRegressor
class myCatBoostClassifier:
    def make(self , params , rand_num = 7 ):
        self.model =  CatBoostClassifier(**params , random_state = rand_num )
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model.fit( xtrain , ytrain , verbose = False , **fit_params )
        else:
            self.model.fit( xtrain , ytrain , eval_set=[(xtest,ytest)] , verbose = False ,**fit_params )
        
    def predict(self , xs ):
        return self.model.predict(xs) 
        
    def predict_proba(self , xs ):
        return self.model.predict_proba(xs)

class myCatBoostBinary:
    def make(self , params , rand_num = 7 ):
        self.model =  CatBoostClassifier(**params , random_state = rand_num )
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model.fit( xtrain , ytrain , verbose = False , **fit_params )
        else:
            self.model.fit( xtrain , ytrain , eval_set=[(xtest,ytest)] , verbose = False ,**fit_params )
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
        
    def predict_proba(self , xs ):
        return self.model.predict_proba(xs)[:,1]

class myCatBoostRegressor:
    def make(self , params , rand_num = 7 ):
        self.model =  CatBoostRegressor(**params , random_state = rand_num )
        return self

    def fit(self ,  xtrain , ytrain , xtest =None, ytest =None , fit_params = {}):
        if type(xtest) == type(None) or type(ytest) == type(None) :
            self.model.fit( xtrain , ytrain , verbose = False , **fit_params )
        else:
            self.model.fit( xtrain , ytrain , eval_set=[(xtest,ytest)] , verbose = False ,**fit_params )
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
        
    def predict_proba(self , xs ):
        return self.model.predict(xs)

        
#random forest 
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
class myRandomForestClassifier(myModel):
    def make(self , make_params  ):
        self.model = RandomForestClassifier(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain , **fit_params)
        
    def predict(self , xs ):
        return self.model.predict(xs)

    def predict_proba(self , xtrain ):
        return self.model.predict_proba(xtrain )

class myRandomForestBinary(myModel):
    def make(self , make_params  ):
        self.model = RandomForestClassifier(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain , **fit_params)
        
    def predict(self , xs ):
        return self.model.predict(xs)

    def predict_proba(self , xtrain ):
        return self.model.predict_proba(xtrain )[:,1]

class myRandomForestRegressor(myModel):
    def make(self , make_params ):
        self.model = RandomForestRegressor(**make_params)
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
                    
    def predict_proba(self , xs ):
        return self.model.predict(xs)

#svm
from sklearn.svm import SVC , SVR
class mySVMClassifier(myModel):
    def make(self , make_params  ):
        self.model = SVC(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)
        
    def predict(self , xs ):
        return self.model.predict(xs)

    def predict_proba(self , xtrain ):
        return self.model.predict_proba(xtrain)
        
class mySVMBinary(myModel):
    def make(self , make_params  ):
        self.model = SVC(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)
        
    def predict(self , xs ):
        return self.model.predict(xs)

    def predict_proba(self , xtrain ):
        return self.model.predict_proba(xtrain )[:,1]

class mySVMRegressor(myModel):
    def make(self , make_params  ):
        self.model = SVR(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
                    
    def predict_proba(self , xs ):
        return self.model.predict(xs)

#
from sklearn.linear_model import LinearRegression , LogisticRegression
class myLinearRegressionBinary(myModel):
    def make(self , make_params ):
        self.model = LogisticRegression(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain , **fit_params)

    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
                    
    def predict_proba(self , xs ):
        return self.model.predict(xs)

class myLinearRegressionRegressor(myModel):
    def make(self , make_params ):
        self.model = LinearRegression(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
                    
    def predict_proba(self , xs ):
        return self.model.predict(xs)

#elasticnet
from sklearn.linear_model import ElasticNet
class myElasticNetBinary(myModel):
    def make(self , make_params ):
        self.model = ElasticNet(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain , **fit_params)

    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
                    
    def predict_proba(self , xs ):
        return self.model.predict(xs)

class myElasticNetRegressor(myModel):
    def make(self , make_params ):
        self.model = ElasticNet(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
                    
    def predict_proba(self , xs ):
        return self.model.predict(xs)

#GBM
from sklearn.gaussian_process import GaussianProcessClassifier , GaussianProcessRegressor
class myGPClassifier(myModel):
    def make(self , make_params  ):
        self.model = GaussianProcessClassifier(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)

    def predict(self , xs , threshold = 0.5):
        return self.model.predict
                    
    def predict_proba(self , xs ):
        return self.model.predict_proba(xs)

class myGPBinary(myModel):
    def make(self , make_params  ):
        self.model = GaussianProcessClassifier(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)

    def predict(self , xs , threshold = 0.5):
        return self.model.predict
                    
    def predict_proba(self , xs ):
        return self.model.predict_proba(xs)[:,1]

class myGPRegressor(myModel):
    def make(self , make_params ):
        self.model = GaussianProcessRegressor(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        self.model.fit(xtrain , ytrain  , **fit_params)
        
    def predict(self , xs , threshold = 0.5):
        return np.where(self.model.predict(xs) > threshold , 1 , 0)
                    
    def predict_proba(self , xs ):
        return self.model.predict(xs)

## classifier 

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class myLDAClassifier(myModel):
    def make(self , make_params ):
        self.model = LinearDiscriminantAnalysis(**make_params )
        return self

    def fit(self, xtrain, ytrain, xtest=None, ytest=None, fit_params={}):
        if type(xtrain) == pd.core.frame.DataFrame:
            self.model.fit(xtrain.astype('float32') , ytrain.astype('float32')  , **fit_params)
        else:
            self.model.fit(xtrain , ytrain  , **fit_params)
        

    def predict(self, xs, threshold=0.5):
        if type(xs) == pd.core.frame.DataFrame:
            return self.model.predict(xs.astype('float32'))
        else:
            return self.model.predict(xs)
                    
    def predict_proba(self, xs):
        if type(xs) == pd.core.frame.DataFrame:
            return self.model.predict_proba(xs.astype('float32'))
        else:
            return self.model.predict_proba(xs)

class myLDABinary(myModel):
    def make(self , make_params ):
        self.model = LinearDiscriminantAnalysis(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        if type(xtrain) == pd.core.frame.DataFrame:
            self.model.fit(xtrain.astype('float32') , ytrain.astype('float32')  , **fit_params)
        else:
            self.model.fit(xtrain , ytrain  , **fit_params)

    def predict(self , xs , threshold = 0.5):
        if type(xs) == pd.core.frame.DataFrame:
            return self.model.predict(xs.astype('float32'))
        else:
            return self.model.predict(xs)
                    
    def predict_proba(self, xs):
        if type(xs) == pd.core.frame.DataFrame:
            return self.model.predict_proba(xs.astype('float32'))[:,1]
        else:
            return self.model.predict_proba(xs)[:,1]

#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class myQDAClassifier(myModel):
    def make(self , make_params ):
        self.model = QuadraticDiscriminantAnalysis(**make_params )
        return self

    def fit(self, xtrain, ytrain, xtest=None, ytest=None, fit_params={}):
        if type(xtrain) == pd.core.frame.DataFrame:
            self.model.fit(xtrain.astype('float32') , ytrain.astype('float32')  , **fit_params)
        else:
            self.model.fit(xtrain , ytrain  , **fit_params)
        

    def predict(self, xs, threshold=0.5):
        if type(xs) == pd.core.frame.DataFrame:
            return self.model.predict(xs.astype('float32'))
        else:
            return self.model.predict(xs)
                    
    def predict_proba(self, xs):
        if type(xs) == pd.core.frame.DataFrame:
            return self.model.predict_proba(xs.astype('float32'))
        else:
            return self.model.predict_proba(xs)

class myQDABinary(myModel):
    def make(self , make_params ):
        self.model = QuadraticDiscriminantAnalysis(**make_params )
        return self

    def fit(self , xtrain , ytrain , xtest =None, ytest =None , fit_params = {} ):
        if type(xtrain) == pd.core.frame.DataFrame:
            self.model.fit(xtrain.astype('float32') , ytrain.astype('float32')  , **fit_params)
        else:
            self.model.fit(xtrain , ytrain  , **fit_params)

    def predict(self , xs , threshold = 0.5):
        if type(xs) == pd.core.frame.DataFrame:
            return self.model.predict(xs.astype('float32'))
        else:
            return self.model.predict(xs)
                    
    def predict_proba(self, xs):
        if type(xs) == pd.core.frame.DataFrame:
            return self.model.predict_proba(xs.astype('float32'))[:,1]
        else:
            return self.model.predict_proba(xs)[:,1]

