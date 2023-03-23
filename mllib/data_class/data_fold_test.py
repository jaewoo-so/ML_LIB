from dataclasses import dataclass
import pandas as pd
import mllib
from dotwiz import DotWiz

@dataclass
class FoldTestData:
    '''
    Only use training data 
    '''
    test_fold_index: pd.DataFrame
    oof_pred: pd.DataFrame
    fold_model: mllib.model_interface