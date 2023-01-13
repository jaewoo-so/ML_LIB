import pandas as pd
from abc import *

class Report(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def oof_score_each_model(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def oof_pred_each_model(self) -> pd.DataFrame:
        pass

