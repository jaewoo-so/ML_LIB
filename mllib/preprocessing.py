import pandas as pd
import numpy as np

def Normalize(df): # test complete
    df_norm = ((df-df.min())/(df.max()-df.min()))
    df_norm2 = (df_norm-df_norm.mean())/df_norm.std()
    return df_norm2


def Binary_to01(df_binary): # test complete
    for scol in df_binary:
        value_count = df_binary[scol].value_counts()
        binary1 = value_count.index[0]
        binary2 = value_count.index[1]

        if (binary1 != 0) or (binary2 != 1):
            df_binary[scol] = np.where(df_binary[scol] == binary1 , 0 , 1)
    return df_binary