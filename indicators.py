import pandas as pd
import numpy as np

def RSI(df, n):
    '''
    Adds Relative Strenght Index indicator as a column of a yahoo finance formated DataFrame
    Made with help of: https://www.learnpythonwithrune.org/pandas-calculate-the-relative-strength-index-rsi-on-a-stock/
    '''

    diff = df['close'].diff()
    up = diff.clip(lower=0)
    down = -1*diff.clip(upper=0)

    mv_up = up.rolling(n).mean()
    mv_down = down.rolling(n).mean()
    rs = mv_up/mv_down
    
    df['RSI-'+str(n)] = 100 - (100/(1+rs))



def ATR(df, n):
    '''
    Adds Average True Range indicator as a column of a yahoo finance formated DataFrame
    Made with help of: https://towardsdatascience.com/building-a-comprehensive-set-of-technical-indicators-in-python-for-quantitative-trading-8d98751b5fb
    '''
    prev_close = df['close'].shift(1)
    TR = np.maximum(df['high']-df['low'], np.maximum(abs(df['high']-prev_close), abs(prev_close-df['low'])))
    df['ATR-'+str(n)] = TR.rolling(window=n).mean()


def MOM(df, n):
    '''
    Adds Momentum indicator as a column of a yahoo finance formated DataFrame
    '''

    df['MOM-'+str(n)] = df['close'].diff(n)



def MFI(df, n):
    '''
    Adds Money Flow Index indicator as a column of a yahoo finance formated DataFrame
    '''

    TP = (df['high'] + df['low'] + df['close'])/3
    MF = TP * df['volume']
    PF, NF = [], []

    for i in range(1, len(TP)):
        if TP[i]>TP[i-1]:
            PF.append(MF[i-1])
            NF.append(0)
        elif TP[i]<TP[i-1]:
            PF.append(0)
            NF.append(MF[i-1])
        else:
            PF.append(0)
            NF.append(0)

    MFR = pd.Series(PF).rolling(n).sum() / pd.Series(NF).rolling(n).sum()
    MFR_list = [np.nan] + list(100 - (100/(1+MFR)))
    df['MFI-'+str(n)] = MFR_list


def OBV(df, n):
    '''
    Adds On Balance Volume indicator as a column of a yahoo finance formated DataFrame
    '''

    OBV_list = [0]
    for i in range(1, len(df['close'])):
        if df['close'][i] > df['close'][i-1]:
            OBV_list.append(OBV_list[-1]+df['volume'][i])
        elif df['close'][i] < df['close'][i-1]:
            OBV_list.append(OBV_list[-1]-df['volume'][i])
        else:
            OBV_list.append(OBV_list[-1])


    df['OBV-'+str(n)] = list(pd.Series(OBV_list).ewm(com=n).mean())