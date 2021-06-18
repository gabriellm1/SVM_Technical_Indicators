import numpy as np
import pandas as pd
import math
import yahoofinancials as yf


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Para fazer o backtesting
from backtesting import TradingSystem, MarketData, Order, Strategy, Event, evaluateHist


def getData(tickers,start_date,end_date,backtesting_size):
    '''
    Pulls data from yahoo finance based on ticker and saves on 
    two different files, one for training a model and other for backtesting
    '''

    for ticker in tickers:
        data = yf.YahooFinancials(ticker).get_historical_price_data(start_date, end_date, 'daily')

        # Ler os dados do JSON
        raw = pd.DataFrame(data[ticker]['prices']).dropna()
        # Converter a data para o tipo correto datetime
        raw['formatted_date'] = pd.to_datetime(raw['formatted_date'])
        # Indica a data como o índice de cada linha
        raw = raw.set_index('formatted_date')
        # Removendo as colunas que não interessam
        df = raw.iloc[:,1:]

        # Acertando a ordem das colunas
        df = df.reindex(columns=['open', 'high', 'low', 'close', 'adjclose', 'volume'])
        # Salvando o CSV
        df.iloc[:-backtesting_size,:].to_csv('./data/{}-train.csv'.format(ticker))
        df.iloc[-backtesting_size:,:].to_csv('./data/{}-back.csv'.format(ticker))


def formatDF(path,indicators):
    '''
    Read a DataFrame from yahoo finance and adds indicators and signals for training
    '''

    df = pd.read_csv(path)
    df = df.set_index('formatted_date')

    df['L0'] = df['close'].pct_change()

    def signal(x):
        if pd.isnull(x):
            return x
        elif x > 0:
            return 1
        elif x < 0:
            return  -1
        else:
            return 0

    df['sig'] = pd.Categorical(df['L0'].apply(signal)).shift(-1)

    for indicator in indicators:
        indicator[0](df, indicator[1])

    # df = df.shift(-1)
    df = df.dropna()

    return df

def train_test_model(df):
    '''
    Trains a SVM classifier to predict good buy and sell signals
    '''

    scaler = preprocessing.MaxAbsScaler()

    X = np.array(df.drop(['L0','sig','close','adjclose','volume','high','low','open'],1))
    X = scaler.fit_transform(X)
    Y = np.array(df['sig'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    model = svm.SVC(kernel='rbf',decision_function_shape='ovo')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)


    return model, scaler, report


class MLClassifier(Strategy):

    def __init__(self, model, lags, scaler,print_filled = False):
        
        self.model = model
        self.lags = lags
        self.print_filled = print_filled

        # Inicializando os dicionários
        self.prices = []
        self.highs = []
        self.lows = []

        self.obv = [] # Volume indicator

        self.TR = [] # True Range indicator

        self.TP = [] # True price indicator
        self.MF = [] # Money Flow indicator
        self.PF = [] # Positive Flow indicator
        self.NF = [] # Negative Flow indicator

        self.scaler = scaler
        
        self.side = 0
        
        self.indicator = []

    def calc_RSI(self, n):

        # Se a quantidade de preços observados for superior a janela
        if len(self.prices) >= n + 2:
            
            # Calculo do retorno no período (não esquecer de adicionar 1 ponto por causa do retorno)
            returns = pd.Series(self.prices[-n - 1:]).pct_change().dropna()
            
            # média de retornos positivos e negativos
            up = returns[returns > 0].sum()/n
            down = -returns[returns < 0].sum()/n
            
            # Calculo do RSI
            if down == 0:
                rsi = 100
            else:
                rsi = 100 - 100 / (1 + up/down)
            
            # Guarda o indicador para o gráfico
            # self.upperRSI.append(self.uband)
            # self.lowerRSI.append(self.lband)
            # self.RSI.append(rsi)

        return rsi



    def calc_ATR(self, n):
        
        value = 0

        # Se a quantidade de preços observados for superior a janela
        if len(self.prices) >= n + 2:

            value = sum(self.TR[-n:])/n

        return value

    def calc_OBV(self, n):

        # Se a quantidade de preços observados for superior a janela
        if len(self.prices) >= n + 1:

            # Média móvel exponencial do OBV
            obvEMA = pd.Series(self.obv).ewm(span=n).mean()

            # self.indicatorOBV.append(self.obv[-1])
            # self.OBV.append(obvEMA.iloc[-1])

        return obvEMA.iloc[-1]


    def calc_MFI(self, n):

        value = 0

        # Se a quantidade de preços observados for superior a janela
        if len(self.prices) >= n + 1:

            p = sum(self.PF[-n:])
            n = sum(self.NF[-n:])

            if n == 0 or n==0.0:    n = 0.00001

            MFR = p / n
            
            value =  100 - (100/(1+MFR))
            
        return value


    def calc_MOM(self,n):
        value = 0

        # Se a quantidade de preços observados for superior a janela
        if len(self.prices) >= n + 1:

            value = pd.Series(self.prices).diff(n)
            
        return list(value)[-1]

    def push(self, event):
        
        orders = []
        
        
        price = event.price[3] # Captura o preço atual vindo do evento
        high = event.price[1] # Captura máxima atual vindo do evento
        low = event.price[2] # Captura mínima atual vindo do evento

        # Montando séries
        self.prices.append(price)
        self.highs.append(high)
        self.lows.append(low)

        # Montando a série de OBV:
        if len(self.prices) == 1:
            self.obv.append(0)
        elif price > (self.prices[-2]):
            self.obv.append(self.obv[-1] + event.quantity)
        elif price < (self.prices[-2]):
            self.obv.append(self.obv[-1] - event.quantity)
        else:
            self.obv.append(self.obv[-1])

        # Montando a série de TR
        if len(self.prices) > 1:
            self.TR.append(max(abs(high - low),abs(high - self.prices[-2]),abs(low  - self.prices[-2])))
        else:
            self.TR.append(-1)


        # Montando a série MF
        self.TP.append( (high+low+price)/3 )
        self.MF.append( ((high+low+price)/3)*event.quantity  )

        
        # Montando a série de OBV:
        if len(self.TP) == 1:
            self.PF.append(0)
            self.NF.append(0)
        elif self.TP[-1]>self.TP[-2]:
            self.PF.append(self.MF[-2])
            self.NF.append(0)
        elif self.TP[-1]<self.TP[-2]:
            self.PF.append(0)
            self.NF.append(self.MF[-2])
        else:
            self.PF.append(0)
            self.NF.append(0)


        
        if len(self.prices) >= self.lags+1:
            
            # Montagem
            # x = [[self.calc_RSI(6),self.calc_RSI(12),self.calc_MFI(14),self.calc_ATR(14),self.calc_MOM(1),self.calc_MOM(3),self.calc_OBV(14)]]
            x = [[self.calc_RSI(14),self.calc_MFI(14),self.calc_ATR(14),self.calc_MOM(3),self.calc_OBV(14)]]
            x = self.scaler.transform(x)

            signal = self.model.predict(x)[0]
            
            self.indicator.append(signal)
            
            # Alocação:
            if self.side != signal:
                
                orders.append(Order(event.instrument, -self.side, 0))
                orders.append(Order(event.instrument, signal, 0))
                    
                # Atualiza o sinal
                self.side = signal

        return orders

    def fill(self, id, instrument, price, quantity, status):
        super().fill(id, instrument, price, quantity, status)
        
        #Imprimindo o preenchimento das ordens
        if quantity != 0 and self.print_filled:
            print('Fill: {0} {1}@{2}'.format(instrument, quantity, price))




class BuynHold(Strategy):

    # Chamado de construtor: inicializa variáveis de suporte
    def __init__(self):
        self.bought = False # Indica se já comprou ou não

    # Função chamada a cada preço recebido
    def push(self, event):
        
        # Se ainda não comprou, faça
        # Note o uso do prefixo self. para as variáveis de suporte
        if not self.bought:
            self.bought = True
            
            # Envia uma ordem compra de 1 a mercado (preço zero)
            return [Order(event.instrument, 1, 0)]

        return [] # Caso não entre no if


