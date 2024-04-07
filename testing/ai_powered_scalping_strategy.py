from functools import reduce
import math
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
from typing import Optional
from freqtrade.strategy import (IStrategy)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from freqtrade.strategy.parameters import IntParameter
from datetime import datetime
from freqtrade.exchange import timeframe_to_prev_date
from technical.util import resample_to_interval,resampled_merge
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

class AIPoweredScalpingStrategy(IStrategy):

    # =============================================================
    # ===================== Strategy Config =======================
    # =============================================================
    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = '1m'
    stoploss = -1
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 10
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # =============================================================
    # ===================== STC Scalping Strategy =================
    # =============================================================
    Sensitivity = 1
    Atr_period = 10

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        a = datetime.now()
        # dataframe['predicted_value'] = self.predictSVM(dataframe)
        dataframe['ut_signal'] = self.calculate_ut_bot(dataframe,self.Sensitivity,self.Atr_period)
        dataframe['trend'] = self.adaptiveTrendFinder(dataframe)
        dataframe['trend_direction'] = dataframe['trend'].apply(lambda x: x[0])
        dataframe['trend_period'] = dataframe['trend'].apply(lambda x: x[1])
        dataframe['trend_strength'] = dataframe['trend'].apply(lambda x: x[2])
        b = datetime.now()
        print((b-a).microseconds * 0.001)
        # print(dataframe.loc[len(dataframe)-10:,['date','trend_direction','ut_signal','close']])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition = (dataframe['ut_signal'] == 1) & (dataframe['trend_direction'] > 0)
        sell_condition = (dataframe['ut_signal'] == -1) & (dataframe['trend_direction'] < 0)

        dataframe.loc[
            (
                buy_condition
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                sell_condition
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition = (dataframe['ut_signal'] == 1) & (dataframe['trend_direction'] > 0)
        sell_condition = (dataframe['ut_signal'] == -1) & (dataframe['trend_direction'] < 0)

        dataframe.loc[
            (
                sell_condition
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                buy_condition
            ),
            'exit_short'] = 1
        return dataframe
    
    # =============================================================
    # ============== Custom Entry and Exit Section ================
    # =============================================================  
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        
        is_best_time = self.is_quarter_hour(current_time)
        if (trade.calc_profit_ratio(current_rate) >= 0.01) & (is_best_time):
            return 'sell'

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 20
    
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 0
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 0,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 0,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 0,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 0,
                "required_profit": 0.01
            }
        ]
    
    def is_quarter_hour(self,time:datetime):
        seconds = time.second
        minutes = time.minute
        return (0 <= seconds < 3)
    
    # =============================================================
    # ===================== Machine Learning Helper ===============
    # =============================================================
    def calculate_CCI(self,dataframe,index,diff,ma,p):
        close_array = dataframe['close'].to_numpy()
        close_array = close_array[:index.name]
        diff = diff.to_numpy()
        diff = diff[index.name]
        ma = ma.to_numpy()
        ma = ma[index.name]
        if(len(close_array) < p):
            return 0
        # MAD
        s = 0
        for i in range(len(close_array),len(close_array)-p,-1):
            s = s + abs(dataframe['close'][i] - ma)
        mad = s / p
        # Scalping
        mcci = diff/mad/0.015
        return mcci


    def minimax(self,volume,x,p,min,max):
        volume_array = volume.to_numpy()
        volume_array = volume_array[:x.name+1]

        if(len(volume_array) < p):
            return 0

        hi = np.nan_to_num(np.max(volume_array[-p+1:]))
        lo = np.nan_to_num(np.min(volume_array[-p+1:]))

        return (max - min) * (volume_array[len(volume_array)-1] - lo)/(hi - lo) + min

    def scale(self,mom,index,loopback_period):
        mom_array = mom.to_numpy()
        mom_array = mom_array[:index.name+1]

        if(len(mom_array) < loopback_period):
            return 0
        
        current_mom = mom[index.name]
        hi = np.nan_to_num(np.max(mom_array[-loopback_period+1:]))
        lo = np.nan_to_num(np.min(mom_array[-loopback_period+1:]))
        
        return ((current_mom - lo) / (hi - lo)) * 100

    def calculate_sm1(self, s):
        s = np.array(s)  # Convert s to a NumPy array
        result = np.maximum(s, 0.0)
        return np.sum(result)

    def calculate_sm2(self, s):
        s = np.array(s)  # Convert s to a NumPy array
        result = np.where(s >= 0, 0.0, -s)
        return np.sum(result)

    def calculate_mfi_upper(self,s,x):
        result = np.where(s >= 0, 0.0, x)
        return np.sum(result)

    def calculate_mfi_lower(self,s,x):
        result = np.where(s <= 0, 0.0, x)
        return np.sum(result)

    def pine_cmo(self,src:pd.Series, length):
        # Calculate the momentum (change) of the source data
        mom = src - src.shift(1)
        
        # Calculate the sum of positive and negative momentum over the specified length
        sm1 = mom.rolling(length).apply(self.calculate_sm1,raw=True)
        sm2 = mom.rolling(length).apply(self.calculate_sm2,raw=True)
        
        # Calculate the Chande Momentum Oscillator (CMO)
        cmo = 100 * ((sm1 - sm2) / (sm1 + sm2))
        
        return cmo
    
    def calculate_CCI(self,dataframe,index,diff,ma,p):

        close_array = dataframe['close'].to_numpy()
        close_array = close_array[:index.name]
        diff = diff.to_numpy()
        diff = diff[index.name]
        ma = ma.to_numpy()
        ma = ma[index.name]

        if(len(close_array) < p):
            return 0

        # MAD
        s = 0

        for i in range(len(close_array),len(close_array)-p,-1):
            s = s + abs(dataframe['close'][i] - ma)
        mad = s / p

        # Scalping
        mcci = np.divide(diff, mad, out=np.zeros_like(diff), where=(mad != 0)) / 0.015
        return mcci


    def minimax(self,volume,x,p,min,max):
        volume_array = volume.to_numpy()
        volume_array = volume_array[:x.name+1]

        if(len(volume_array) < p):
            return 0

        hi = np.nan_to_num(np.max(volume_array[-p+1:]))
        lo = np.nan_to_num(np.min(volume_array[-p+1:]))
        if hi == lo or np.isnan(hi) or np.isnan(lo):
            return np.nan
        else:
            return (max - min) * (volume_array[len(volume_array)-1] - lo)/(hi - lo) + min
    
    # =============================================================
    # ===================== Machine Learning ======================
    # =============================================================
    def trainAndLearnSVM(self,features,output):

        data_combined = pd.concat([features, output], axis=1)
        data_combined.dropna(inplace=True)

        X = data_combined[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize']]
        y = np.where(data_combined['output'] > 0,1,-1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=37)

        # Step 2: Model Selection
        knn_model = KNeighborsClassifier(n_neighbors=1)
        gnb_model = GaussianNB()
        random_forest_model = RandomForestClassifier(n_estimators=15, random_state=37)

        model = VotingClassifier(
                                estimators=[
                                    ('knn', knn_model),
                                    ('rf', random_forest_model),
                                    ('gnb', gnb_model)
                                ], 
                                voting='soft',
                                )

        # Step 3: Model Training
        model.fit(X_train, y_train)
        # Step 4: Model Evaluation

        return model
    
    def extractFeatures(self,dataframe):
        LongWindow = 15
        MediumWindow = 5
        ShortWindow = 2

        dataframe_copy = dataframe
        # Input Source
        source = dataframe_copy['close']

        dataframe_copy['mas'] = ta.SMA(source,timeperiod=LongWindow)
        dataframe_copy['diffs'] = source - dataframe_copy['mas']
        dataframe_copy['maf'] = ta.SMA(source,timeperiod=ShortWindow)
        dataframe_copy['difff'] = source - dataframe_copy['maf']


        # ============ Long Window ============
        dataframe_copy['rs'] = ta.RSI(source,timeperiod=LongWindow)
        dataframe_copy['cs'] = dataframe_copy.apply((lambda index: self.calculate_CCI(dataframe_copy,index,dataframe_copy['diffs'],dataframe_copy['mas'],LongWindow)),axis=1)
        dataframe_copy['os'] = ta.ROC(source,timeperiod=LongWindow)
        dataframe_copy['vs'] = dataframe_copy.apply((lambda x: self.minimax(dataframe_copy['volume'],x,LongWindow,0,99)),axis=1)
        dataframe_copy['cmos'] = self.pine_cmo(source,LongWindow)
        dataframe_copy['emas'] = ta.EMA(dataframe_copy,timeperiod=LongWindow)

        # ============ Medium Window ============
        dataframe_copy['rm'] = ta.RSI(source,timeperiod=MediumWindow)
        dataframe_copy['cm'] = dataframe_copy.apply((lambda index: self.calculate_CCI(dataframe_copy,index,dataframe_copy['diffs'],dataframe_copy['mas'],MediumWindow)),axis=1)
        dataframe_copy['om'] = ta.ROC(source,timeperiod=MediumWindow)
        dataframe_copy['vm'] = dataframe_copy.apply((lambda x: self.minimax(dataframe_copy['volume'],x,MediumWindow,0,99)),axis=1)
        dataframe_copy['cmom'] = self.pine_cmo(source,MediumWindow)
        dataframe_copy['emam'] = ta.EMA(dataframe_copy,timeperiod=MediumWindow)

        # =========== Short window =============
        dataframe_copy['rf'] = ta.RSI(source,timeperiod=ShortWindow)
        dataframe_copy['cf'] = dataframe_copy.apply((lambda index: self.calculate_CCI(dataframe_copy,index,dataframe_copy['difff'],dataframe_copy['maf'],ShortWindow)),axis=1)
        dataframe_copy['of'] = ta.ROC(source,timeperiod=ShortWindow)
        dataframe_copy['vf'] = dataframe_copy.apply((lambda x: self.minimax(dataframe_copy['volume'],x,ShortWindow,0,99)),axis=1)
        dataframe_copy['cmof'] = self.pine_cmo(source,ShortWindow)
        dataframe_copy['emaf'] = ta.EMA(dataframe_copy,timeperiod=ShortWindow)

        dataframe_copy['f1'] = dataframe_copy.loc[:,['os','emas','cmos','rs','maf']].mean(axis=1)
        dataframe_copy['f2'] = dataframe_copy.loc[:,['om','emam','cmom','rm','cm','vm']].mean(axis=1)
        dataframe_copy['f3'] = dataframe_copy.loc[:,['of','emaf','cmof','rf','cf','vf']].mean(axis=1)
        dataframe_copy['output'] = dataframe_copy["close"].shift(-1) - dataframe_copy["close"]

        min_max_scaler = preprocessing.MinMaxScaler()
        dataframe_copy[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize']] = min_max_scaler.fit_transform(dataframe_copy[['f1','f2','f3']])

        # Step 1: Data Preparation
        features = dataframe_copy[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize','date','close']]
        output = dataframe_copy[['output']]
        return features,output

    def predictSVM(self,dataframe):
        features , output = self.extractFeatures(dataframe)
        model = self.trainAndLearnSVM(features,output)

        # Step 5: Prediction
        data_for_prediction = pd.concat([dataframe], axis=1)
        data_for_prediction.fillna(0,inplace=True)
        data_for_prediction['prediction_value_y'] = model.predict(data_for_prediction[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize']])
        return data_for_prediction['prediction_value_y']
    
    # =============================================================
    # ===================== STC Indicator =========================
    # =============================================================
    def calculateSTCIndicator(self,dataframe,length,fastLength,slowLength):
        EEEEEE = length
        BBBB = fastLength
        BBBBB = slowLength
        mAAAAA = self.AAAAA(dataframe,EEEEEE, BBBB, BBBBB)
        return mAAAAA

    
    def AAAA(self,BBB, BBBB, BBBBB):
        fastMA = ta.EMA(BBB, timeperiod=BBBB)
        slowMA = ta.EMA(BBB, timeperiod=BBBBB)
        AAAA = fastMA - slowMA
        return AAAA

    def AAAAA(self,dataframe,EEEEEE, BBBB, BBBBB):

        AAA = 0.5
        dataframe['DDD'] = 0.0
        dataframe['CCCCC'] = 0
        dataframe['DDDDDD'] = 0
        dataframe['EEEEE'] = 0.0
        
        dataframe['BBBBBB'] = self.AAAA(dataframe['close'], BBBB, BBBBB)
        dataframe['CCC'] = dataframe['BBBBBB'].rolling(window=EEEEEE).min()
        dataframe['CCCC'] = dataframe['BBBBBB'].rolling(window=EEEEEE).max() - dataframe['CCC']
        dataframe['CCCCC'] = np.where(dataframe['CCCC'] > 0, (dataframe['BBBBBB'] - dataframe['CCC']) / dataframe['CCCC'] * 100, dataframe['CCCCC'].shift(1).fillna(0))

        for i in range(0, len(dataframe)):
            if(i>0):
                dataframe.at[i, 'DDD'] = dataframe['DDD'].iloc[i-1] + (AAA * (dataframe.at[i, 'CCCCC'] - dataframe['DDD'].iloc[i-1]))

        dataframe['DDDD'] = dataframe['DDD'].rolling(window=EEEEEE).min()
        dataframe['DDDDD'] = dataframe['DDD'].rolling(window=EEEEEE).max() - dataframe['DDDD']
        dataframe['DDDDDD'] = np.where(dataframe['DDD'] > 0, (dataframe['DDD'] - dataframe['DDDD']) / dataframe['DDDDD'] * 100 , dataframe['DDD'].fillna(dataframe['DDDDDD'].shift(1)))

        for i in range(0, len(dataframe)):
            if(i>0):
                dataframe.at[i, 'EEEEE'] = dataframe['EEEEE'].iloc[i-1] + (AAA * (dataframe.at[i, 'DDDDDD'] - dataframe['EEEEE'].iloc[i-1]))

        dataframe.loc[
            (
                (dataframe['EEEEE'] > dataframe['EEEEE'].shift(1))
            ),
            'stc_signal'] = 1
        
        dataframe.loc[
            (
                (dataframe['EEEEE'] < dataframe['EEEEE'].shift(1))
            ),
            'stc_signal'] = -1
        
        return dataframe['stc_signal']
    
    # =============================================================
    # ===================== UT Bot ================================
    # =============================================================
    def xATRTrailingStop_func(self,close, prev_close, prev_atr, nloss):
        if close > prev_atr and prev_close > prev_atr:
            return max(prev_atr, close - nloss)
        elif close < prev_atr and prev_close < prev_atr:
            return min(prev_atr, close + nloss)
        elif close > prev_atr:
            return close - nloss
        else:
            return close + nloss
        
    def calculateEMA(self,src, length):
        alpha = 2 / (length + 1)
        
        # Initialize sum with the Simple Moving Average (SMA) for the first value
        sma_first_value = src.head(length).mean()
        sum_value = sma_first_value
        
        ema_values = []
        
        for value in src:
            if pd.isna(sum_value):
                sum_value = sma_first_value
            else:
                sum_value = alpha * value + (1 - alpha) * sum_value
            
            ema_values.append(sum_value)
        
        return pd.Series(ema_values, index=src.index)

    def heikinashi(self,df: pd.DataFrame) -> pd.DataFrame:
        df_HA = df.copy()
        df_HA['close'] = (df_HA['open'] + df_HA['high'] + df_HA['low'] + df_HA['close']) / 4

        for i in range(0, len(df_HA)):
            if i == 0:
                df_HA.loc[i, 'open'] = ((df_HA.loc[i, 'open'] + df_HA.loc[i, 'close']) / 2)
            else:
                df_HA.loc[i, 'open'] = ((df_HA.loc[i-1, 'open'] + df_HA.loc[i-1, 'close']) / 2)

        df_HA['high'] = df_HA[['open', 'close', 'high']].max(axis=1)
        df_HA['low'] = df_HA[['open', 'close', 'low']].min(axis=1)
        
        return df_HA

    def calculate_ut_bot(self,dataframe,SENSITIVITY,ATR_PERIOD):
        # UT Bot Parameters
        SENSITIVITY = SENSITIVITY
        ATR_PERIOD = ATR_PERIOD

        # Compute ATR And nLoss variable
        dataframe["xATR"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=ATR_PERIOD)
        dataframe["nLoss"] = SENSITIVITY * dataframe["xATR"]

        dataframe["ATRTrailingStop"] = [0.0] + [np.nan for i in range(len(dataframe) - 1)]

        for i in range(1, len(dataframe)):
            dataframe.loc[i, "ATRTrailingStop"] = self.xATRTrailingStop_func(
                dataframe.loc[i, "close"],
                dataframe.loc[i - 1, "close"],
                dataframe.loc[i - 1, "ATRTrailingStop"],
                dataframe.loc[i, "nLoss"],
            )

        dataframe['Ema'] = self.calculateEMA(dataframe['close'],1)
        dataframe["Above"] = self.calculate_crossover(dataframe['Ema'],dataframe["ATRTrailingStop"])
        dataframe["Below"] = self.calculate_crossover(dataframe["ATRTrailingStop"],dataframe['Ema'])

        # Buy Signal
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ATRTrailingStop"]) 
                & 
                (dataframe["Above"]==True)
            ),
            'UT_Signal'] = 1
        
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ATRTrailingStop"]) 
                & 
                (dataframe["Below"]==True)
            ),
            'UT_Signal'] = -1
        
        return dataframe['UT_Signal']

    def calculate_crossover(self,source1,source2):
        return (source1 > source2) & (source1.shift(1) <= source2.shift(1))
    

    # =============================================================
    # ============== Adaptive Trend Finder ========================
    # =============================================================
    periods = np.array([0,15,20,25,30])

    def adaptiveTrendFinder(self,dataframe:DataFrame):
        dataframe['trend_direction'] = dataframe.apply((lambda x: self.calculate_trend_direction(x,dataframe)),axis=1)
        return dataframe['trend_direction']
    
    def calculate_trend_direction(self,x,dataframe):
        if(x.name != len(dataframe)-1):
            return (0,0,0)

        devMultiplier = 2.0
        # Calculate Deviation,PersionR,Slope,Intercept
        stdDev01, pearsonR01, slope01, intercept01 = self.calcDev(self.periods[1],dataframe)
        stdDev02, pearsonR02, slope02, intercept02 = self.calcDev(self.periods[2],dataframe)
        stdDev03, pearsonR03, slope03, intercept03 = self.calcDev(self.periods[3],dataframe)
        # stdDev04, pearsonR04, slope04, intercept04 = self.calcDev(self.periods[4],dataframe)
        # stdDev05, pearsonR05, slope05, intercept05 = self.calcDev(self.periods[5],dataframe)
        # stdDev06, pearsonR06, slope06, intercept06 = self.calcDev(self.periods[6],dataframe)
        # stdDev07, pearsonR07, slope07, intercept07 = self.calcDev(self.periods[7],dataframe)
        # stdDev08, pearsonR08, slope08, intercept08 = self.calcDev(self.periods[8],dataframe)
        # stdDev09, pearsonR09, slope09, intercept09 = self.calcDev(self.periods[9],dataframe)
        # stdDev10, pearsonR10, slope10, intercept10 = self.calcDev(self.periods[10],dataframe)
        # stdDev11, pearsonR11, slope11, intercept11 = self.calcDev(self.periods[11],dataframe)
        # stdDev12, pearsonR12, slope12, intercept12 = self.calcDev(self.periods[12],dataframe)
        # stdDev13, pearsonR13, slope13, intercept13 = self.calcDev(self.periods[13],dataframe)
        # stdDev14, pearsonR14, slope14, intercept14 = self.calcDev(self.periods[14],dataframe)
        # stdDev15, pearsonR15, slope15, intercept15 = self.calcDev(self.periods[15],dataframe)
        # stdDev16, pearsonR16, slope16, intercept16 = self.calcDev(self.periods[16],dataframe)
        # stdDev17, pearsonR17, slope17, intercept17 = self.calcDev(self.periods[17],dataframe)
        # stdDev18, pearsonR18, slope18, intercept18 = self.calcDev(self.periods[18],dataframe)
        # stdDev19, pearsonR19, slope19, intercept19 = self.calcDev(self.periods[19],dataframe)

        # Find the highest Pearson's R
        # float highestPearsonR = pearsonR01
        highestPearsonR = max(pearsonR01, pearsonR02,pearsonR03,
                            #   pearsonR03, pearsonR04, pearsonR05, pearsonR06, pearsonR07, pearsonR08,
                            #    pearsonR09, pearsonR10, pearsonR11, pearsonR12, pearsonR13, pearsonR14, pearsonR15, pearsonR16, pearsonR17, pearsonR18, pearsonR19
                               )

        # Determine selected length, slope, intercept, and deviations
        detectedPeriod  = 0
        detectedSlope   = 0
        detectedIntrcpt = 0
        detectedStdDev  = 0

        if highestPearsonR == pearsonR01:
            detectedPeriod = self.periods[1]
            detectedSlope = slope01
            detectedIntrcpt = intercept01
            detectedStdDev = stdDev01
        elif highestPearsonR == pearsonR02:
            detectedPeriod = self.periods[2] 
            detectedSlope = slope02
            detectedIntrcpt = intercept02
            detectedStdDev = stdDev02
        elif highestPearsonR == pearsonR03:
            detectedPeriod = self.periods[3]  
            detectedSlope = slope03
            detectedIntrcpt = intercept03
            detectedStdDev = stdDev03
        # elif highestPearsonR == pearsonR04:
        #     detectedPeriod = self.periods[4]  
        #     detectedSlope = slope04
        #     detectedIntrcpt = intercept04
        #     detectedStdDev = stdDev04
        # elif highestPearsonR == pearsonR05:
        #     detectedPeriod = self.periods[5]  
        #     detectedSlope = slope05
        #     detectedIntrcpt = intercept05
        #     detectedStdDev = stdDev05
        # elif highestPearsonR == pearsonR06:
        #     detectedPeriod = self.periods[6]       
        #     detectedSlope = slope06
        #     detectedIntrcpt = intercept06
        #     detectedStdDev = stdDev06
        # elif highestPearsonR == pearsonR07:
        #     detectedPeriod = self.periods[7]      
        #     detectedSlope = slope07
        #     detectedIntrcpt = intercept07
        #     detectedStdDev = stdDev07
        # elif highestPearsonR == pearsonR08:
        #     detectedPeriod = self.periods[8]       
        #     detectedSlope = slope08
        #     detectedIntrcpt = intercept08
        #     detectedStdDev = stdDev08
        # elif highestPearsonR == pearsonR09:
        #     detectedPeriod = self.periods[9]       
        #     detectedSlope = slope09
        #     detectedIntrcpt = intercept09
        #     detectedStdDev = stdDev09
        # elif highestPearsonR == pearsonR10:
        #     detectedPeriod = self.periods[10]
        #     detectedSlope = slope10
        #     detectedIntrcpt = intercept10
        #     detectedStdDev = stdDev10
        # elif highestPearsonR == pearsonR11:
        #     detectedPeriod = self.periods[11]
        #     detectedSlope = slope11
        #     detectedIntrcpt = intercept11
        #     detectedStdDev = stdDev11
        # elif highestPearsonR == pearsonR12:
        #     detectedPeriod = self.periods[12]
        #     detectedSlope = slope12
        #     detectedIntrcpt = intercept12
        #     detectedStdDev = stdDev12
        # elif highestPearsonR == pearsonR13:
        #     detectedPeriod = self.periods[13]
        #     detectedSlope = slope13
        #     detectedIntrcpt = intercept13
        #     detectedStdDev = stdDev13
        # elif highestPearsonR == pearsonR14:
        #     detectedPeriod = self.periods[14]
        #     detectedSlope = slope14
        #     detectedIntrcpt = intercept14
        #     detectedStdDev = stdDev14
        # elif highestPearsonR == pearsonR15:
        #     detectedPeriod = self.periods[15]
        #     detectedSlope = slope15
        #     detectedIntrcpt = intercept15
        #     detectedStdDev = stdDev15
        # elif highestPearsonR == pearsonR16:
        #     detectedPeriod = self.periods[16]
        #     detectedSlope = slope16
        #     detectedIntrcpt = intercept16
        #     detectedStdDev = stdDev16
        # elif highestPearsonR == pearsonR17:
        #     detectedPeriod = self.periods[17]
        #     detectedSlope = slope17
        #     detectedIntrcpt = intercept17
        #     detectedStdDev = stdDev17
        # elif highestPearsonR == pearsonR18:
        #     detectedPeriod = self.periods[18]
        #     detectedSlope = slope18
        #     detectedIntrcpt = intercept18
        #     detectedStdDev = stdDev18
        # elif highestPearsonR == pearsonR19:
        #     detectedPeriod = self.periods[19]
        #     detectedSlope = slope19
        #     detectedIntrcpt = intercept19
        #     detectedStdDev = stdDev19
        else:
            # Default case
            raise Exception("Cannot Find Highest PearsonR") 
        
        # Calculate start and end price based on detected slope and intercept
        startPrice = math.exp(detectedIntrcpt + detectedSlope * (detectedPeriod - 1))
        endPrice = math.exp(detectedIntrcpt)
        startAtBar = len(dataframe) - detectedPeriod + 1

        # Calculate Upper Upper Price and Upper End price
        upperStartPrice = startPrice * math.exp(devMultiplier * detectedStdDev)
        upperEndPrice   =   endPrice * math.exp(devMultiplier * detectedStdDev)

        # Calculate Lower Price and Lower End Price
        lowerStartPrice = startPrice / math.exp(devMultiplier * detectedStdDev)
        lowerEndPrice =   endPrice / math.exp(devMultiplier * detectedStdDev)

        # Calculate If Uptrend or Downtrend and how strength is this trend
        # Also Know how many this trend exist with period
        # ====== Strategies ======
        # If EndPrice > StartPrice Uptrend
        # If EndPrice < StartPrice Downtrend
        trend_direction = endPrice - startPrice
        return (trend_direction,detectedPeriod,highestPearsonR)
    
    def calcDev(self,length:int,dataframe:DataFrame):
        logSource = dataframe['close'].apply(lambda x: math.log(x))
        period_1 = length -1
        sumX = 0.0
        sumXX = 0.0
        sumYX = 0.0
        sumY = 0.0
        for i in range(1,length+1):
            lSrc = logSource[len(dataframe)-i]
            sumX += i
            sumXX += i * i
            sumYX += i * lSrc
            sumY += lSrc

        slope = np.nan_to_num((length * sumYX - sumX * sumY) / (length * sumXX - sumX * sumX))
        average = sumY / length
        intercept = average - (slope * sumX / length) + slope
        sumDev = 0.0
        sumDxx = 0.0
        sumDyy = 0.0
        sumDyx = 0.0
        regres = intercept + slope * period_1 * 0.5
        sumSlp = intercept

        for i in range(1,period_1+1):
            lSrc = logSource[len(dataframe)-i]
            dxt = lSrc - average
            dyt  = sumSlp - regres
            lSrc   -= sumSlp
            sumSlp += slope
            sumDxx +=  dxt * dxt
            sumDyy +=  dyt * dyt
            sumDyx +=  dxt * dyt
            sumDev += lSrc * lSrc

        unStdDev = math.sqrt(sumDev / period_1)
        divisor  = sumDxx * sumDyy
        pearsonR = np.nan_to_num(sumDyx / math.sqrt(divisor))
        return unStdDev,pearsonR,slope,intercept