from functools import reduce
import sched
import time
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
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date
from technical.util import resample_to_interval,resampled_merge
import ccxt
import matplotlib.pyplot as plt
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
    trailing_stop = True
    trailing_stop_positive = 0.0001
    trailing_stop_positive_offset = 0.002
    trailing_only_offset_is_reached = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 60
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # =============================================================
    # ===================== STC Scalping Strategy =================
    # =============================================================
    stc_length_buy = IntParameter(2,50,default=12,space="buy",optimize=True)
    stc_fastLength_buy = IntParameter(5,50,default=26,space="buy",optimize=True)
    stc_slowLength_buy = IntParameter(10,80,default=50,space="buy",optimize=True)

    stc_length_sell = IntParameter(2,50,default=12,space="sell",optimize=True)
    stc_fastLength_sell = IntParameter(5,50,default=26,space="sell",optimize=True)
    stc_slowLength_sell = IntParameter(10,80,default=50,space="sell",optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.calculateIndicator(dataframe)
        dataframe['stc_signal_buy'] = self.calculateSTCIndicator(dataframe,self.stc_length_buy.value,self.stc_fastLength_buy.value,self.stc_slowLength_buy.value)
        dataframe['stc_signal_sell'] = self.calculateSTCIndicator(dataframe,self.stc_length_sell.value,self.stc_fastLength_sell.value,self.stc_slowLength_sell.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition =  (dataframe['predicted_value'] > 0) & (dataframe['stc_signal_buy'] == 1)
        sell_condition = (dataframe['predicted_value'] < 0) & (dataframe['stc_signal_sell'] == -1)
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
        buy_condition =  dataframe['predicted_value'] == 20
        sell_condition = dataframe['predicted_value'] == 20

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
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:

        # df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # last_candle = df.iloc[-1].squeeze()

        # is_the_best_time_to_trade = self.is_quarter_hour(current_time)
        is_the_best_time_to_trade = True

        if side == "long" and is_the_best_time_to_trade:              
            return True
        elif side == "short" and is_the_best_time_to_trade: 
            return True
        
        return False
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        # dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # last_candle = dataframe.iloc[-1].squeeze()

        # trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        is_the_best_time_to_trade = self.is_quarter_hour(current_time)
        # is_the_best_time_to_trade = True
        if(is_the_best_time_to_trade) & (current_profit > 0) & is_the_best_time_to_trade:
            return 'sell'
        if(is_the_best_time_to_trade) & (current_profit < 0) & ((current_time - trade.open_date_utc).seconds > 500) & is_the_best_time_to_trade:
            return 'stopsell'
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        is_the_best_time_to_trade = self.is_quarter_hour(current_time)
        if is_the_best_time_to_trade:
            # with open("output.txt", "a") as f:
            #     f.write("\n")
            #     f.write(f"============{pair}{trade.id}============\n")
            #     f.write(f"current-time: {current_time}\n")
            #     f.write(f"trade-open: {trade.open_date_utc}\n")
            #     f.write(f"profit : {trade.calc_profit_ratio(rate)}\n")
            #     f.write(f"reason: {exit_reason}\n")
            return True


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

    # =============================================================
    # ===================== Helper Function =======================
    # =============================================================
    def is_quarter_hour(self,time:datetime):
        seconds = time.second
        minutes = time.minute
        return (0 <= seconds < 3)
    
    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])
    
    def calculateBigTimeTrame(self,dataframe,minutes):
        big_timeframe = resample_to_interval(dataframe, self.get_ticker_indicator() * minutes)
        dataframe = resampled_merge(dataframe, big_timeframe)
        return dataframe
    
    # =============================================================
    # ===================== Machine Learning Helper ===============
    # =============================================================

    def convertBigTimeToNormal(self,data:DataFrame):
        columns = {
            "resample_5_open" : "open",
            "resample_5_high" : "high",
            "resample_5_low" : "low",
            "resample_5_close" : "close",
            "resample_5_volume" : "volume"
        }
        data = data.rename(columns=columns)
        return data

    def convertNormalToBigTime(self,data:DataFrame):
        columns = {
            "open" : "resample_5_open",
            "high" : "resample_5_high",
            "low" : "resample_5_low",
            "close" : "resample_5_close",
            "volume" : "resample_5_volume"
        }
        data = data.rename(columns=columns)
        return data

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
    
    # =============================================================
    # ===================== Machine Learning ======================
    # =============================================================
    def calculateIndicator(self,dataframe):
        LongWindow = 59
        MediumWindow = 28
        ShortWindow = 14

        dataframe_copy = dataframe
        # Input Source
        source = dataframe_copy['close']

        # ============ Long Window ============
        dataframe_copy['os'] = ta.ROC(source,timeperiod=LongWindow)
        dataframe_copy['cmos'] = self.pine_cmo(source,LongWindow)
        dataframe_copy['emas'] = ta.EMA(dataframe_copy,timeperiod=LongWindow)

        # ============ Medium Window ============
        dataframe_copy['om'] = ta.ROC(source,timeperiod=MediumWindow)
        dataframe_copy['cmom'] = self.pine_cmo(source,MediumWindow)
        dataframe_copy['emam'] = ta.EMA(dataframe_copy,timeperiod=MediumWindow)

        # =========== Short window =============
        dataframe_copy['of'] = ta.ROC(source,timeperiod=ShortWindow)
        dataframe_copy['cmof'] = self.pine_cmo(source,ShortWindow)
        dataframe_copy['emaf'] = ta.EMA(dataframe_copy,timeperiod=ShortWindow)

        dataframe_copy['f1'] = dataframe_copy.loc[:,['os','om','of']].mean(axis=1)
        dataframe_copy['f2'] = dataframe_copy.loc[:,['emas','emam','emaf']].mean(axis=1)
        dataframe_copy['f3'] = dataframe_copy.loc[:,['cmos','cmom','cmof']].mean(axis=1)
        dataframe_copy['output'] = dataframe_copy["close"].shift(-10) - dataframe_copy["close"]

        min_max_scaler = preprocessing.MinMaxScaler()
        dataframe_copy[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize']] = min_max_scaler.fit_transform(dataframe_copy[['f1','f2','f3']])

        # Step 1: Data Preparation
        features = dataframe_copy[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize','date','close']]
        output = dataframe_copy[['output']]

        data_combined = pd.concat([features, output], axis=1)
        data_for_prediction = features.loc[len(features)-10:].copy()
        data_combined.dropna(inplace=True)

        X = data_combined[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize']]
        y = np.where(data_combined['output'] > 0,1,-1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=37)

        # Step 2: Model Selection
        knn_model = KNeighborsClassifier(n_neighbors=1)
        gnb_model = GaussianNB()
        random_forest_model = RandomForestClassifier(n_estimators=15, random_state=20)

        model = VotingClassifier(
                                estimators=[
                                    ('knn', knn_model),
                                    ('rf', random_forest_model),
                                    ('gnb', gnb_model)
                                ], 
                                voting='hard',
                                )

        # Step 3: Model Training
        model.fit(X_train, y_train)

        # Step 5: Prediction
        data_for_prediction = pd.concat([dataframe], axis=1)
        data_for_prediction.fillna(0,inplace=True)
        data_for_prediction['last_10_predicted_value'] = model.predict(data_for_prediction[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize']])
        data_for_prediction['predicted_value'] = data_for_prediction['last_10_predicted_value'] + data_for_prediction['last_10_predicted_value'].shift(-1) + data_for_prediction['last_10_predicted_value'].shift(-2) + data_for_prediction['last_10_predicted_value'].shift(-3) + data_for_prediction['last_10_predicted_value'].shift(-4)
        data_for_prediction['predicted_value'] = data_for_prediction['predicted_value'].shift(10)
        return data_for_prediction
    
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
        
        dataframe.loc[
            (
                (dataframe['stc_signal'] == 1)
                &
                (dataframe['stc_signal'].shift(1) == -1)
            ),
            'stc_entry_exit'] = 1
        
        dataframe.loc[
            (
                (dataframe['stc_signal'] == -1)
                &
                (dataframe['stc_signal'].shift(1) == 1)
            ),
            'stc_entry_exit'] = -1
        
        return dataframe['stc_entry_exit']