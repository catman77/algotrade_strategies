from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.base import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from freqtrade.persistence import Trade
from typing import Optional
from freqtrade.strategy import (IStrategy)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.strategy.parameters import IntParameter
from technical.util import resample_to_interval,resampled_merge

class AIPoweredScalpingStrategy(IStrategy):

    # =============================================================
    # ===================== Strategy Config =======================
    # =============================================================
    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = '1m'
    stoploss = -0.1
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
    # ===================== STC Scalping Strategy ==================
    # =============================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.calculateIndicator(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition =  dataframe['predicted_value'] == 'up'
        sell_condition = dataframe['predicted_value'] == 'down'
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
        buy_condition = (dataframe['predicted_value'] == '3')
        sell_condition = (dataframe['predicted_value'] == '3')

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
        # # & (last_candle['resample_5_stc_signal_buy'] == 1)
        # should_buy_or_not_in_5m = (rate < last_candle['close']) & (rate < last_candle['resample_5_resample_5_close'])  & is_the_best_time_to_trade
        # should_sell_or_not_in_5m = (rate > last_candle['close']) & (rate > last_candle['resample_5_resample_5_close']) & is_the_best_time_to_trade

        # # istrue = self.count_digits_in_float(rate) == self.count_digits_in_float(last_candle['close'])

        # if ((side == "long") & should_buy_or_not_in_5m):              
        #     return True
        # elif ((side == "short") & should_sell_or_not_in_5m): 
        #     return True
        
        return True
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        if(current_profit > 0):
            return 'sell'
        
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                            rate: float, time_in_force: str, exit_reason: str,
                            current_time: datetime, **kwargs) -> bool:
        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 20

    # =============================================================
    # ===================== Helper Function =======================
    # =============================================================
    def ExtractAndReturnNewDataframe(self,dataframe):
        dataframe['date'] = dataframe['resample_5_date']
        dataframe['open'] = dataframe['resample_5_open']
        dataframe['high'] = dataframe['resample_5_high']
        dataframe['low'] = dataframe['resample_5_low']
        dataframe['close'] = dataframe['resample_5_close']
        dataframe['volume'] = dataframe['resample_5_volume']
        return dataframe

    def addzeroontheend(self,value):
        original_float_str = str(value)
        modified_float_str = original_float_str + '0'
        return modified_float_str

    def is_quarter_hour(self,time:datetime):
        minutes = time.minute
        seconds = time.second
        return (seconds <= 50) & (seconds >= 30) & (minutes in [2,7,12,17,22,27,32,37,42,47,52,57])

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])
    
    def calculate5mintimeframe(self,dataframe):
        dataframe_short = resample_to_interval(dataframe, self.get_ticker_indicator() * 5)
        dataframe = resampled_merge(dataframe, dataframe_short)
        return dataframe , dataframe_short

    def count_digits_in_float(self,num):
        if isinstance(num, float):
            num_str = str(num)
            integer_part, _, fractional_part = num_str.partition('.')
            return len(integer_part) + len(fractional_part)
        else:
            return 0
        
    def add_deviation(self,number,isShort,deviation):
        num_str = str(number)
        integer_part, decimal_part = num_str.split('.')
        last_two_digits = decimal_part[-2:]
        last_two_digits_int = int(last_two_digits)
        new_last_two_digits = 0
        if isShort == False:
            new_last_two_digits = str(last_two_digits_int + deviation).zfill(2)
        if isShort == True:
            new_last_two_digits = str(last_two_digits_int - deviation).zfill(2)
        new_last_two_digits = new_last_two_digits.replace("-","")
        new_decimal_part = decimal_part[:-2] + new_last_two_digits
        new_number_str = integer_part + '.' + new_decimal_part
        new_number = float(new_number_str)
        return new_number
    
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

    def calculate_sm1(self,s):
        result = []
        for value in s:
            result.append(value if value >= 0 else 0.0)
        return np.sum(result)

    def calculate_sm2(self,s):
        result = []
        for value in s:
            result.append(0.0 if value >= 0 else -value)
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
        source = dataframe_copy['open']+dataframe_copy['high']+dataframe_copy['low']/3

        # DIFF
        dataframe_copy['mas'] = ta.SMA(source,timeperiod=LongWindow)
        dataframe_copy['diffs'] = source - dataframe_copy['mas']

        dataframe_copy['maf'] = ta.SMA(source,timeperiod=ShortWindow)
        dataframe_copy['difff'] = source - dataframe_copy['maf']

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

        # Core Data
        # Step 1: Data Preparation
        features = dataframe_copy[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize','date','close']]
        output = dataframe_copy[['output']]

        data_combined = pd.concat([features, output], axis=1)
        data_for_prediction = features.loc[len(features)-10:].copy()
        data_combined.dropna(inplace=True)

        X = data_combined[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize']]
        y = np.where(data_combined['output'] > 0,'up','down')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=37)

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
                                voting='soft',
                                )

        # Step 3: Model Training
        model.fit(X_train, y_train)

        # Step 4: Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Step 5: Prediction
        # Assuming X_new is your new data for prediction
        data_for_prediction = pd.concat([dataframe], axis=1)
        data_for_prediction.fillna(0,inplace=True)
        data_for_prediction['predicted_value'] = model.predict(data_for_prediction[['f1_slow_normalize','f2_medium_normalize','f3_fast_normalize']])
        print(data_for_prediction.loc[len(data_for_prediction)-10:,['date','predicted_value','close']])
        return data_for_prediction