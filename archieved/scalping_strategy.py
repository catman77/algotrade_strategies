from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
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

class ScalpingStrategy(IStrategy):

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
    # Hyper Parameter
    stc_length_buy = IntParameter(2,50,default=12,space="buy",optimize=True)
    stc_fastLength_buy = IntParameter(5,50,default=26,space="buy",optimize=True)
    stc_slowLength_buy = IntParameter(10,80,default=50,space="buy",optimize=True)

    stc_length_sell = IntParameter(2,50,default=12,space="sell",optimize=True)
    stc_fastLength_sell = IntParameter(5,50,default=26,space="sell",optimize=True)
    stc_slowLength_sell = IntParameter(10,80,default=50,space="sell",optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe , dataframe_5min =  self.calculate5mintimeframe(dataframe)
        dataframe_5min = self.ExtractAndReturnNewDataframe(dataframe_5min)
        dataframe_5min['stc_signal_buy'] = self.calculateSTCIndicator(dataframe_5min,self.stc_length_buy.value,self.stc_fastLength_buy.value,self.stc_slowLength_buy.value)
        dataframe_5min['stc_signal_sell'] = self.calculateSTCIndicator(dataframe_5min,self.stc_length_sell.value,self.stc_fastLength_sell.value,self.stc_slowLength_sell.value)

        dataframe = resampled_merge(dataframe, dataframe_5min)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition =  dataframe['resample_5_stc_signal_buy'] == 1
        sell_condition = dataframe['resample_5_stc_signal_sell'] == -1
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
        buy_condition = (dataframe['resample_5_stc_signal_buy'] == 3)
        sell_condition = (dataframe['resample_5_stc_signal_sell'] == 3)

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

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        is_the_best_time_to_trade = self.is_quarter_hour(current_time)
        # & (last_candle['resample_5_stc_signal_buy'] == 1)
        should_buy_or_not_in_5m = (rate < last_candle['close']) & (rate < last_candle['resample_5_resample_5_close'])  & is_the_best_time_to_trade
        should_sell_or_not_in_5m = (rate > last_candle['close']) & (rate > last_candle['resample_5_resample_5_close']) & is_the_best_time_to_trade

        # istrue = self.count_digits_in_float(rate) == self.count_digits_in_float(last_candle['close'])

        if ((side == "long") & should_buy_or_not_in_5m):              
            return True
        elif ((side == "short") & should_sell_or_not_in_5m): 
            return True
        
        return False
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        # dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # last_candle = dataframe.iloc[-1].squeeze()

        # trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        # limit_to_exit = 0
        # istrue = self.count_digits_in_float(trade.open_rate) == self.count_digits_in_float(current_rate)

        # if(pair == "1000BONK/USDT:USDT"):
        #     if(istrue):
        #         coin1tradeprice = trade.open_rate
        #         limit_to_exit = self.add_deviation(coin1tradeprice,trade.is_short,4)
        #     else:
        #         coin1tradeprice = self.addzeroontheend(trade.open_rate)
        #         limit_to_exit = self.add_deviation(coin1tradeprice,trade.is_short,4)
        # if(pair == "SOL/USDT:USDT"):
        #     if(istrue):
        #         coin2tradeprice = trade.open_rate
        #         limit_to_exit = self.add_deviation(coin2tradeprice,trade.is_short,10)
        #     else:
        #         coin2tradeprice = self.addzeroontheend(trade.open_rate)
        #         limit_to_exit = self.add_deviation(coin2tradeprice,trade.is_short,10)
            
        # print("========================")
        # print(current_time)
        # print(current_rate)
        # print(trade.open_rate)
        # print(limit_to_exit)

        # if (trade.is_short == False) & (current_rate >= limit_to_exit):
        #     return "scalpexitlong"
        # if (trade.is_short == True) & (current_rate <= limit_to_exit):
        #     return "scalpexitshort"
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
    # ===================== STC Indicator Section ========================
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

    def calculateSTCIndicator(self,dataframe,length,fastLength,slowLength):
        EEEEEE = length
        BBBB = fastLength
        BBBBB = slowLength
        # dataframe = self.heikinashi(dataframe)
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
        
        return dataframe['stc_signal']