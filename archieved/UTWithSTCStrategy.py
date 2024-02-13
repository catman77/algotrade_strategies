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

class UTWithSTCStrategy(IStrategy):

    # =============================================================
    # ===================== Strategy Config =======================
    # =============================================================
    INTERFACE_VERSION = 3
    can_short: bool = True
    stoploss = -0.1
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.002
    trailing_only_offset_is_reached = True 
    timeframe = '5m'
    process_only_new_candles = False
    use_exit_signal = False
    exit_profit_only = False
    startup_candle_count: int = 60
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }


    # =============================================================
    # ===================== UT with STC Strategy ==================
    # =============================================================
    # Hyper Parameter
    ut_bot_SENSITIVITY_BUY  = IntParameter(1, 3, default=1,space="buy",optimize=True)
    ut_bot_ATR_PERIOD_BUY = IntParameter(1,20,default=10,space="buy",optimize=True)
    stc_length_buy = IntParameter(2,10,default=2,space="buy",optimize=True)
    stc_fastLength_buy = IntParameter(5,30,default=5,space="buy",optimize=True)
    stc_slowLength_buy = IntParameter(10,60,default=10,space="buy",optimize=True)

    ut_bot_SENSITIVITY_SELL  = IntParameter(1, 3, default=1,space="sell",optimize=True)
    ut_bot_ATR_PERIOD_SELL = IntParameter(1,20,default=10,space="sell",optimize=True)
    stc_length_sell = IntParameter(2,10,default=2,space="sell",optimize=True)
    stc_fastLength_sell = IntParameter(5,30,default=5,space="sell",optimize=True)
    stc_slowLength_sell = IntParameter(10,60,default=10,space="sell",optimize=True)

    loopback_period_long = IntParameter(2,10,default=5,space="buy",optimize=True)
    loopback_period_short = IntParameter(2,10,default=5,space="sell",optimize=True)

    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['UT_Signal_Buy'] = self.calculate_ut_bot(dataframe,self.ut_bot_SENSITIVITY_BUY.value,self.ut_bot_ATR_PERIOD_BUY.value)
        dataframe['stc_signal_buy'] = self.calculateSTCIndicator(dataframe,self.stc_length_buy.value,self.stc_fastLength_buy.value,self.stc_slowLength_buy.value)

        dataframe['UT_Signal_Sell'] = self.calculate_ut_bot(dataframe,self.ut_bot_SENSITIVITY_SELL.value,self.ut_bot_ATR_PERIOD_SELL.value)
        dataframe['stc_signal_sell'] = self.calculateSTCIndicator(dataframe,self.stc_length_sell.value,self.stc_fastLength_sell.value,self.stc_slowLength_sell.value)

        dataframe['stop_loss_long'] = self.calculate_stoploss_long(dataframe,self.loopback_period_long.value)
        dataframe['stop_loss_short'] = self.calculate_stoploss_short(dataframe,self.loopback_period_short.value)
        # print(dataframe.loc[len(dataframe)-50:,['stc_signal','close']])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition =  (dataframe['UT_Signal_Buy'] == 1) & (dataframe['stc_signal_buy'] == 1)
        sell_condition = (dataframe['UT_Signal_Sell'] == -1) & (dataframe['stc_signal_sell'] == -1)
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
        buy_condition = (dataframe['stc_signal_buy'] == 3)
        sell_condition = (dataframe['stc_signal_sell'] == 3)

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
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"]):
                return True
        else:
            if rate < (last_candle["close"]):
                return True

        return False
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            take_profit_limit = 0
            if(trade.is_short == True):
                take_profit_limit = trade.open_rate + ((trade.open_rate - trade_candle['stop_loss_short']) * 1)
            if(trade.is_short == False):
                take_profit_limit = trade.open_rate + ((trade.open_rate - trade_candle['stop_loss_long']) * 1)

            if trade.is_short == True and (current_rate >= trade_candle['stop_loss_short']) :
                return 'sl_short'
            if trade.is_short == False and (current_rate <= trade_candle['stop_loss_long']):
                return 'sl_long'
            
            if trade.is_short == True and (current_rate <= take_profit_limit) :
                return 'tp_short'
            if trade.is_short == False and (current_rate >= take_profit_limit):
                return 'tp_long'
        
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                            rate: float, time_in_force: str, exit_reason: str,
                            current_time: datetime, **kwargs) -> bool:
        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 40
    

    # =============================================================
    # ===================== StopLoss Rate ========================
    # =============================================================
    def calculate_stoploss_long(self,dataframe,window):
        dataframe['stop_loss_long'] = dataframe['low'].rolling(window=window).min()      
        return dataframe['stop_loss_long']
    
    def calculate_stoploss_short(self,dataframe,window):
        dataframe['stop_loss_short'] = dataframe['high'].rolling(window=window).max()           
        return dataframe['stop_loss_short']

    # =============================================================
    # ===================== STC Indicator Section ========================
    # =============================================================
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

        AAA = 0.1
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

    # =============================================================
    # ===================== UT Bot Section ========================
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

        # dataframe = self.heikinashi(dataframe)

        # Compute ATR And nLoss variable
        dataframe["xATR"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=ATR_PERIOD)
        dataframe["nLoss"] = SENSITIVITY * dataframe["xATR"]

        #Drop all rows that have nan, X first depending on the ATR preiod for the moving average
        # dataframe = dataframe.dropna()
        # dataframe = dataframe.reset_index()

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