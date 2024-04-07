from decimal import Decimal
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

class AIPoweredScalpingStrategyV2(IStrategy):

    # =============================================================
    # ===================== Strategy Config =======================
    # =============================================================
    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = '3m'
    stoploss = -1
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 12
    leverage_value = 20
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
    periodfortrade = [0,5,10]
    heikin_ashi_dataframe = DataFrame()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.unlock_pair(metadata['pair'])
        a = datetime.now()
        self.heikin_ashi_dataframe = self.heikinashi(dataframe)
        dataframe['ha_signal'] = self.haCandleSignal(self.heikin_ashi_dataframe)
        dataframe['trendfortrade'] = self.adaptiveTrendFinder(self.heikin_ashi_dataframe,self.periodfortrade,True)
        b = datetime.now()
        print(f'{(b-a).microseconds * 0.001} ms')
        # print(dataframe.loc[len(dataframe)-10:,['date','ha_signal','trendfortrade','close']])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition = (dataframe['ha_signal'] > 0) & (dataframe['trendfortrade'] > 0)
        sell_condition = (dataframe['ha_signal'] < 0) & (dataframe['trendfortrade'] < 0)
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
        buy_condition = (dataframe['ha_signal'] == 3) 
        sell_condition = (dataframe['ha_signal'] == 3) 
                
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
    def bot_start(self, **kwargs) -> None:
        for pair in self.dp.current_whitelist():
            if(self.trade_candle_previous_min_and_max.keys().__contains__(pair) == False):
                            self.trade_candle_previous_min_and_max[pair] = {"min":0,"max":0}
  
    trade_candle_previous_min_and_max = {}

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
        is_best_time = self.is_quarter_hour(current_time)
        dataframe,_ = self.dp.get_analyzed_dataframe(pair,self.timeframe)
        last_candle = dataframe.iloc[-2].squeeze()
        previous_last_candle_max_value = max(last_candle['open'],last_candle['high'],last_candle['low'],last_candle['close'])
        previous_last_candle_min_value = min(last_candle['open'],last_candle['high'],last_candle['low'],last_candle['close'])

        # if(side == "long"):
        #     deviation_of_previous_candle = self.calculate_deviation(rate,previous_last_candle_max_value)
        # else:
        #     deviation_of_previous_candle = self.calculate_deviation(rate,previous_last_candle_min_value)

        # price_to_exit = self.add_deviation(rate,(side == "short"),deviation_of_previous_candle)
        # profit_ratio = self.calculate_profit_ratio(open_rate=rate,exit_rate=price_to_exit,is_short=(side == "short"),leverage=self.leverage_value,stake_ammount=10)
        should_trade_by_checking_previous = True

        if (side == "long") & is_best_time & should_trade_by_checking_previous :
            self.trade_candle_previous_min_and_max[pair] = {"min":previous_last_candle_min_value,"max":previous_last_candle_max_value}
            return True
        if (side == "short") & is_best_time & should_trade_by_checking_previous :
            self.trade_candle_previous_min_and_max[pair] = {"min":previous_last_candle_min_value,"max":previous_last_candle_max_value}
            return True
        return False
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        if(sell_reason == 'exit_signal'):
            self.trade_candle_previous_min_and_max[pair] = {"min":0,"max":0}
            return False
        if(sell_reason == 'stop_loss'):
            self.trade_candle_previous_min_and_max[pair] = {"min":0,"max":0}
            return False
        if(sell_reason == 'swp'):
            self.trade_candle_previous_min_and_max[pair] = {"min":0,"max":0}
            return True
        if(sell_reason == 'swl'):
            self.trade_candle_previous_min_and_max[pair] = {"min":0,"max":0}
            return True
        return False

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        current_profit_in_trade = trade.calc_profit_ratio(current_rate)
        trade_open_rate = trade.open_rate

        previous_candle_max_value = self.trade_candle_previous_min_and_max[pair]["max"]
        previous_candle_min_value = self.trade_candle_previous_min_and_max[pair]["min"]

        if(self.count_digits_in_float(trade.open_rate) != self.count_digits_in_float(current_rate)):
            trade_open_rate = self.addZeroToLastDigit(trade_open_rate)

        # if(trade.is_short == False):
        #     deviation_of_previous_candle = self.calculate_deviation(trade_open_rate,previous_candle_max_value)
        # else:
        #     deviation_of_previous_candle = self.calculate_deviation(trade_open_rate,previous_candle_min_value)

        price_to_exit = self.add_deviation(trade_open_rate,trade.is_short,10)

        if(current_profit_in_trade < 0):
            if(trade.is_short == False):
                if(current_rate <= previous_candle_min_value):
                    return "swl"
            if(trade.is_short == True):
                if(current_rate >= previous_candle_max_value):
                    return "swl"
        else:
            if(trade.is_short == False):
                if(current_rate >= price_to_exit):
                    return "swp"
            if(trade.is_short == True):
                if(current_rate <= price_to_exit):
                    return "swp"

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_value
    
    # =============================================================
    # ===================== Strategy Helper =======================
    # =============================================================
    def is_quarter_hour(self,time:datetime):
        seconds = time.second
        # minute = time.minute
        #  & (minute in [0,5,10,15,20,25,30,35,40,45,50,55])
        return (seconds <= 2) & (seconds >= 0)

    def calculate_deviation(self,num1, num2):
        diff = abs(Decimal(str(num1)) - Decimal(str(num2)))
        diff_str = str(diff).lstrip('0').replace('.', '')        
        deviation_result = int(diff_str)
        return deviation_result
    
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
        last_three_digits = decimal_part[-3:]
        last_three_digits_int = int(last_three_digits)
        new_last_three_digits = 0
        if isShort == False:
            new_last_three_digits = str(last_three_digits_int + deviation).zfill(3)
        if isShort == True:
            new_last_three_digits = str(last_three_digits_int - deviation).zfill(3)
        new_last_three_digits = new_last_three_digits.replace("-","")
        new_decimal_part = decimal_part[:-3] + new_last_three_digits
        new_number_str = integer_part + '.' + new_decimal_part
        new_number = float(new_number_str)
        return new_number

    def addZeroToLastDigit(self,num:float):
        str_num = str(num) + '0'
        return str_num
    
    def calculate_profit_ratio(self,open_rate,exit_rate,is_short,leverage,stake_ammount):
        quantity = (stake_ammount*leverage)/open_rate
        initial_margin = quantity * open_rate * (1/leverage)
        pnl = 0
        roi = 0
        if(is_short == False):
            pnl = (exit_rate - open_rate) * quantity
        else:
            pnl = (open_rate - exit_rate) * quantity

        roi = pnl / initial_margin
        return round(roi * 100,2)
    
    # =============================================================
    # ===================== Heikinashi ============================
    # =============================================================
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

    def haCandleSignal(self,dataframe:DataFrame):
        dataframe['up_or_down_'] = dataframe['close'] - dataframe['open']
        dataframe['up_or_down_'] = np.sign(dataframe['up_or_down_'])
        dataframe.loc[
            (
                (dataframe['up_or_down_'] > 0)
            ),
            'up_or_down'] = 1

        dataframe.loc[
            (
                (dataframe['up_or_down_'] < 0)
            ),
            'up_or_down'] = -1
        return dataframe['up_or_down']

    # =============================================================
    # ============== Adaptive Trend Finder ========================
    # =============================================================
    def adaptiveTrendFinder(self,dataframe:DataFrame,periods:list[int],isTrade:bool):
        dataframe['trend_direction'] = dataframe.apply((lambda x: self.calculate_trend_direction(x,dataframe,periods)),axis=1)
        dataframe['trend_direction_temp'] = dataframe['trend_direction'].apply(lambda x: x[0])
        if(isTrade):
            # Trade
            dataframe.loc[
                (
                    (dataframe['trend_direction_temp'].shift(1) < 0)
                    &
                    (dataframe['trend_direction_temp'] > 0)
                ),
                'trend_direction_real'] = 1

            dataframe.loc[
                (
                    (dataframe['trend_direction_temp'].shift(1) > 0)
                    &
                    (dataframe['trend_direction_temp'] < 0)
                ),
                'trend_direction_real'] = -1
        else:
            # Exit
            dataframe.loc[
                (
                    (dataframe['trend_direction_temp'] > 0)
                ),
                'trend_direction_real'] = 1

            dataframe.loc[
                (
                    (dataframe['trend_direction_temp'] < 0)
                ),
                'trend_direction_real'] = -1
            
        return dataframe['trend_direction_real']
    
    def calculate_trend_direction(self,x,dataframe,periods):
        # if(x.name == len(dataframe)-1) | (x.name == len(dataframe)-2):
        if (x.name >= periods[2]):
            devMultiplier = 2.0
            # Calculate Deviation,PersionR,Slope,Intercept
            stdDev01, pearsonR01, slope01, intercept01 = self.calcDev(periods[1],dataframe,x.name)
            stdDev02, pearsonR02, slope02, intercept02 = self.calcDev(periods[2],dataframe,x.name)

            # Find the highest Pearson's R
            highestPearsonR = max(pearsonR01, pearsonR02)

            # Determine selected length, slope, intercept, and deviations
            detectedPeriod  = 0
            detectedSlope   = 0
            detectedIntrcpt = 0
            detectedStdDev  = 0

            if highestPearsonR == pearsonR01:
                detectedPeriod = periods[1]
                detectedSlope = slope01
                detectedIntrcpt = intercept01
                detectedStdDev = stdDev01
            elif highestPearsonR == pearsonR02:
                detectedPeriod = periods[2] 
                detectedSlope = slope02
                detectedIntrcpt = intercept02
                detectedStdDev = stdDev02
            else:
                # Default case
                raise Exception(f"Cannot Find Highest PearsonR") 
            
            # Calculate start and end price based on detected slope and intercept
            startPrice = math.exp(detectedIntrcpt + detectedSlope * (detectedPeriod - 1))
            endPrice = math.exp(detectedIntrcpt)

            trend_direction = endPrice - startPrice
            return (trend_direction,detectedPeriod,highestPearsonR)
        return (0,0,0)
    
    def calcDev(self,length:int,dataframe:DataFrame,index:int):
        logSource = dataframe['close'].apply(lambda x: math.log(x))
        period_1 = length -1
        sumX = 0.0
        sumXX = 0.0
        sumYX = 0.0
        sumY = 0.0
        for i in range(1,length+1):
            lSrc = logSource[index+1-i]
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
            lSrc = logSource[index+1-i]
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
        if divisor == 0 or np.isnan(divisor):
            pearsonR = 0  # Set Pearson correlation coefficient to NaN
        else:
            pearsonR = sumDyx / math.sqrt(divisor)
        return unStdDev,pearsonR,slope,intercept
    

    # =============================================================
    # ============== Parabolic Sar ================================
    # =============================================================
    def PSAR(self,df, af=0.02, max=0.2):
        df.loc[0, 'AF'] = 0.02
        df.loc[0, 'PSAR'] = df.loc[0, 'low']
        df.loc[0, 'EP'] = df.loc[0, 'high']
        df.loc[0, 'PSARdir'] = 1

        for a in range(1, len(df)):
            if df.loc[a-1, 'PSARdir'] == 1:
                df.loc[a, 'PSAR'] = df.loc[a-1, 'PSAR'] + (df.loc[a-1, 'AF']*(df.loc[a-1, 'EP']-df.loc[a-1, 'PSAR']))
                df.loc[a, 'PSARdir'] = 1

                if df.loc[a, 'low'] < df.loc[a-1, 'PSAR'] or df.loc[a, 'low'] < df.loc[a, 'PSAR']:
                    df.loc[a, 'PSARdir'] = -1
                    df.loc[a, 'PSAR'] = df.loc[a-1, 'EP']
                    df.loc[a, 'EP'] = df.loc[a-1, 'low']
                    df.loc[a, 'AF'] = af
                else:
                    if df.loc[a, 'high'] > df.loc[a-1, 'EP']:
                        df.loc[a, 'EP'] = df.loc[a, 'high']
                        if df.loc[a-1, 'AF'] <= 0.18:
                            df.loc[a, 'AF'] =df.loc[a-1, 'AF'] + af
                        else:
                            df.loc[a, 'AF'] = df.loc[a-1, 'AF']
                    elif df.loc[a, 'high'] <= df.loc[a-1, 'EP']:
                        df.loc[a, 'AF'] = df.loc[a-1, 'AF']
                        df.loc[a, 'EP'] = df.loc[a-1, 'EP']

            elif df.loc[a-1, 'PSARdir'] == -1:
                df.loc[a, 'PSAR'] = df.loc[a-1, 'PSAR'] - (df.loc[a-1, 'AF']*(df.loc[a-1, 'PSAR']-df.loc[a-1, 'EP']))
                df.loc[a, 'PSARdir'] = -1

                if df.loc[a, 'high'] > df.loc[a-1, 'PSAR'] or df.loc[a, 'high'] > df.loc[a, 'PSAR']:
                    df.loc[a, 'PSARdir'] = 1
                    df.loc[a, 'PSAR'] = df.loc[a-1, 'EP']
                    df.loc[a, 'EP'] = df.loc[a-1, 'high']
                    df.loc[a, 'AF'] = af
                else:
                    if df.loc[a, 'low'] < df.loc[a-1, 'EP']:
                        df.loc[a, 'EP'] = df.loc[a, 'low']
                        if df.loc[a-1, 'AF'] < max:
                            df.loc[a, 'AF'] = df.loc[a-1, 'AF'] + af
                        else:
                            df.loc[a, 'AF'] = df.loc[a-1, 'AF']

                    elif df.loc[a, 'low'] >= df.loc[a-1, 'EP']:
                        df.loc[a, 'AF'] = df.loc[a-1, 'AF']
                        df.loc[a, 'EP'] = df.loc[a-1, 'EP']
        return df['PSARdir']

    def parabolic_sar_ma_strategy(self,dataframe:DataFrame):
        # Calculate Parabolic SAR
        dataframe['sar'] = self.PSAR(dataframe)

        buy_signal = (dataframe['sar'] == 1) & (dataframe['sar'].shift(1) == -1)
        sell_signal = (dataframe['sar'] == -1) & (dataframe['sar'].shift(1) == 1)

        dataframe.loc[
            (
                buy_signal
            ),
            'psar_ma_signal'] = 1
        
        dataframe.loc[
            (
                sell_signal
            ),
            'psar_ma_signal'] = -1
        return dataframe['psar_ma_signal']
    