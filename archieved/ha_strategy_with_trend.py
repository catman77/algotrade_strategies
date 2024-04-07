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
    timeframe = '1m'
    stoploss = -0.1
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 12
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
    # look for 10s candle and stc_signal
    periodfortrade = [0,3,3]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.unlock_pair(metadata['pair'])
        a = datetime.now()
        heikin_ashi_dataframe = self.heikinashi(dataframe)
        dataframe['psar_signal'] = self.parabolic_sar_ma_strategy(heikin_ashi_dataframe)
        dataframe['ha_signal'] = self.haCandleSignal(heikin_ashi_dataframe)
        dataframe['trendfortrade'] = self.adaptiveTrendFinder(heikin_ashi_dataframe,self.periodfortrade,True)
        b = datetime.now()
        print(f'{(b-a).microseconds * 0.001} ms')
        print(dataframe.loc[len(dataframe)-10:,['date','psar_signal','ha_signal','trendfortrade','close']])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition = (dataframe['trendfortrade'] > 0) & (dataframe['ha_signal'] == 1) & (dataframe['psar_signal'] == -1)
        sell_condition = (dataframe['trendfortrade'] < 0) & (dataframe['ha_signal'] == -1) & (dataframe['psar_signal'] == 1)
       
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
        buy_condition = (dataframe['trendfortrade'] > 0) & (dataframe['ha_signal'] == 1) & (dataframe['psar_signal'] == -1)
        sell_condition = (dataframe['trendfortrade'] < 0) & (dataframe['ha_signal'] == -1) & (dataframe['psar_signal'] == 1)
                
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
        is_best_time = self.is_quarter_hour(current_time)
        if (side == "long") & (is_best_time):
            return True
        if (side == "short") & (is_best_time):
            return True
        return False
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        is_best_time = self.is_quarter_hour(current_time)
        if(sell_reason == 'exit_signal') & is_best_time:
            return True
        if(sell_reason == 'stop_loss'):
            return False
        if(sell_reason == 'swp'):
            return False
        if(sell_reason == 'swl'):
            return False
        return False
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        is_best_time = self.is_quarter_hour(current_time)
        if (trade.calc_profit_ratio(current_rate) >= 0.01):
            return 'swp'
        if (trade.calc_profit_ratio(current_rate) < 0) & is_best_time & ((current_time - trade.open_date_utc).seconds >= 1200):
            return 'swl'

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 20
    
    # =============================================================
    # ===================== Strategy Helper =======================
    # =============================================================
    def is_quarter_hour(self,time:datetime):
        seconds = time.second
        return (seconds <= 3) & (seconds >= 0)
    
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
        dataframe['up_or_down_'] = dataframe['close'] - dataframe['close'].shift(1)
        dataframe['up_or_down_'] = np.sign(dataframe['up_or_down_'])
        dataframe.loc[
            (
                (dataframe['up_or_down_'].shift(1) < 0)
                &
                (dataframe['up_or_down_'] > 0)
            ),
            'up_or_down'] = 1

        dataframe.loc[
            (
                (dataframe['up_or_down_'].shift(1) > 0)
                &
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

        return dataframe['sar']
    