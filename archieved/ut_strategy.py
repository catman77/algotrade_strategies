from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
from typing import Optional, Union
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import math
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

class UTStrategy(IStrategy):

    INTERFACE_VERSION = 3
    use_custom_stoploss = False

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.002
    trailing_only_offset_is_reached = True

    can_short: bool = True

    # TODO Adjust this parameter
    stoploss = -1
    minimal_roi = {
        "0": 0.2
    }

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 60

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Strategy Constant

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # dataframe['UT_Signal_Sell'] = self.calculate_ut_bot(dataframe,2,300)
        # dataframe['UT_Signal_Buy'] = self.calculate_ut_bot(dataframe,2,300)

        dataframe[f'trend_direction'] = self.adaptiveTrendFinder_2(dataframe)
        dataframe[f'trend'] = dataframe['trend_direction'].apply(lambda x: x[0])
        dataframe[f'trend-period'] = dataframe['trend_direction'].apply(lambda x: x[1])
        dataframe[f'trend-strength'] = dataframe['trend_direction'].apply(lambda x: x[2])

        # STC Indicator
        dataframe['stc_signal'] = self.calculateSTCIndicator(dataframe,2,50,80)
        # print(dataframe.loc[len(dataframe)-50:,['stc_signal','close']])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trend'] > 0)
                &
                (dataframe['stc_signal'] == 1)
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['trend'] < 0)
                &
                (dataframe['stc_signal'] == -1)
            ),
            'enter_short'] = 1
        return dataframe
    

    # ============== Confirm Trade Entry==========
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

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # (dataframe['UT_Signal_Buy'] == -1)
                # &
                (dataframe['stc_signal'] == -1)
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                # (dataframe['UT_Signal_Sell'] == -1)
                # &
                (dataframe['stc_signal'] == -1)
            ),
            'exit_short'] = 1
        return dataframe
    
    # ================== Confrim Custom Exit
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Sell any positions at a loss if they are losing in 10 minutes.
        # if current_profit > 0 and ((current_time - trade.open_date_utc).seconds >= 1200):
        #     return 'swp'
        # if current_profit < 0 and ((current_time - trade.open_date_utc).seconds >= 7000):
        #     return 'fexit'
        
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                            rate: float, time_in_force: str, exit_reason: str,
                            current_time: datetime, **kwargs) -> bool:
        
        # if exit_reason == "swp":
        #     return True
        # if exit_reason == "exit_signal":
        #     return True
        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:

        return 40
    

    # Helper Function
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
                # &
                # (dataframe['EEEEE'] < 15)
            ),
            'stc_signal'] = 1
        
        dataframe.loc[
            (
                (dataframe['EEEEE'] < dataframe['EEEEE'].shift(1))
                # &
                # (dataframe['EEEEE'] > 90.5)
            ),
            'stc_signal'] = -1
        
        return dataframe['stc_signal']

    # Function to compute ATRTrailingStop

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

    def calculateFilter(self,dataframe):
        # Calculate ATR VALUE FOR 10 and 40
        dataframe['atr_low'] = ta.ATR(dataframe['high'],dataframe['low'],dataframe['close'],timeperiod=1)
        dataframe['atr_high'] = ta.ATR(dataframe['high'],dataframe['low'],dataframe['close'],timeperiod=10)
        dataframe['atr_filter'] = dataframe['atr_low'] > dataframe['atr_high']

        return dataframe
    

    periods = np.array([0,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])

    def adaptiveTrendFinder_2(self,dataframe:DataFrame):

        dataframe['trend_direction'] = dataframe.apply((lambda x: self.calculate_trend_direction(x,dataframe)),axis=1)

        return dataframe['trend_direction']

    def calculate_trend_direction(self,x,dataframe):
        dataframe = dataframe[:x.name]

         # Variable Can Modify
        devMultiplier = 2.0
        # Calculate Deviation,PersionR,Slope,Intercept
        stdDev01, pearsonR01, slope01, intercept01 = self.calcDev(self.periods[1],dataframe)
        stdDev02, pearsonR02, slope02, intercept02 = self.calcDev(self.periods[2],dataframe)
        stdDev03, pearsonR03, slope03, intercept03 = self.calcDev(self.periods[3],dataframe)
        stdDev04, pearsonR04, slope04, intercept04 = self.calcDev(self.periods[4],dataframe)
        stdDev05, pearsonR05, slope05, intercept05 = self.calcDev(self.periods[5],dataframe)
        stdDev06, pearsonR06, slope06, intercept06 = self.calcDev(self.periods[6],dataframe)
        stdDev07, pearsonR07, slope07, intercept07 = self.calcDev(self.periods[7],dataframe)
        stdDev08, pearsonR08, slope08, intercept08 = self.calcDev(self.periods[8],dataframe)
        stdDev09, pearsonR09, slope09, intercept09 = self.calcDev(self.periods[9],dataframe)
        stdDev10, pearsonR10, slope10, intercept10 = self.calcDev(self.periods[10],dataframe)
        stdDev11, pearsonR11, slope11, intercept11 = self.calcDev(self.periods[11],dataframe)
        stdDev12, pearsonR12, slope12, intercept12 = self.calcDev(self.periods[12],dataframe)
        stdDev13, pearsonR13, slope13, intercept13 = self.calcDev(self.periods[13],dataframe)
        stdDev14, pearsonR14, slope14, intercept14 = self.calcDev(self.periods[14],dataframe)
        stdDev15, pearsonR15, slope15, intercept15 = self.calcDev(self.periods[15],dataframe)
        stdDev16, pearsonR16, slope16, intercept16 = self.calcDev(self.periods[16],dataframe)
        stdDev17, pearsonR17, slope17, intercept17 = self.calcDev(self.periods[17],dataframe)
        stdDev18, pearsonR18, slope18, intercept18 = self.calcDev(self.periods[18],dataframe)
        stdDev19, pearsonR19, slope19, intercept19 = self.calcDev(self.periods[19],dataframe)

        # Find the highest Pearson's R
        # float highestPearsonR = pearsonR01
        highestPearsonR = max(pearsonR01, pearsonR02, pearsonR03,
                               pearsonR04, pearsonR05, pearsonR06,
                                pearsonR07, pearsonR08, pearsonR09, 
                                pearsonR10, pearsonR11, pearsonR12,
                                pearsonR13, pearsonR14, pearsonR15, 
                                pearsonR16, pearsonR17, pearsonR18, 
                                pearsonR19
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
        elif highestPearsonR == pearsonR04:
            detectedPeriod = self.periods[4]  
            detectedSlope = slope04
            detectedIntrcpt = intercept04
            detectedStdDev = stdDev04
        elif highestPearsonR == pearsonR05:
            detectedPeriod = self.periods[5]  
            detectedSlope = slope05
            detectedIntrcpt = intercept05
            detectedStdDev = stdDev05
        elif highestPearsonR == pearsonR06:
            detectedPeriod = self.periods[6]       
            detectedSlope = slope06
            detectedIntrcpt = intercept06
            detectedStdDev = stdDev06
        elif highestPearsonR == pearsonR07:
            detectedPeriod = self.periods[7]      
            detectedSlope = slope07
            detectedIntrcpt = intercept07
            detectedStdDev = stdDev07
        elif highestPearsonR == pearsonR08:
            detectedPeriod = self.periods[8]       
            detectedSlope = slope08
            detectedIntrcpt = intercept08
            detectedStdDev = stdDev08
        elif highestPearsonR == pearsonR09:
            detectedPeriod = self.periods[9]       
            detectedSlope = slope09
            detectedIntrcpt = intercept09
            detectedStdDev = stdDev09
        elif highestPearsonR == pearsonR10:
            detectedPeriod = self.periods[10]
            detectedSlope = slope10
            detectedIntrcpt = intercept10
            detectedStdDev = stdDev10
        elif highestPearsonR == pearsonR11:
            detectedPeriod = self.periods[11]
            detectedSlope = slope11
            detectedIntrcpt = intercept11
            detectedStdDev = stdDev11
        elif highestPearsonR == pearsonR12:
            detectedPeriod = self.periods[12]
            detectedSlope = slope12
            detectedIntrcpt = intercept12
            detectedStdDev = stdDev12
        elif highestPearsonR == pearsonR13:
            detectedPeriod = self.periods[13]
            detectedSlope = slope13
            detectedIntrcpt = intercept13
            detectedStdDev = stdDev13
        elif highestPearsonR == pearsonR14:
            detectedPeriod = self.periods[14]
            detectedSlope = slope14
            detectedIntrcpt = intercept14
            detectedStdDev = stdDev14
        elif highestPearsonR == pearsonR15:
            detectedPeriod = self.periods[15]
            detectedSlope = slope15
            detectedIntrcpt = intercept15
            detectedStdDev = stdDev15
        elif highestPearsonR == pearsonR16:
            detectedPeriod = self.periods[16]
            detectedSlope = slope16
            detectedIntrcpt = intercept16
            detectedStdDev = stdDev16
        elif highestPearsonR == pearsonR17:
            detectedPeriod = self.periods[17]
            detectedSlope = slope17
            detectedIntrcpt = intercept17
            detectedStdDev = stdDev17
        elif highestPearsonR == pearsonR18:
            detectedPeriod = self.periods[18]
            detectedSlope = slope18
            detectedIntrcpt = intercept18
            detectedStdDev = stdDev18
        elif highestPearsonR == pearsonR19:
            detectedPeriod = self.periods[19]
            detectedSlope = slope19
            detectedIntrcpt = intercept19
            detectedStdDev = stdDev19
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

        if(len(dataframe) < 200):
            return 0,0,0,0

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