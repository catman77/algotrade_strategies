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

class UTBotStrategy(IStrategy):

    INTERFACE_VERSION = 3
    use_custom_stoploss = True

    can_short: bool = True

    # TODO Adjust this parameter
    minimal_roi = {
        "0": 0.2
    }
    stoploss = -1

    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True

    startup_candle_count: int = 60

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Strategy Constant
    # periods = np.array([0,3,3,4,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6])
    # periods = np.array([0,3,4,5])
    periods = np.array([0,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Start AI
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["ATRTrailingStop"] , dataframe["Ema"] , dataframe['UT_Signal'] = self.calculate_ut_bot(dataframe)
        dataframe['sar'] = ta.SAR(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        # For Backtesting
        # dataframe[f'trend_direction'] = self.adaptiveTrendFinder_2(dataframe)
        # dataframe[f'trend'] = dataframe['trend_direction'].apply(lambda x: x[0])
        # dataframe[f'trend-period'] = dataframe['trend_direction'].apply(lambda x: x[1])
        # dataframe[f'trend-strength'] = dataframe['trend_direction'].apply(lambda x: x[2])
        print(dataframe.loc[:,['UT_Signal','sar','macd','macdsignal','close']])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1)
                &
                (dataframe["&s-up_or_down"] == "up")
                # &
                # (dataframe['macd'] > dataframe['macdsignal'])
                # &
                (dataframe['UT_Signal'] > 0)
                # &
                # (dataframe['sar'] < dataframe['close'])
                # &
                # (dataframe['trend'] > 0)
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1)
                &
                (dataframe["&s-up_or_down"] == "down")
                # &
                # (dataframe['macd'] < dataframe['macdsignal'])
                # &
                (dataframe['UT_Signal'] < 0)
                # &
                # (dataframe['sar'] > dataframe['close'])
                # &
                # (dataframe['trend'] < 0)
            ),
            'enter_short'] = 1
        return dataframe
    
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
                (dataframe["do_predict"] == 1)
                &
                (dataframe["&s-up_or_down"] == "down")
                &
                (dataframe['UT_Signal'] < 0)
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1)
                &
                (dataframe["&s-up_or_down"] == "up")
                &
                (dataframe['UT_Signal'] > 0)
            ),
            'exit_short'] = 1
        return dataframe
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Sell any positions at a loss if they are losing in 10 minutes.
        if current_profit < 0 and ((current_time - trade.open_date_utc).seconds >= 300):
            return 'unclog'
        
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        
        if  trade.calc_profit_ratio(rate) < 0 and (current_time - trade.open_date_utc).seconds >= 150:
            return True
        if trade.calc_profit_ratio(rate) < 0:
            return False
        return True
    
    def set_freqai_targets(self, dataframe, **kwargs) -> DataFrame:
        # dataframe["&-action"] = 0
        dataframe['&s-up_or_down'] = np.where( dataframe["close"].shift(-3) >
                                        dataframe["close"], 'up', 'down')
        return dataframe
    
    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        # The following features are necessary for RL models
        dataframe[f"%-raw_close"] = dataframe["close"]
        dataframe[f"%-raw_open"] = dataframe["open"]
        dataframe[f"%-raw_high"] = dataframe["high"]
        dataframe[f"%-raw_low"] = dataframe["low"]
        dataframe[f"%-raw-volume"] = dataframe["volume"]
        dataframe[f"%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe[f"%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25

        # Bollinger Bands
        # bollinger = qtpylib.bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=6, stds=2
        # )
        # dataframe["%bb_lowerband-period"] = bollinger["lower"]
        # dataframe["%bb_middleband-period"] = bollinger["mid"]
        # dataframe["%bb_upperband-period"] = bollinger["upper"]

        # dataframe["%-bb_width-period"] = (
        #     dataframe["%bb_upperband-period"]
        #     - dataframe["%bb_lowerband-period"]
        # ) / dataframe["%bb_middleband-period"]

        # dataframe['%sar'] = ta.SAR(dataframe)

        # # Hammer: values [0, 100]
        # dataframe['%CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe['%CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe['%CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['%CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['%CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['%CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]
        # dataframe['%CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # dataframe['%CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # dataframe['%CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # dataframe['%CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # dataframe['%CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # dataframe['%CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)
        # dataframe['%CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # dataframe['%CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe)
        # dataframe['%CDLENGULFING'] = ta.CDLENGULFING(dataframe)
        # dataframe['%CDLHARAMI'] = ta.CDLHARAMI(dataframe)
        # dataframe['%CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe)
        # dataframe['%CDL3INSIDE'] = ta.CDL3INSIDE(dataframe)

        dataframe['%sma3'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['%sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['%sma10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%sma21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['%sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['%sma100'] = ta.SMA(dataframe, timeperiod=100)

        dataframe['%ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['%ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['%ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['%ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['%ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['%ema100'] = ta.EMA(dataframe, timeperiod=100)
        # dataframe['%mfi'] = ta.MFI(dataframe)  

        # MACD Strategy
        # dataframe = self.calculateFilter(dataframe)

        # UT BOT
        # dataframe[f"%-ATRTrailingStop"] = dataframe['ATRTrailingStop']
        # dataframe[f"%-Ema"] = dataframe['Ema']
        return dataframe
    

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:

        return 5
    

    # Helper Function
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

    def calculate_ut_bot(self,dataframe):
        # UT Bot Parameters
        SENSITIVITY = 1
        ATR_PERIOD = 10

        dataframe = self.heikinashi(dataframe)

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
        dataframe["Above"] = (dataframe['Ema'] > dataframe["ATRTrailingStop"])
        dataframe["Below"] = (dataframe['Ema'] < dataframe["ATRTrailingStop"])

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

        return dataframe["ATRTrailingStop"] , dataframe['Ema'] , dataframe['UT_Signal']


    def calculate_pip(self, price, initial_price , final_price):
        pip_percentage = ((final_price - initial_price) / initial_price) * 100
        pip_value = price * (pip_percentage / 100)
        return pip_value

    def calculate_vortex(self, dataframe,atr_value ,index, period):
        high = dataframe['high'].to_numpy()
        low = dataframe['low'].to_numpy()

        if(index.name < period):
            return 0
        VMP = 0
        VMM = 0
        STR = 0
        for i in range(index.name,index.name-period,-1):
            VMP = VMP + abs(high[i] - low[i-1])

        for i in range(index.name,index.name-period,-1):
            VMM = VMM + abs(low[i] - high[i-1])

        for i in range(index.name,index.name-period,-1):
            STR = STR + atr_value[i]

        VIP = VMP / STR
        VIM = VMM / STR
        
        return (VIP,VIM)
    
    def calculateVortexFilter(self,dataframe,index):
        # If VIM(VI -) is 1 , 0.9 , 0.7
        # If VIP(VI +) is 1 , 1.1 , 1.3
        vortex_value_array = dataframe['vortex_value'].to_numpy()
        # Format Will be (VIP,VIM)
        if(index.name < 2):
            return 0
        current_vortex_value = vortex_value_array[index.name]
        previous_vortex_value = vortex_value_array[index.name-1]
        previous_previous_vortex_value = vortex_value_array[index.name - 2]

        # current_vortex_diff = current_vortex_value - previous_vortex_value
        # previous_vortex_diff = previous_vortex_value - previous_previous_vortex_value
        
        # If Current VIM - Previous VIM is + this is called current_vortex_diff
        # Previous VIM - Previous Previous VIM is - this is called previous_vortex_diff
        # Compare current_vortex_diff and previous_vortex_diff return greater vortex

        return 0

    def calculateHMA(self,dataframe,period):
        # Calculate weighted moving average with half the period
        wma_half_period = dataframe['rsi'].rolling(window=period // 2).mean()

        # Calculate weighted moving average with the full period
        wma_full_period = dataframe['rsi'].rolling(window=period).mean()

        # Calculate the final HMA
        hma = pd.Series(2 * wma_half_period - wma_full_period).rolling(window=int(np.sqrt(period))).mean()

        return hma

    def calculateFilter(self,dataframe):
        # Calculate ATR VALUE FOR 10 and 40
        dataframe['atr_low'] = ta.ATR(dataframe['high'],dataframe['low'],dataframe['close'],timeperiod=1)
        dataframe['atr_high'] = ta.ATR(dataframe['high'],dataframe['low'],dataframe['close'],timeperiod=10)
        dataframe['atr_filter'] = dataframe['atr_low'] > dataframe['atr_high']

        # volumeBreakThreshold = 47
        # dataframe['rsi'] = ta.RSI(dataframe['volume'],timeperiod=14)
        # dataframe['osc'] = self.calculateHMA(dataframe,10)
        # dataframe['volume_filter'] = dataframe['osc'] > volumeBreakThreshold

        # dataframe['atr_value'] = ta.ATR(dataframe['high'],dataframe['low'],dataframe['close'],timeperiod=1)
        # atr_value = dataframe['atr_value'].to_numpy()
        # dataframe['vortex_value'] = dataframe.apply((lambda index : self.calculate_vortex(dataframe,atr_value,index,14)),axis=1)
        # dataframe['vortex_filter'] = dataframe.apply((lambda index : self.calculateVortexFilter(dataframe,index)),axis=1)
        return dataframe
    

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