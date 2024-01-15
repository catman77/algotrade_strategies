import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import math
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


# This class is a sample. Feel free to customize it.
class TestStrategy(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.6

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    # Strategy Constant
    periods = np.array([0,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        # For Backtesting
        dataframe = self.adaptiveTrendFinder_2(dataframe)
        dataframe['trend'] = dataframe['trend_direction'].apply(lambda x: x[0])
        dataframe['trend_strength'] = dataframe['trend_direction'].apply(lambda x: x[2])


        # For Live Trading
        # trend_direction,detectedPeriod,highestPearsonR = self.adaptiveTrendFinder(dataframe)
        # dataframe['trend'] = trend_direction
        # dataframe['trend_strength'] = highestPearsonR
        # KNN Strategy
        dataframe = self.trainModelML(dataframe)

        # print("=======================")
        print(dataframe.loc[:,['trend','trend_strength']])
        print(datetime.now(timezone.utc))
        print("=====================")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trend'] > 0)
                &
                (dataframe['trend_strength'] >= 0.90) 
                &
                (dataframe['predicted_value'] > 0)
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['trend'] < 0) 
                &
                (dataframe['trend_strength'] >= 0.90)
                &
                (dataframe['predicted_value'] < 0)
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trend'] < 0) 
                &
                (dataframe['trend_strength'] >= 0.90)
            ),
            'exit_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['trend'] > 0)
                &
                (dataframe['trend_strength'] >= 0.90)
            ),
            'exit_short'] = 1
        return dataframe
    


    # Helper Function
    def adaptiveTrendFinder(self,dataframe:DataFrame):
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
        highestPearsonR = max(pearsonR01, pearsonR02, pearsonR03, pearsonR04, pearsonR05, pearsonR06, pearsonR07, pearsonR08, pearsonR09, pearsonR10, pearsonR11, pearsonR12, pearsonR13, pearsonR14, pearsonR15, pearsonR16, pearsonR17, pearsonR18, pearsonR19)

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

        return trend_direction,detectedPeriod,highestPearsonR
    

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
    
    def minimax(self,volume,x,p,min,max):
        volume_array = volume.to_numpy()
        volume_array = volume_array[:x.name]

        if(len(volume_array) < p):
            return 0
    
        hi = np.nan_to_num(np.max(volume_array[-p:]))
        lo = np.nan_to_num(np.min(volume_array[-p:]))

        return (max - min) * (volume_array[len(volume_array)-1] - lo)/(hi - lo) + min
    
    def get_class_label(self,x,close):
        close_array = close.to_numpy()
        close_array = close_array[:x.name]

        if(len(close_array) < 1):
            return 1
        
        current_value =  close_array[len(close_array) - 1] 
        previous_value = close_array[len(close_array) - 2]
        
        if(previous_value > current_value):
            return 1
        elif(previous_value < current_value):
            return -1
        else:
            return 1
    
    def get_long_or_short_prediction(self,index,predicte_list):
        predicte_array = predicte_list.to_numpy()
        predicte_array = predicte_array[:index.name]

        if(len(predicte_array) < 1):
            return 0
        
        current_value =  predicte_array[len(predicte_array) - 1] 
        previous_value = predicte_array[len(predicte_array) - 2]
        return current_value - previous_value
    
    def trainModelML(self,dataframe):
            BaseK = 252
            LongWindow = 28
            ShortWindow = 14

            # 3paris of predictor indicators , long,short each
            dataframe['rs'] = ta.RSI(dataframe['close'],timeperiod=LongWindow)
            dataframe['cs'] = ta.CCI(dataframe['high'],dataframe['low'],dataframe['close'],timeperiod=LongWindow)
            dataframe['os'] = ta.ROC(dataframe['close'],timeperiod=LongWindow)
            dataframe['vs'] = dataframe.apply((lambda x: self.minimax(dataframe['volume'],x,LongWindow,0,99)),axis=1)

            dataframe['rf'] = ta.RSI(dataframe['close'],timeperiod=ShortWindow)
            dataframe['cf'] = ta.CCI(dataframe['high'],dataframe['low'],dataframe['close'],timeperiod=ShortWindow)
            dataframe['of'] = ta.ROC(dataframe['close'],timeperiod=ShortWindow)
            dataframe['vf'] = dataframe.apply((lambda x: self.minimax(dataframe['volume'],x,ShortWindow,0,99)),axis=1)

            dataframe['f1'] = dataframe.loc[:,['rs','cs','os','vs']].mean(axis=1)
            dataframe['f2'] = dataframe.loc[:,['rf','cf','of','vf']].mean(axis=1)

            # Classification data, what happens on the next bar

            dataframe['class_label'] = dataframe.apply((lambda x: self.get_class_label(x,dataframe['close'])),axis=1)

            # Loop Through Training Arrays and get distances
            dataframe['predicted_value'] = dataframe.apply((lambda x : self.getPredictedValueFromKNN(x,dataframe)),axis=1)
            return dataframe
    
    def getPredictedValueFromKNN(self,x,dataframe):
        maxdist = -999.0
        k = math.floor(math.sqrt(252))
        predictions = []

        feature_array_1 = dataframe['f1'].to_numpy()
        feature_array_1 = feature_array_1[:x.name]

        feature_array_2 = dataframe['f2'].to_numpy()
        feature_array_2 = feature_array_2[:x.name]

        direction_array = dataframe['class_label'].to_numpy()
        direction_array = direction_array[:x.name]

        if(len(feature_array_1) < 1):
            return 0
        elif(len(feature_array_2) < 1):
            return 0
        
        current_feature1 =  feature_array_1[len(feature_array_1) - 1] 
        current_feature2 =  feature_array_2[len(feature_array_2) - 1] 
        size = len(feature_array_1)
        for i in range(0,size-1):
            d = math.sqrt(math.pow(current_feature1 - feature_array_1[i], 2) + math.pow(current_feature2 - feature_array_2[i], 2))
            if(d > maxdist):
                maxdist = d
                if(len(predictions) >= k):
                    predictions.pop()
                predictions.append(direction_array[i])

        prediction = np.sum(predictions)
        return prediction
    

    def adaptiveTrendFinder_2(self,dataframe:DataFrame):

        dataframe['trend_direction'] = dataframe.apply((lambda x: self.calculate_trend_direction(x,dataframe)),axis=1)

        return dataframe
    
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
        highestPearsonR = max(pearsonR01, pearsonR02, pearsonR03, pearsonR04, pearsonR05, pearsonR06, pearsonR07, pearsonR08, pearsonR09, pearsonR10, pearsonR11, pearsonR12, pearsonR13, pearsonR14, pearsonR15, pearsonR16, pearsonR17, pearsonR18, pearsonR19)

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