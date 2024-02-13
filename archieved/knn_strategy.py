# --- Do not remove these libs ---
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
import pandas as pd
import talib
import numpy as np
import math
from datetime import datetime,timedelta
import matplotlib.pyplot as plt


# This class is a sample. Feel free to customize it.
class KNNStrategy(IStrategy):
    """
    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

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
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core logic of the algorithm
        k = 63  # K value
        holding_period = 1
        time_threshold = 99.9
        filter_type = "Both"
        feature1 = []
        feature2 = []
        directions = []
        predictions = []
        prediction = 0
        signal = 0
        hp_counter = 0

        to_stop_count = 0

        # Fetch Data
        dataframe = dataframe
        dataframe = add_talib_indicators(dataframe)

        for candle_index,current_candle in dataframe.iterrows():
            to_stop_count += 1
            # First Let's Add Indicator To Extract Feature and Directions
            # Calculate the 'class' column for Direction
            # Let's Extract Feature and Directions For Each Candle Stick Appear
            f1 = calculate_feature_1_slow("All",current_candle)
            f2 = calculate_feature_2_fast("All",current_candle)

            dataframe = calculate_directions(dataframe)
            
            current_direction = dataframe.loc[candle_index,'class']
            feature1.append(f1)
            feature2.append(f2)
            directions.append(current_direction)

            # Core logic of the kNN algorithm
            size = len(directions)
            maxdist = -999.0

            for i in range(size):
                # Calculate the Euclidean distance
                d = np.sqrt(np.power(f1 - feature1[i], 2) + np.power(f2 - feature2[i], 2))
                if d > maxdist:
                    maxdist = d
                    if len(predictions) >= k:
                        predictions.pop(0)
                    predictions.append(directions[i])

            prediction = np.sum(predictions)
            # Example timestamps (replace these with your actual timestamps)
            current_timestamp = dataframe.iloc[candle_index].date
            time = pd.to_datetime(current_timestamp)
            print(f"====================================================={current_timestamp}=========================")
            prev_timestamp = subtract_interval(time, self.timeframe)
            time_close = pd.to_datetime(prev_timestamp)
            timenow = pd.to_datetime('now')

            # Convert Timestamp objects to Unix timestamps (seconds since the epoch)
            time_unix = time.timestamp()
            time_close_unix = time_close.timestamp()
            timenow_unix = timenow.timestamp()

            # Calculate tbase, tcurr, and barlife
            tbase = (time_unix - time_close_unix) / 1000
            tcurr = (timenow_unix - time_close_unix) / 1000
            barlife = tcurr / tbase

            # Signal generation based on prediction and filter conditions
            filter_condition = True

            if filter_type == 'Volatility':
                filter_condition = volatility_break(1,10,dataframe,candle_index)
            elif filter_type == 'Volume':
                filter_condition = volume_break(49, dataframe,candle_index)
            elif filter_type == 'Both':
                filter_condition = volatility_break(1,10,dataframe,candle_index) and volume_break(49, dataframe,candle_index)
            else:
                filter_condition = True

            signal = 1 if prediction > 0 and barlife > time_threshold and filter_condition else \
                -1 if prediction < 0 and barlife > time_threshold and filter_condition else 0
            dataframe.loc[current_timestamp,'Signal'] = signal
            changed = 0

            try:
                # Code that might raise an exception
                changed = dataframe.loc[current_timestamp,'Signal'] - dataframe.loc[prev_timestamp,'Signal']
            except Exception as e:
                # Code to execute if no exception occurred
                changed = 0
            else:
                # Code to execute regardless of whether an exception occurred
                changed = dataframe.loc[current_timestamp,'Signal'] - dataframe.loc[prev_timestamp,'Signal']

            startLongTrade  = changed and signal==1
            startShortTrade = changed and signal==-1
            end_long_trade = (changed and signal == -1) or (signal == 1 and hp_counter == holding_period and not changed)
            end_short_trade = (changed and signal == 1) or (signal == -1 and hp_counter == holding_period and not changed)
        
            dataframe.loc[current_timestamp,'StartLong'] = startLongTrade
            dataframe.loc[current_timestamp,'StartShort'] = startShortTrade
            dataframe.loc[current_timestamp,'EndLong'] = end_long_trade
            dataframe.loc[current_timestamp,'EndShort'] = end_short_trade
            print(startLongTrade,startShortTrade,end_long_trade,end_short_trade)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['StartLong'] == True) 
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['StartShort'] == True) 
            ),
            'enter_short'] = 1
        print(dataframe.loc[
            (
                (dataframe['StartLong'] == True) 
            ),
            'enter_long'])
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['EndLong'] == True)
            ),

            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['EndShort'] == True)
            ),
            'exit_short'] = 1

        return dataframe












# ======================== Helper =================================
def scale(x, p):
    min_val = x[-p:].min()
    max_val = x[-p:].max()

    scaled_column = (x - min_val) / (max_val - min_val)

    return scaled_column

def add_talib_indicators(data):
    # RSI
    data['rs'] = talib.RSI(data['close'], timeperiod=28)
    data['rf'] = talib.RSI(data['close'], timeperiod=14)
    # ROC
    data['os'] = talib.ROC(data['close'], timeperiod=28)
    data['of'] = talib.ROC(data['close'], timeperiod=14)

    # CCI
    data['cs'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=28)
    data['cf'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=14)

    # MOM
    data['ms'] = scale(talib.MOM(data['close'], timeperiod=28),63) *100
    data['mf'] = scale(talib.MOM(data['close'], timeperiod=14),63) *100

    data['rs'] = np.nan_to_num(data['rs'])
    data['os'] = np.nan_to_num(data['os'])
    data['cs'] = np.nan_to_num(data['cs'])
    data['ms'] = np.nan_to_num(data['ms'])

    data['rf'] = np.nan_to_num(data['rf'])
    data['of'] = np.nan_to_num(data['of'])
    data['cf'] = np.nan_to_num(data['cf'])
    data['mf'] = np.nan_to_num(data['mf'])

    return data


ind = "All"
# Constants
BUY = 1
SELL = -1
HOLD = 0

# Only Return Single RSI for current time
def calculate_feature_1_slow(ind,current_candle):

    if ind == 'RSI':
        return current_candle['rs']
    elif ind == 'ROC':
        return current_candle['os']
    elif ind == 'CCI':
        return current_candle['cs']
    elif ind == 'MOM':
        return current_candle['ms']
    else:
        # Assuming avg is a function that calculates the average
        return np.average([current_candle['rs'],current_candle['os'],current_candle['cs'],current_candle['ms']])

def calculate_feature_2_fast(ind,current_candle):

    if ind == 'RSI':
        return current_candle['rf']
    elif ind == 'ROC':
        return current_candle['of']
    elif ind == 'CCI':
        return current_candle['cf']
    elif ind == 'MOM':
        return current_candle['mf']
    else:
        # Assuming avg is a function that calculates the average
        return np.average([current_candle['rf'],current_candle['of'],current_candle['cf'],current_candle['mf']])

def calculate_directions(data):
  conditions = [
      data['close'].shift(1) < data['close'],
      data['close'].shift(1) > data['close'],
      data['close'].shift(1) == data['close']
  ]

  choices = [-1, 1, 0]

  data['class'] = np.select(conditions, choices, default=0)

  return data

# Time calculation for interval like 15m 1m 3m
def subtract_interval(timestamp, interval_str):
    # Parse the interval string to determine the timedelta
    interval_parts,unit = extract_interval_parts(interval_str)
    value = int(interval_parts)

    # Define the timedelta based on the provided interval
    if unit == 'm':
        delta = pd.Timedelta(minutes=value)
    elif unit == 'h':
        delta = pd.Timedelta(hours=value)
    elif unit == 'd':
        delta = pd.Timedelta(days=value)
    else:
        raise ValueError("Unsupported interval unit")

    # Subtract the interval from the timestamp
    result_timestamp = timestamp - delta

    return result_timestamp

def volume_break(thres, data,candle_index):
    # Calculate RSI for volume
    rsivol = talib.RSI(data.loc[:candle_index,'volume'], timeperiod=14)

    # Calculate Hull Moving Average (HMA) for RSI
    hma_rsi = talib.WMA(talib.WMA(rsivol, timeperiod=round(10 / 2)), timeperiod=round(math.sqrt(10)))

    # Check if HMA(RSI) is greater than the threshold
    return np.nan_to_num(hma_rsi.loc[candle_index]) > thres

def calculate_atr(high, low, close, period):
    atr_values = []
    for i in range(1, len(close)):
        true_range = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr_values.append(true_range)

    atr = sum(atr_values[-period:]) / period
    return atr

def volatility_break(volmin, volmax, data,candle_index):
    # Calculate ATR for volmin and volmax
    atr_volmin = calculate_atr(data.loc[:candle_index,'high'],data.loc[:candle_index,'low'],data.loc[:candle_index,'close'],volmin)
    atr_volmax = calculate_atr(data.loc[:candle_index,'high'],data.loc[:candle_index,'low'],data.loc[:candle_index,'close'],volmax)

    # Check if ATR for volmin is greater than ATR for volmax
    return atr_volmin > atr_volmax

def extract_interval_parts(interval_str):
    # Find the numerical part and the unit part using isdigit and isalpha
    numerical_part = ''
    unit_part = ''

    for char in interval_str:
        if char.isdigit():
            numerical_part += char
        elif char.isalpha():
            unit_part += char

    # Convert numerical part to an integer (if it's not empty)
    numerical_part = int(numerical_part) if numerical_part else None

    return numerical_part, unit_part