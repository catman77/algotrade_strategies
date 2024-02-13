import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
from freqtrade.exchange import timeframe_to_minutes
from datetime import datetime
import talib.abstract as ta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import StandardScaler
from technical.util import resample_to_interval, resampled_merge
from sklearn.model_selection import train_test_split

# This class is a sample. Feel free to customize it.
class SVMStrategy(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '3m'

    # Can this strategy go short?
    can_short: bool = True

    max_leverage = 20.0

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    timeframe_mins = timeframe_to_minutes(timeframe)
    # minimal_roi = {
    #     str(timeframe_mins * 1) : 0.1,
    #     str(timeframe_mins * 2) : 0.2,
    #     str(timeframe_mins * 3) : 0.0,    # 3% after 3 candles
    # }


    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

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

        # Add RSI,ROC,CCI and MOC indicators
        dataframe = self.add_talib_indicators(dataframe)
        # Calculate Actual Regression from future candle stick
        dataframe['actual_regression'] = np.nan_to_num(self.calculate_regression_value(dataframe))
        # Extract Features
        dataframe['feature_slow'] = np.nan_to_num(self.calculate_feature_1_slow("All",dataframe))
        dataframe['feature_fast'] = np.nan_to_num(self.calculate_feature_2_fast("All",dataframe))
        # Train ML Model and Predicted value to store df['predicted_regression']
        dataframe = self.trainModel(dataframe)
        # Filter Section
        dataframe = self.volatilifyFilter(dataframe)
        dataframe = self.volumeFilter(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['predicted_regression'] > 0)
                  &
                (dataframe['volatility_filter'] > 0) 
                &
                (dataframe['rf'] < 30)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['predicted_regression'] < 0)
                  &
                (dataframe['volatility_filter'] < 0)
                & 
                (dataframe['rf'] > 70)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['predicted_regression'] < 0)
                &
                (dataframe['volatility_filter'] < 0)
                & 
                (dataframe['rf'] > 70)
            ),

            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['predicted_regression'] > 0)
                &
                 (dataframe['volatility_filter'] < 0)
                & 
                (dataframe['rf'] < 30)
            ),
            'exit_short'] = 1

        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
                 
        entry_tag = ''

        return self.max_leverage    
    

    # Helper Function
    candle_number_to_calculate = 3

    def scale(self,x, p):
        min_val = x[-p:].min()
        max_val = x[-p:].max()

        scaled_column = (x - min_val) / (max_val - min_val)

        return scaled_column

    def add_talib_indicators(self,data):
        # RSI
        data['rs'] = ta.RSI(data['close'], timeperiod=28)
        data['rf'] = ta.RSI(data['close'], timeperiod=14)
        # ROC
        data['os'] = ta.ROC(data['close'], timeperiod=28)
        data['of'] = ta.ROC(data['close'], timeperiod=14)

        # CCI
        data['cs'] = ta.CCI(data['high'], data['low'], data['close'], timeperiod=28)
        data['cf'] = ta.CCI(data['high'], data['low'], data['close'], timeperiod=14)

        # MOM
        data['ms'] = self.scale(ta.MOM(data['close'], timeperiod=28),63) *100
        data['mf'] = self.scale(ta.MOM(data['close'], timeperiod=14),63) *100

        data['rs'] = np.nan_to_num(data['rs'])
        data['os'] = np.nan_to_num(data['os'])
        data['cs'] = np.nan_to_num(data['cs'])
        data['ms'] = np.nan_to_num(data['ms'])

        data['rf'] = np.nan_to_num(data['rf'])
        data['of'] = np.nan_to_num(data['of'])
        data['cf'] = np.nan_to_num(data['cf'])
        data['mf'] = np.nan_to_num(data['mf'])

        return data
    
    def calculate_feature_1_slow(self,ind,dataframe):

        if ind == 'RSI':
            return dataframe['rs']
        elif ind == 'ROC':
            return dataframe['os']
        elif ind == 'CCI':
            return dataframe['cs']
        elif ind == 'MOM':
            return dataframe['ms']
        else:
            # Assuming avg is a function that calculates the average
            return (dataframe['rs']+dataframe['os']+dataframe['cs']+dataframe['ms'])/4
        

    def calculate_feature_2_fast(self,ind,dataframe):

        if ind == 'RSI':
            return dataframe['rf']
        elif ind == 'ROC':
            return dataframe['of']
        elif ind == 'CCI':
            return dataframe['cf']
        elif ind == 'MOM':
            return dataframe['mf']
        else:
            # Assuming avg is a function that calculates the average
            return (dataframe['rf']+dataframe['of']+dataframe['cf']+dataframe['mf'])/4

    def calculate_regression_value(self,dataframe):
        dataframe['actual_regression'] = -1
        condition = (dataframe['close'].shift(-self.candle_number_to_calculate) - dataframe['close']) > 0
        dataframe.loc[condition, 'actual_regression'] = 1
        return dataframe['actual_regression']
    
    def trainModel(self,df):
        # Generate some sample data
        filter_index = df[(df['feature_slow'] == 0) | (df['feature_fast'] == 0)].index

        X = df.iloc[:-self.candle_number_to_calculate, df.columns.isin(['feature_slow', 'feature_fast','close','rs'])]
        # Drop X 0 and last 3 candle
        X = X.drop(filter_index)
        X = X.iloc[:, X.columns.isin(['feature_slow', 'feature_fast','close','rs'])]

        y = df.iloc[:-self.candle_number_to_calculate, df.columns.isin(['actual_regression'])]
        # Drop Y 0 and last 3 candle
        y = y.drop(filter_index)
        y = y.iloc[:, y.columns.isin(['actual_regression'])]

        test_data = df[['feature_slow', 'feature_fast','close','rs']]

        # Convert X_train and X_test to NumPy arrays
        X = np.array(X)
        y = np.ravel(y)
        test_data = np.array(test_data)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        X_test_std = scaler.transform(test_data)

        X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.1, random_state=42)

        # Create SVM classifier
        svm_classifier = SVC(
                            kernel='rbf',
                            C=100,
                            gamma='auto',
                            random_state=42,
                            probability=True,
                            cache_size=1,
                            )
        
        # Ensemble Model
        rf_model = RandomForestClassifier(n_estimators=100, 
                                          random_state=42,
                                          min_samples_leaf=50,
                                          oob_score=True,
                                          )

        ensemble_model = VotingClassifier(estimators=[
                                                    ('svm', svm_classifier), 
                                                    ('rf', rf_model),
                                                    ], voting='hard')

        # Train the classifier
        ensemble_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = ensemble_model.predict(X_test_std)
        df['predicted_regression'] = y_pred

        return df
    
    def volatilifyFilter(self,dataframe):
        # Define BB windows 
        bb_window = 20
        bb_dev = 2.0

        # Calculate Bollinger Bands
        upper_band , middle_band, lower_band = ta.BBANDS(dataframe['close'],timeperiod=bb_window,nbdevup=bb_dev,nbdevdn=bb_dev)

        # Calculate ATR

        # Define threshold for atr
        dataframe.loc[
            (
                (dataframe['close'] < lower_band)
            ),
            'volatility_filter'] = 1
        
        dataframe.loc[
            (
                (dataframe['close'] > upper_band)
            ),
            'volatility_filter'] = -1
        
        return dataframe
    
    def volumeFilter(self,dataframe):
        dataframe['hma_rsi'] = ta.WMA(ta.WMA(dataframe['rf'], timeperiod=10), timeperiod=10)
        return dataframe
