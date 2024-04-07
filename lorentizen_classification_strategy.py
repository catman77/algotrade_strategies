from decimal import Decimal
from functools import reduce
import math
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.exchange.exchange_utils import timeframe_to_prev_date
from freqtrade.persistence import Trade
from typing import Optional
from freqtrade.strategy import (IStrategy)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from freqtrade.strategy.parameters import IntParameter
from datetime import datetime, timedelta
from ml_helper import mlRunModel, FeatureName, Filter


class LorentizenStrategy(IStrategy):

    # =============================================================
    # ===================== Strategy Config =======================
    # =============================================================
    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = '5m'
    stoploss = -0.1
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 50
    leverage_value = 20
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    training_params = {
        # done
        "filter_method": [Filter.regime, Filter.volatility, Filter.kernel, Filter.sma, Filter.ema],
        "filter_params": {
            "kernel": {
                "loopback": 8,
                "weighting": 8,
                "regression_level": 25
            },
            "regime": {
                "threshold": -0.1
            },
            "ema": {
                "threshold": 200
            },
            "sma": {
                "threshold": 200
            }
        },
        "neighbor_count": 8,  # done
        "feature_count": 5,  # done
        "future_count": 4,  # done
        "f1": {
            "name": FeatureName.rsi,
            "paramsA": 14,
            "paramsB": 2
        },
        "f2": {
            "name": FeatureName.wt,
            "paramsA": 10,
            "paramsB": 11
        },
        "f3": {
            "name": FeatureName.cci,
            "paramsA": 20,
            "paramsB": 2
        },
        "f4": {
            "name": FeatureName.adx,
            "paramsA": 20,
            "paramsB": 2
        },
        "f5": {
            "name": FeatureName.rsi,
            "paramsA": 9,
            "paramsB": 2
        },
    }

    # =============================================================
    # ===================== LorentizenStrategy =================
    # =============================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.unlock_pair(metadata['pair'])
        a = datetime.now()
        dataframe = self.lorentizen_train(dataframe)
        b = datetime.now()
        print(f'{(b-a).microseconds * 0.001} ms')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_condition = dataframe['buy_signal'] == True
        sell_condition = dataframe['sell_signal'] == True
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
        buy_condition = dataframe['buy_signal'] == True
        sell_condition = dataframe['sell_signal'] == True

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
            if (self.trade_candle_previous_min_and_max.keys().__contains__(pair) == False):
                self.trade_candle_previous_min_and_max[pair] = {
                    "min": 0, "max": 0}

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
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-2].squeeze()
        previous_last_candle_max_value = max(
            last_candle['open'], last_candle['high'], last_candle['low'], last_candle['close'])
        previous_last_candle_min_value = min(
            last_candle['open'], last_candle['high'], last_candle['low'], last_candle['close'])

        if (side == "long"):
            deviation_of_previous_candle = self.calculate_deviation(
                rate, previous_last_candle_max_value)
        else:
            deviation_of_previous_candle = self.calculate_deviation(
                rate, previous_last_candle_min_value)

        price_to_exit = self.add_deviation(
            rate, (side == "short"), deviation_of_previous_candle)
        profit_ratio = self.calculate_profit_ratio(open_rate=rate, exit_rate=price_to_exit, is_short=(
            side == "short"), leverage=self.leverage_value, stake_ammount=10)
        should_trade_by_checking_previous = profit_ratio > 3

        if (side == "long") & is_best_time & should_trade_by_checking_previous:
            self.trade_candle_previous_min_and_max[pair] = {
                "min": previous_last_candle_min_value, "max": previous_last_candle_max_value}
            return True
        if (side == "short") & is_best_time & should_trade_by_checking_previous:
            self.trade_candle_previous_min_and_max[pair] = {
                "min": previous_last_candle_min_value, "max": previous_last_candle_max_value}
            return True
        return False

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        if (sell_reason == 'exit_signal'):
            self.trade_candle_previous_min_and_max[pair] = {"min": 0, "max": 0}
            return False
        if (sell_reason == 'stop_loss'):
            self.trade_candle_previous_min_and_max[pair] = {"min": 0, "max": 0}
            return False
        if (sell_reason == 'swp'):
            self.trade_candle_previous_min_and_max[pair] = {"min": 0, "max": 0}
            return True
        if (sell_reason == 'swl'):
            self.trade_candle_previous_min_and_max[pair] = {"min": 0, "max": 0}
            return True
        return False

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        prediction_future_count = self.training_params['future_count']
        timeframe_int = int(self.timeframe[:-1])
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_date = timeframe_to_prev_date(
            self.timeframe, trade.open_date_utc)
        trade_date_previous = timeframe_to_prev_date(
            self.timeframe, (current_time - timedelta(minutes=timeframe_int*(prediction_future_count))))
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        trade_candle_previous_n = dataframe.loc[dataframe['date']
                                                == trade_date_previous]

        if (not trade_candle.empty) & (not trade_candle_previous_n.empty):
            trade_candle = trade_candle.squeeze()
            trade_candle_previous_n = trade_candle_previous_n.squeeze()

            # Logic To Exit (when start trade bar is 0 - 4)
            isHeldFourBars = current_time >= (
                trade_date + timedelta(minutes=timeframe_int*prediction_future_count))
            isLastSignalBuy = trade_candle_previous_n['final_prediction'] == 1
            isLastSignalSell = trade_candle_previous_n['final_prediction'] == -1
            isHeldLessThanFourBars = self.is_between(
                trade_date, trade_date + timedelta(minutes=timeframe_int*prediction_future_count), current_time)
            isNewBuySignal = last_candle['buy_signal']
            isNewSellSignal = last_candle['sell_signal']
            startLongTrade_previous = trade_candle_previous_n['buy_signal']
            startShortTrade_previous = trade_candle_previous_n['sell_signal']
            isProfit = (isHeldLessThanFourBars & (current_profit < 0.1))

            endLongTradeStrict = (isProfit | (isHeldFourBars & isLastSignalBuy) | (
                isHeldLessThanFourBars & isNewSellSignal & isLastSignalBuy)) & startLongTrade_previous
            endShortTradeStrict = (isProfit | (isHeldFourBars & isLastSignalSell) | (
                isHeldLessThanFourBars & isNewBuySignal & isLastSignalSell)) & startShortTrade_previous

            if (endLongTradeStrict):
                return "swp"
            elif (endShortTradeStrict):
                return "swl"

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_value

    # =============================================================
    # ===================== Strategy Helper =======================
    # =============================================================
    def is_between(self, start_time, end_time, currentTimeStamp):
        current_time = currentTimeStamp
        return start_time <= current_time <= end_time

    def is_quarter_hour(self, time: datetime):
        seconds = time.second
        # minute = time.minute
        #  & (minute in [0,5,10,15,20,25,30,35,40,45,50,55])
        return (seconds <= 2) & (seconds >= 0)

    def calculate_deviation(self, num1, num2):
        diff = abs(Decimal(str(num1)) - Decimal(str(num2)))
        diff_str = str(diff).lstrip('0').replace('.', '')
        deviation_result = int(diff_str)
        return deviation_result

    def count_digits_in_float(self, num):
        if isinstance(num, float):
            num_str = str(num)
            integer_part, _, fractional_part = num_str.partition('.')
            return len(integer_part) + len(fractional_part)
        else:
            return 0

    def add_deviation(self, number, isShort, deviation):
        num_str = str(number)
        integer_part, decimal_part = num_str.split('.')
        last_three_digits = decimal_part[-3:]
        last_three_digits_int = int(last_three_digits)
        new_last_three_digits = 0
        if isShort == False:
            new_last_three_digits = str(
                last_three_digits_int + deviation).zfill(3)
        if isShort == True:
            new_last_three_digits = str(
                last_three_digits_int - deviation).zfill(3)
        new_last_three_digits = new_last_three_digits.replace("-", "")
        new_decimal_part = decimal_part[:-3] + new_last_three_digits
        new_number_str = integer_part + '.' + new_decimal_part
        new_number = float(new_number_str)
        return new_number

    def addZeroToLastDigit(self, num: float):
        str_num = str(num) + '0'
        return str_num

    def calculate_profit_ratio(self, open_rate, exit_rate, is_short, leverage, stake_ammount):
        quantity = (stake_ammount*leverage)/open_rate
        initial_margin = quantity * open_rate * (1/leverage)
        pnl = 0
        roi = 0
        if (is_short == False):
            pnl = (exit_rate - open_rate) * quantity
        else:
            pnl = (open_rate - exit_rate) * quantity

        roi = pnl / initial_margin
        return round(roi * 100, 2)

    # =============================================================
    # ===================== Heikinashi ============================
    # =============================================================
    def heikinashi(self, df: pd.DataFrame) -> pd.DataFrame:
        df_HA = df.copy()
        df_HA['close'] = (df_HA['open'] + df_HA['high'] +
                          df_HA['low'] + df_HA['close']) / 4

        for i in range(0, len(df_HA)):
            if i == 0:
                df_HA.loc[i, 'open'] = (
                    (df_HA.loc[i, 'open'] + df_HA.loc[i, 'close']) / 2)
            else:
                df_HA.loc[i, 'open'] = (
                    (df_HA.loc[i-1, 'open'] + df_HA.loc[i-1, 'close']) / 2)

        df_HA['high'] = df_HA[['open', 'close', 'high']].max(axis=1)
        df_HA['low'] = df_HA[['open', 'close', 'low']].min(axis=1)

        return df_HA

    # =============================================================
    # ============== Machine Learning Model =======================
    # =============================================================
    def lorentizen_train(self, dataframe: pd.DataFrame):
        dataframe = mlRunModel(dataframe, self.training_params)
        print(dataframe['buy_signal'].value_counts())
        print(dataframe['sell_signal'].value_counts())
        return dataframe
