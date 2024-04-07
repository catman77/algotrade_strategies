import numpy as np
import pandas as pd
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import math
from enum import Enum


def calculate_CCI(dataframe, index, diff, ma, p):
    close_array = dataframe['close'].to_numpy()
    close_array = close_array[:index.name]
    diff = diff.to_numpy()
    diff = diff[index.name]
    ma = ma.to_numpy()
    ma = ma[index.name]

    if (len(close_array) < p-1):
        return np.nan

    s = 0

    for i in range(len(close_array), len(close_array)-p, -1):
        s = s + abs(dataframe['close'][i] - ma)
    mad = s / p

    mcci = diff/mad/0.015

    return mcci


def rescale(src, old_min, old_max, new_min, new_max):
    return new_min + (new_max - new_min) * (src - old_min) / max((old_max - old_min), 10e-10)


def normalize(value, min_val, max_val, df):
    index = value.name
    src = pd.Series(df[:index+1])

    historic_min = 10e10
    historic_max = -10e10

    src_filled_min = src.fillna(historic_min)
    src_filled_max = src.fillna(historic_max)
    historic_min = min(src_filled_min.min(), historic_min) if not pd.isna(
        src_filled_min.min()) else historic_min
    historic_max = max(src_filled_max.max(), historic_max) if not pd.isna(
        src_filled_max.max()) else historic_max

    normalized_src = (min_val + (max_val - min_val) *
                      (src[index] - historic_min)) / max((historic_max - historic_min), 10e-10)
    return normalized_src


def n_rsi(src, n1, n2):
    rsi = ta.RSI(src, n1)
    ema_rsi = ta.EMA(rsi, n2)
    return rescale(ema_rsi, 0, 100, 0, 1)


def n_cci(dataframe, n1, n2):
    df = dataframe.copy()
    source = df['close']

    df['mas'] = ta.SMA(source, n1)
    df['diffs'] = source - df['mas']
    df['cci'] = df.apply((lambda index: calculate_CCI(
        df, index, df['diffs'], df['mas'], n1)), axis=1)

    df['ema_cci'] = ta.EMA(df['cci'], n2)

    normalized_wt_diff = df.apply(
        (lambda x: normalize(x, 0, 1, df['ema_cci'])), axis=1)
    return normalized_wt_diff


def n_wt(src, n1, n2):
    df = pd.DataFrame({"src": src})
    ema1 = ta.EMA(src, n1)
    ema2 = ta.EMA(np.abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = ta.EMA(ci, n2)
    wt2 = ta.SMA(wt1, 4)
    diff = wt1 - wt2
    normalized_wt_diff = df.apply((lambda x: normalize(x, 0, 1, diff)), axis=1)
    return normalized_wt_diff


def calculate_tr(index, high, low, close):
    high = pd.Series(high[:index+1])
    low = pd.Series(low[:index+1])
    close = pd.Series(close[:index+1])
    prev_close = close.shift(1).fillna(0)

    diff_h_n_l = high[index] - low[index]
    abs_value_h_n_c = abs(high[index] - prev_close[index])
    abs_value_l_n_c = abs(low[index] - prev_close[index])

    tr = max(max(diff_h_n_l, abs_value_h_n_c), abs_value_l_n_c)
    return tr


def calculate_directionalMovementPlus(index, high, low):
    high = pd.Series(high[:index+1])
    low = pd.Series(low[:index+1])
    prev_high = high.shift(1).fillna(0)
    prev_low = low.shift(1).fillna(0)

    diff_h_n_ph = high[index] - prev_high[index]
    diff_pl_n_l = prev_low[index] - low[index]
    dmp_value = max(diff_h_n_ph, 0) if (diff_h_n_ph > diff_pl_n_l) else 0

    return dmp_value


def calculate_negMovement(index, high, low):
    high = pd.Series(high[:index+1])
    low = pd.Series(low[:index+1])
    prev_high = high.shift(1).fillna(0)
    prev_low = low.shift(1).fillna(0)

    diff_h_n_ph = high[index] - prev_high[index]
    diff_pl_n_l = prev_low[index] - low[index]
    negMovement = max(diff_pl_n_l, 0) if (diff_pl_n_l > diff_h_n_ph) else 0
    return negMovement


def n_adx(highSrc, lowSrc, closeSrc, dataframe, n1):
    df = dataframe.copy()
    length = n1
    th = 20
    tr = df.apply((lambda x: calculate_tr(
        x.name, highSrc, lowSrc, closeSrc)), axis=1)
    directionalMovementPlus = df.apply(
        (lambda x: calculate_directionalMovementPlus(x.name, highSrc, lowSrc)), axis=1)
    negMovement = df.apply(
        (lambda x: calculate_negMovement(x.name, highSrc, lowSrc)), axis=1)

    trSmooth = np.zeros_like(closeSrc)
    trSmooth[0] = np.nan
    for i in range(0, len(tr)):
        trSmooth[i] = trSmooth[i-1] - trSmooth[i-1] / length + tr[i]

    smoothDirectionalMovementPlus = np.zeros_like(closeSrc)
    smoothDirectionalMovementPlus[0] = np.nan
    for i in range(0, len(directionalMovementPlus)):
        smoothDirectionalMovementPlus[i] = smoothDirectionalMovementPlus[i-1] - \
            smoothDirectionalMovementPlus[i-1] / \
            length + directionalMovementPlus[i]

    smoothnegMovement = np.zeros_like(closeSrc)
    smoothnegMovement[0] = np.nan
    for i in range(0, len(negMovement)):
        smoothnegMovement[i] = smoothnegMovement[i-1] - \
            smoothnegMovement[i-1] / length + negMovement[i]

    diPositive = smoothDirectionalMovementPlus / trSmooth * 100
    diNegative = smoothnegMovement / trSmooth * 100
    dx = np.abs(diPositive - diNegative) / (diPositive + diNegative) * 100
    dx_series = pd.Series(dx)

    adx = dx_series.copy()
    adx.iloc[:length] = adx.rolling(length).mean().iloc[:length]
    adx = adx.ewm(alpha=(1.0/length), adjust=False).mean()
    return rescale(adx, 0, 100, 0, 1)


def heikinashi(df: pd.DataFrame) -> pd.DataFrame:
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


class FeatureName(Enum):
    rsi = "RSI"
    wt = "WT"
    cci = "CCI"
    adx = "ADX"


def chooseFeatureName(name: FeatureName, dataframe, paramsA, paramsB):
    df = dataframe.copy()
    source = df['close']
    hlc3 = (df['high'] + df['low'] + df['close']) / 3

    if (name.name == FeatureName.rsi.name):
        return n_rsi(source, paramsA, paramsB)
    if (name.name == FeatureName.wt.name):
        return n_wt(hlc3, paramsA, paramsB)
    if (name.name == FeatureName.cci.name):
        return n_cci(df, paramsA, paramsB)
    if (name.name == FeatureName.adx.name):
        return n_adx(df['high'], df['low'], df['close'], df, paramsA)


def extract_features(dataframe: pd.DataFrame, training_params):
    df = dataframe.copy()

    f1_name = training_params['f1']['name']
    f1_param_A = training_params['f1']['paramsA']
    f1_param_B = training_params['f1']['paramsB']

    f2_name = training_params['f2']['name']
    f2_param_A = training_params['f2']['paramsA']
    f2_param_B = training_params['f2']['paramsB']

    f3_name = training_params['f3']['name']
    f3_param_A = training_params['f3']['paramsA']
    f3_param_B = training_params['f3']['paramsB']

    f4_name = training_params['f4']['name']
    f4_param_A = training_params['f4']['paramsA']
    f4_param_B = training_params['f4']['paramsB']

    f5_name = training_params['f5']['name']
    f5_param_A = training_params['f5']['paramsA']
    f5_param_B = training_params['f5']['paramsB']

    df['f1'] = chooseFeatureName(f1_name, df, f1_param_A, f1_param_B)
    df['f2'] = chooseFeatureName(f2_name, df, f2_param_A, f2_param_B)
    df['f3'] = chooseFeatureName(f3_name, df, f3_param_A, f3_param_B)
    df['f4'] = chooseFeatureName(f4_name, df, f4_param_A, f4_param_B)
    df['f5'] = chooseFeatureName(f5_name, df, f5_param_A, f5_param_B)
    return df


def filter_volatility(dataframe: pd.DataFrame, minLength: int, maxLength: int):
    df = dataframe.copy()
    recentAtr = ta.ATR(df["high"], df["low"],
                       df["close"], timeperiod=minLength)
    historicalAtr = ta.ATR(df["high"], df["low"],
                           df["close"], timeperiod=maxLength)
    return recentAtr > historicalAtr


def regime_filter(dataframe, threshold):
    df = dataframe.copy()
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    src = ohlc4

    value1 = pd.Series(0, index=df['close'].index, dtype=float)
    value2 = pd.Series(0, index=df['close'].index, dtype=float)
    klmf = pd.Series(0, index=df['close'].index, dtype=float)

    for i in range(0, len(value1)):
        if (i == 0):
            value1[i] = 0
        else:
            value1[i] = 0.2 * (src[i] - src[i-1]) + 0.8 * value1[i-1]

    for i in range(0, len(value1)):
        if (i == 0):
            value2[i] = 0.1 * (df['high'][i] - df['low'][i]) + 0.8 * 0
        else:
            value2[i] = 0.1 * (df['high'][i] - df['low']
                               [i]) + 0.8 * value2[i-1]

    omega = abs(value1 / value2)
    alpha = (-omega**2 + np.sqrt(omega**4 + 16 * omega**2)) / 8

    for i in range(0, len(value1)):
        if (i == 0):
            klmf[i] = alpha[i] * src[i] + (1 - alpha[i]) * 0
        else:
            klmf[i] = alpha[i] * src[i] + (1 - alpha[i]) * klmf[i-1]

    absCurveSlope = klmf.diff().abs()
    exponentialAverageAbsCurveSlope = 1.0 * ta.EMA(absCurveSlope, 200)
    normalized_slope_decline = (
        absCurveSlope - exponentialAverageAbsCurveSlope) / exponentialAverageAbsCurveSlope
    return normalized_slope_decline >= threshold


def ema_filter(dataframe, period):
    df = dataframe.copy()
    ema = ta.EMA(df['close'], period)
    filter_value = (df['close'] > ema).astype(
        int) - (df['close'] < ema).astype(int)
    return filter_value


def sma_filter(dataframe, period):
    df = dataframe.copy()
    sma = ta.SMA(df['close'], period)
    filter_value = (df['close'] > sma).astype(
        int) - (df['close'] < sma).astype(int)
    return filter_value


def kernel_filter(dataframe, loopback, relative_weight, start_at_bar):
    df = dataframe.copy()
    khat1 = pd.Series(rational_quadratic(
        df['close'], loopback, relative_weight, start_at_bar))
    # wasBearishRate = khat1.shift(2) > khat1.shift(1)
    # isBearishRate = khat1.shift(1) > khat1
    # wasBullishRate = khat1.shift(2) < khat1.shift(1)
    filter_rate = (khat1.shift(1) < khat1).astype(
        int) - (khat1.shift(1) > khat1).astype(int)
    return filter_rate


def rational_quadratic(
    price_feed: np.ndarray,
    lookback: int,
    relative_weight: float,
    start_at_bar: int,
) -> np.ndarray:
    length_of_prices = len(price_feed)
    bars_calculated = start_at_bar + 1

    result = np.zeros(length_of_prices, dtype=float)
    lookback_squared = np.power(lookback, 2)
    denominator = lookback_squared * 2 * relative_weight

    for index in range(length_of_prices):
        current_weight = 0.0
        cumulative_weight = 0.0

        for i in range(bars_calculated):
            y = np.nan if (index - i) < 0 else price_feed[index - i]
            w = np.power(
                1 + (np.power(i, 2) / denominator),
                -relative_weight,
            )
            current_weight += y * w
            cumulative_weight += w

        result[index] = current_weight / cumulative_weight

    return result


def gaussian(
    price_feed: np.ndarray,
    lookback: int,
    start_at_bar: int,
) -> np.ndarray:
    length_of_prices = len(price_feed)
    bars_calculated = start_at_bar + 1

    result = np.zeros(length_of_prices, dtype=float)
    lookback_squared = np.power(lookback, 2)
    denominator = lookback_squared * 2

    for index in range(length_of_prices):
        current_weight = 0.0
        cumulative_weight = 0.0

        for i in range(bars_calculated):
            y = np.nan if (index - i) < 0 else price_feed[index - i]
            w = np.exp(-(np.power(i, 2) / denominator))
            current_weight += y * w
            cumulative_weight += w

        result[index] = current_weight / cumulative_weight

    return result


class Filter(Enum):
    volatility = "filter_volatility"
    regime = "regime_filter"
    ema = "ema_filter"
    sma = "sma_filter"
    kernel = "kernel_filter"


def getLorentizanDistance(i, current_feature, feature_array):
    feature_distance = math.log(1 + abs(current_feature - feature_array[i]))
    return feature_distance


def fractalFilters(predict_value: pd.Series):
    isDifferentSignalType = predict_value.ne(predict_value.shift())
    return isDifferentSignalType


def compare_value(index, length, value):
    df = value
    prev_df = value.shift(length)
    if (prev_df[index] < df[index]):
        return -1
    elif (prev_df[index] > df[index]):
        return 1
    else:
        return 0


def setPredictionAsClearWay(index, dataframe: pd.DataFrame, filter_method):
    df = dataframe.copy()
    global signal_predictions
    prediction_value = 0
    predicted_value = df['predicted_value'].iloc[index]
    filter_value = True

    for filter in filter_method:
        if ((filter.name == "ema") or (filter.name == "sma") or (filter.name == "kernel")):
            filter_value = True and filter_value
        else:
            filter_value = (df[filter.value].iloc[index]) and (filter_value)

    if (predicted_value > 0) & filter_value:
        prediction_value = 1
    elif (predicted_value < 0) & filter_value:
        prediction_value = -1
    else:
        if index == 0:
            prediction_value = 0
        else:
            prediction_value = signal_predictions[index-1]
    signal_predictions[index] = prediction_value
    return prediction_value


def train_model(index, df, training_params):
    current_index = index
    lastDistance = -1.0
    neighbour_count = training_params['neighbor_count']
    feature_count = training_params['feature_count']
    # Variable Used for ML
    global distances
    global predictions

    feature_array_1 = df['f1'].to_numpy()
    feature_array_2 = df['f2'].to_numpy()
    feature_array_3 = df['f3'].to_numpy()
    feature_array_4 = df['f4'].to_numpy()
    feature_array_5 = df['f5'].to_numpy()
    y_train_array = df['y_train'].to_numpy()

    current_feature_1 = feature_array_1[current_index]
    current_feature_2 = feature_array_2[current_index]
    current_feature_3 = feature_array_3[current_index]
    current_feature_4 = feature_array_4[current_index]
    current_feature_5 = feature_array_5[current_index]

    feature_array_1 = feature_array_1[:current_index+1]
    feature_array_2 = feature_array_2[:current_index+1]
    feature_array_3 = feature_array_3[:current_index+1]
    feature_array_4 = feature_array_4[:current_index+1]
    feature_array_5 = feature_array_5[:current_index+1]

    y_train_array = y_train_array[:current_index+1]

    for i in range(0, current_index+1, 1):
        d = 0
        current_feature_names = [current_feature_1, current_feature_2,
                                 current_feature_3, current_feature_4, current_feature_5]
        feature_array_names = [feature_array_1, feature_array_2,
                               feature_array_3, feature_array_4, feature_array_5]
        current_feature_names = current_feature_names[:feature_count]
        feature_array_names = feature_array_names[:feature_count]

        for var_index, _ in enumerate(current_feature_names):
            current_feature_count = current_feature_names[var_index]
            feature_array_count = feature_array_names[var_index]
            d = getLorentizanDistance(
                i, current_feature_count, feature_array_count) + d

        if (d >= lastDistance) and (i % 4):
            lastDistance = d
            distances.append(d)
            predictions.append(round(y_train_array[i]))
            if len(predictions) > neighbour_count:
                lastDistance = distances[round(neighbour_count*3/4)]
                distances.pop(0)
                predictions.pop(0)

    prediction = sum(predictions)
    return prediction


def predict_future(dataframe: pd.DataFrame, training_params):
    df = dataframe.copy()
    global signal_predictions
    signal_predictions = {}
    filter_method = training_params['filter_method']
    future_count = training_params['future_count']

    df['y_train'] = df.apply(lambda x: compare_value(
        x.name, future_count, df['close']), axis=1)
    df['predicted_value'] = df.apply(
        (lambda x: train_model(x.name, df, training_params)), axis=1)
    dataframe['final_prediction'] = df.apply(
        lambda x: setPredictionAsClearWay(x.name, df, filter_method), axis=1)
    df['isDifferentSignalType'] = fractalFilters(dataframe['final_prediction'])

    dataframe['buy_signal'] = (dataframe['final_prediction'] > 0) & (
        df['isDifferentSignalType'])
    dataframe['sell_signal'] = (dataframe['final_prediction'] < 0) & (
        df['isDifferentSignalType'])

    for filter in filter_method:
        value = df[filter.value]
        if ((filter.name == "ema") or (filter.name == "sma") or (filter.name == "kernel")):
            dataframe['buy_signal'] = dataframe['buy_signal'] & (value > 0)
            dataframe['sell_signal'] = dataframe['sell_signal'] & (value < 0)

    return dataframe


def mlRunModel(dataframe, training_params):
    df = dataframe.copy()

    kernel_loopback = training_params['filter_params']['kernel']['loopback']
    kernel_relative_weight = training_params['filter_params']['kernel']['weighting']
    kernel_start_at_bar = training_params['filter_params']['kernel']['regression_level']

    df = extract_features(df, training_params)

    df['filter_volatility'] = filter_volatility(df, 1, 10)
    df['regime_filter'] = regime_filter(
        df, training_params['filter_params']['regime']['threshold'])
    df['ema_filter'] = ema_filter(
        df, training_params['filter_params']['ema']['threshold'])
    df['sma_filter'] = sma_filter(
        df, training_params['filter_params']['sma']['threshold'])
    df['kernel_filter'] = kernel_filter(
        df, kernel_loopback, kernel_relative_weight, kernel_start_at_bar)
    df = predict_future(df, training_params)
    return df


distances = []
predictions = []
signal_predictions = {}
