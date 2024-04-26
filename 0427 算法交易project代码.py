from datetime import datetime
import os
import numpy as np
import pandas as pd
import tushare as ts  # 股市数据接口
import time
from tqdm import tqdm  # 读取进度
import warnings

warnings.filterwarnings('ignore')  # 过滤不必要的warnings

# ------------------------------------------------------------------------------------------------
'''
RunParams类：
    start_date: datetime，运行开始日期
    end_date: datetime，运行结束日期
    type: str，运行类型（回测/模拟）
'''


class RunParams:
    def __init__(self) -> None:
        self.start_date = None  # 起始日期
        self.end_date = None  # 终止日期
        self.run_type = None  # 运行类型（回测/模拟）
        self.commission_rate = None  # 佣金费率
        self.algo_type = None  # 算法类型
        self.algo_params = None  # 算法参数
        self.order_type = None  # 订单类型


'''
Portfolio类：
    position: dict，记录当前持仓
    available_cash: float，记录当前可用资金
    total_value: float，记录当前总资产
    returns: float，记录当前收益
    starting_cash: float，记录初始资金
    position_value: float，记录当前持仓市值
'''


class Portfolio:
    def __init__(self, starting_cash=100_000_000):
        self.positions = {}  # 记录当前持仓
        self.available_cash = starting_cash  # 记录当前可用资金
        self.total_value = starting_cash  # 记录当前总资产
        self.returns = 0  # 记录当前收益
        self.starting_cash = starting_cash  # 记录初始资金
        self.position_value = 0  # 记录当前持仓市值


'''
Context类：
    portfolio: Portfolio，记录当前持仓信息
    current_dt: datetime，记录当前时间
    previous_date: datetime，记录上一个交易日
    run_params: RunParams，运行参数
    universe: list，股票池
'''


class Context:
    def __init__(self, run_params, portfolio):
        self.run_params = run_params  # RunParams类
        self.portfolio = portfolio  # Portfolio类
        self.current_dt = run_params.start_date  # 记录当前时间
        self.previous_dt = run_params.start_date  # 记录上一个交易日
        self.universe = None  # 股票池
        self.benchmark = None  # 基准指数
        self.trade_book = {}  # 交易记录本
        self.target_book = pd.DataFrame(columns=self.universe)
        self.position_book = None  # 交易信号本
        # 新建一个pd用于存放total_value_book
        self.total_value_book = pd.DataFrame(columns=['total_value'])
        self.tca_book = {}


'''
全局变量G类：
    用于存储全局变量
'''


class G:
    pass

# ------------------------------------------------------------------------------------------------
'''
数据获取相关函数
'''


# 获取股票代码对应的tick数据
def get_tick_data(stock_codes):
    # 设置用户的token来访问金融数据接口
    ts.set_token('567cc6a3f980227a5844d977ee1e53b96555778bb3d038a0613f7699')
    # 根据股票代码获取ts库端口的实时数据
    df = ts.realtime_quote(ts_code=stock_codes)
    # 将DATE和TIME合并为一个时间列
    df['TIME'] = df['DATE'] + df['TIME'].str.zfill(6)  # 向左填充0操作，长度为6位
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H:%M:%S')
    df.set_index('TIME', inplace=True)
    return df


# 从数据库中读取分钟数据
def read_from_database(stock_code: str, start_date=None, end_date=None):
    # 数据库路径（修改路径）
    datebase = '/Users/tommylxt/Desktop/研一下/0001算法交易/003Project/data/2024'
    # 将字符串转换为时间格式
    start_date = start_date if start_date else '20240101'   # 日期可修改
    end_date = end_date if end_date else '20241231'   # 日期可修改
    # 读取数据
    data = pd.read_csv(datebase + '/' + stock_code + '.csv')
    # 将时间列转换为时间格式
    data['trading_time'] = pd.to_datetime(data['trading_time'], format='%Y%m%d%H%M%S%f')
    # 将时间列设置为索引
    data.set_index('trading_time', inplace=True)
    # 筛选数据
    data = data.loc[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]
    return data


'''
策略函数，用于生成因子值进行选股
'''


def RVar(data, winLen, log_label=True):
    '''
    出处：《1、2018-11-03_海通证券_金融工程_高频量价因子在股票与期货中的表现.pdf》第7页；
    RVar高频已实现方差
    公式：(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    :param data, Series: 生成因子的原始数据
    :param winLen, int: 窗长；
    :param log_label, bool: 是否取对数；
    :return: res, Series: 因子的值
    '''
    # # 对日期索引进行排序；如果日期列不是索引，需要对日期列进行排序；
    # data = data.sort_index()
    if log_label:
        # 计算取对数收益率的每日因子值 -- 替换过程中可能出现的无穷大值为0
        res_RVar = data.groupby(pd.to_datetime(data.index).date).apply(
            lambda x: np.sum(np.square(np.log(x / x.shift(1)).replace([np.inf, -np.inf], 0))))
        # res_RVar2 = data.groupby(pd.to_datetime(data.index).date).apply(lambda x: np.sum(np.square(np.log(x / x.shift(1))[~np.isinf(np.log(x / x.shift(1)))])))   # 计算方法2
        # res_RVar3 = data.groupby(pd.to_datetime(data.index).date).apply(lambda x: np.sum(np.square(log_retRatio(x)[~np.isinf(log_retRatio(x))])))     # 计算方法3
    else:
        # 计算普通收益率的每日因子值
        res_RVar = data.groupby(pd.to_datetime(data.index).date).apply(
            lambda x: np.sum(np.square(x.pct_change().replace([np.inf, -np.inf], 0))))

    # 滚动计算窗长内因子值的平均值
    res_RVar = res_RVar.rolling(window=winLen).mean()
    # 日期索引格式化为时间戳格式（所有日期都要求统一为datetime.datetime 时间戳格式）
    res_RVar.index = pd.to_datetime(res_RVar.index)

    return res_RVar


def RSkew(data, winLen, log_label=True):
    if log_label:
        # 使用对数收益率来计算
        log_returns = np.log(data / data.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        # 使用简单收益率来计算
        log_returns = data.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    # 每日的立方求和
    daily_skew_numerator = data.groupby(pd.to_datetime(data.index).date).apply(
        lambda x: np.sum((log_returns ** 3).replace([np.inf, -np.inf], 0)))

    # 计算已实现方差 RVari
    daily_rvar = data.groupby(pd.to_datetime(data.index).date).apply(
        lambda x: np.sum((log_returns ** 2).replace([np.inf, -np.inf], 0)))

    # 计算偏度
    res_RSkew = (np.sqrt(winLen) * daily_skew_numerator) / (daily_rvar ** (3 / 2))

    # 滚动计算窗长内因子值的平均值
    res_RSkew = res_RSkew.rolling(window=winLen).mean()

    # 索引格式化为时间戳格式
    res_RSkew.index = pd.to_datetime(res_RSkew.index)

    return res_RSkew


def RKurtosis(data, winLen, log_label=True):
    if log_label:
        # 使用对数收益率来计算
        log_returns = np.log(data / data.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        # 使用简单收益率来计算
        log_returns = data.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    # 每日的四次方求和
    daily_kurt_numerator = data.groupby(pd.to_datetime(data.index).date).apply(
        lambda x: np.sum((log_returns ** 4).replace([np.inf, -np.inf], 0)))

    # 计算已实现方差 RVari
    daily_rvar = data.groupby(pd.to_datetime(data.index).date).apply(
        lambda x: np.sum((log_returns ** 2).replace([np.inf, -np.inf], 0)))

    # 计算峰度
    res_RKurtosis = (winLen * daily_kurt_numerator) / (daily_rvar ** 2)

    # 滚动计算窗长内因子值的平均值
    res_RKurtosis = res_RKurtosis.rolling(window=winLen).mean()

    # 索引格式化为时间戳格式
    res_RKurtosis.index = pd.to_datetime(res_RKurtosis.index)

    return res_RKurtosis

'''
交易相关订单函数
'''


# 单只股票买卖订单基础函数
def _order(stock_code, price, volume):
    # 现金不足，将volume调整为100的倍数
    if context.portfolio.available_cash - volume * price * (1 + context.run_params.commission_rate) < 0:
        volume = int(context.portfolio.available_cash / (price * (1 + context.run_params.commission_rate)) / 100) * 100
        # print(f"现金不足，已调整为{volume}")

    # 如果volume不是100的倍数，将其调整为100的倍数
    if volume % 100 != 0:
        volume = int(volume / 100) * 100
        # print(f"不是100的倍数, 已调整为{volume}")

    # 卖出数量超过持仓数量，调整为持仓数量
    if context.portfolio.positions.get(stock_code, 0) < -volume:
        volume = -context.portfolio.positions.get(stock_code, 0)
        # print(f"卖出数量不能超过持仓数量, 已调整为{volume}")

    # 将买卖股票数量存入持仓标的信息
    context.portfolio.positions[stock_code] = context.portfolio.positions.get(stock_code, 0) + volume
    # # 如果持仓数量为0，删除该标的
    # if context.portfolio.positions[stock_code] == 0:
    #     context.portfolio.positions.pop(stock_code)

    # 剩余资金
    # 买入持仓
    if volume > 0:
        context.portfolio.available_cash -= volume * price * (1 + context.run_params.commission_rate)

    # 卖出持仓
    elif volume < 0:
        context.portfolio.available_cash -= volume * price * (1 - context.run_params.commission_rate)

    # 交易记录
    context.trade_book[stock_code] = pd.concat([context.trade_book.get(stock_code, pd.DataFrame()),
                                                pd.DataFrame({'price': price, 'volume': volume},
                                                             index=[context.current_dt])])
    # 返回实际成交量
    return volume


# 定制化订单函数
# 按股数下单 -- 直接按照股数下单
def order_by_volume(stock_code, volume):
    # 获取当前股票价格
    if context.run_params.run_type == 'paper_trade':
        price = get_tick_data(stock_code)['PRICE'].values[0]
    elif context.run_params.run_type == 'backtest':
        # print(stock_code, context.current_dt)
        price = read_from_database(stock_code, context.current_dt, context.current_dt)['open'].values[0]
    volume = _order(stock_code, price, volume)
    return volume


# 目标股数下单 -- 按照希望持有的仓位下单
def order_target_volume(stock_code, volume):
    if volume < 0:
        print("数量不能为负,已调整为0")
        volume = 0
    # 当前持有数量
    hold_volume = context.positions.get(stock_code, 0)
    # 交易数量
    delta_volume = volume - hold_volume
    # 获取当前股票价格
    if context.run_params.run_type == 'paper_trade':
        price = get_tick_data(stock_code)['PRICE'].values[0]
    elif context.run_params.run_type == 'backtest':
        price = read_from_database(stock_code, context.current_dt, context.current_dt)['open'].values[0]
    volume = _order(stock_code, price, delta_volume)
    return volume


# 按价值下单 -- 根据所给价值除以价格算出仓位直接下单
def order_by_value(stock_code, value):
    # 获取当前股票价格
    if context.run_params.run_type == 'paper_trade':
        price = get_tick_data(stock_code)['PRICE'].values[0]
    elif context.run_params.run_type == 'backtest':
        price = read_from_database(stock_code, context.current_dt, context.current_dt)['open'].values[0]
    volume = _order(stock_code, price, int(value / price))
    return volume


# 目标价值下单 -- 根据目标价值减去现有价值再除以价格得到应该买入的仓位进行下单
def order_target_value(stock_code, value):
    if value < 0:
        print("价值不能为负,已调整为0")
        value = 0
    # 获取当前股票价格
    if context.run_params.run_type == 'paper_trade':
        price = get_tick_data(stock_code)['PRICE'].values[0]
    elif context.run_params.run_type == 'backtest':
        price = read_from_database(stock_code, context.current_dt, context.current_dt)['open'].values[0]
    hold_value = context.portfolio.positions.get(stock_code, 0) * price
    delta_value = value - hold_value
    volume = order_by_value(stock_code, delta_value)
    return volume


# 根据下单类型，批量下单
def order_batch(stock_codes, targets, order_type='volume'):
    # stock_codes为一个列表，targets为一个列表，分别对应目标股票和目标的下单数量
    for stock_code, target in zip(stock_codes, targets):
        if order_type == 'volume':
            order_by_volume(stock_code, target)
        elif order_type == 'target_volume':
            order_target_volume(stock_code, target)
        elif order_type == 'value':
            order_by_value(stock_code, target)
        elif order_type == 'target_value':
            order_target_value(stock_code, target)


# 算法交易下单函数
# twap下单
# twap_param为拆分份数：16份
# twap_gap为拆分间隔分钟数：15分钟
def order_twap(stock_codes, volumes, twap_shares=16, twap_gap=15):
    target_volume = [int(volume / twap_shares) for volume in volumes]  # 目标成交量 -- 列表
    traded_volume = [0] * len(stock_codes)  # 已成交量 -- 列表
    # 下单twap_param次 = 下单份数 x 股票数
    for i in range(twap_shares):
        for j in range(len(stock_codes)):
            order_volume = order_by_volume(stock_codes[j], target_volume[j])  # 下单量
            traded_volume[j] += order_volume  # 单只股票的已成交数量
            # 重新计算下单数量
            target_volume[j] = int(
                (volumes[j] - traded_volume[j]) / (twap_shares - i - 1)) if twap_shares - i - 1 != 0 else volumes[j] - \
                                                                                                          traded_volume[
                                                                                                              j]
        # 更新时间或休眠
        if context.run_params.run_type == 'paper_trade':
            time.sleep(twap_gap * 60)  # 休眠15分钟
        elif context.run_params.run_type == 'backtest':
            context.current_dt += pd.Timedelta(minutes=twap_gap)
            if context.current_dt.hour == 11 and context.current_dt.minute == 45:  # 11:30 上午的最后一单
                context.current_dt += pd.Timedelta(hours=1, minutes=30)  # 13:15 下午的第一单


# vwap下单
# vwap_shares为拆分份数
# back_days为回看天数
# vwap_gap为拆分间隔分钟数

# 获取同时段前n个交易日的数据
def get_port_mean(stock_code, curr_time, vwap_shares=16, back_days=20, vwap_gap=15):
    curr_time = pd.to_datetime(curr_time, format='%Y%m%d')  # 日期格式转换
    past_time = curr_time - pd.tseries.offsets.BDay(back_days)  # 获取回看时间
    past_volume_1min = read_from_database(stock_code, past_time, curr_time)['volume']  # 从数据库中读取回看窗口期间数据
    past_volume_resample = past_volume_1min.resample(str(vwap_gap) + 'T').sum().to_frame()  # 将1min数据转换为vwap_gap分钟数据
    past_volume_resample = past_volume_resample.loc[past_volume_resample.index.isin(past_volume_1min.index)]  # 筛选数据
    # 新建日期列用于分组计算同一日期的交易量占比
    past_volume_resample['date'] = past_volume_resample.index.date
    # 根据日期分组的交易量占比
    past_volume_resample['port'] = past_volume_resample['volume'] / past_volume_resample.groupby('date')[
        'volume'].transform('sum')
    # 新建时间列用于分组计算同一时间段的交易量占比均值
    past_volume_resample['time'] = list(zip(past_volume_resample.index.hour, past_volume_resample.index.minute))
    past_volume_resample['port_mean'] = past_volume_resample.groupby('time')['port'].transform('mean')
    # 由于resample的特性，需要将最后一分钟的交易量占比加到前一个时间段
    past_volume_resample.iloc[vwap_shares - 1, -1] += past_volume_resample.iloc[vwap_shares, -1]
    # 返回vwap_shares个时间段的交易量占比均值
    return past_volume_resample.iloc[:vwap_shares, -1]


def order_vwap(stock_codes, volumes, vwap_shares=16, back_days=20, vwap_gap=15):
    # 计算当前时间前若干（默认20）个交易日的同时间段的交易量占比，作为拆分交易份额
    df_port_mean = pd.DataFrame(columns=stock_codes, index=range(vwap_shares))
    for stock_code in stock_codes:
        df_port_mean[stock_code] = get_port_mean(stock_code, context.current_dt, vwap_shares, back_days,
                                                 vwap_gap).values
    traded_volume = [0] * len(stock_codes)
    delta_volume = [0] * len(stock_codes)
    # print(df_port_mean[stock_codes[0]])
    # 交易vwap_shares次
    for i in range(vwap_shares):
        # 交易每只股票
        for j in range(len(stock_codes)):
            # 计算目标交易量：vwap_shares个时间段的交易量占比均值 * 总交易量 + 上次的未成交量
            target_volume = volumes[j] * df_port_mean.loc[i, stock_codes[j]] + delta_volume[j]
            # print(target_volume)
            # 下单
            order_volume = order_by_volume(stock_codes[j], target_volume)
            # 计算本次已成交数量
            traded_volume[j] += order_volume
            # 计算本次未成交数量，计入下次交易目标量
            delta_volume[j] = target_volume - order_volume
        # 更新时间或休眠
        if context.run_params.run_type == 'paper_trade':
            time.sleep(vwap_gap * 60)
        elif context.run_params.run_type == 'backtest':
            context.current_dt += pd.Timedelta(minutes=vwap_gap)
            if context.current_dt.hour == 11 and context.current_dt.minute == 45:
                context.current_dt += pd.Timedelta(hours=1, minutes=30)
    # print(traded_volume)


# 总交易函数
def order(stock_codes, targets, algo_type, algo_prams, order_type):
    if algo_type == 'twap':
        order_twap(stock_codes, targets, algo_prams['twap_shares'], algo_prams['twap_gap'])
    elif algo_type == 'vwap':
        order_vwap(stock_codes, targets, algo_prams['vwap_shares'], algo_prams['back_days'], algo_prams['vwap_gap'])
    else:
        order_batch(stock_codes, targets, order_type)


# 计算历史的vwap
def calculate_history_vwap(stock_code, start_date, end_date):
    data = read_from_database(stock_code, start_date, end_date)
    data['avg_price'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    # 将1min频率数据转换为日频数据的均价
    vwap = data.groupby(data.index.date).apply(lambda x: np.sum(x['avg_price'] * x['volume']) / np.sum(x['volume']))
    return vwap


# ------------------------------------------------------------------------------------------------
def initialize(context):
    # 初始化运行设置
    universe = pd.read_csv('/Users/tommylxt/Desktop/研一下/0001算法交易/003Project/data/000300.SH_info.csv')[
        'con_code'].tolist()

    context.run_params.start_date = '20240101'
    context.run_params.end_date = '20240412'
    context.run_params.run_type = 'backtest'
    context.run_params.commission_rate = 0.0003
    context.run_params.algo_type = 'vwap'
    context.run_params.algo_params = {'vwap_shares': 16, 'back_days': 20, 'vwap_gap': 15}
    context.run_params.order_type = 'volume'
    context.benchmark = '000300.SH'
    context.universe = universe
    context.position_book = pd.DataFrame(columns=universe)


def run():
    # 初始化
    initialize(context)
    # 选股
    df_signal = pd.DataFrame(columns=context.universe)
    print('生成因子信号')
    for stock_code in tqdm(context.universe):
        series_data = read_from_database(stock_code, context.run_params.start_date, context.run_params.end_date)[
            'close'].copy()
        df_signal[stock_code] = RVar(series_data, 20, log_label=True)
    df_position = (df_signal.shift(1).dropna().rank(axis=1, pct=True) > 0.9).astype(int)
    # print(df_position)

    # 初始化上一个交易日
    context.previous_dt = df_position.index[0]
    # print(context.previous_dt)
    # 初始化持仓记录
    context.position_book.loc[df_position.index[0], context.universe] = 0
    # print(context.position_book)
    context.total_value_book.loc[df_position.index[0], 'total_value'] = context.portfolio.total_value

    # 回测运行
    print('\n开始回测')
    for day in tqdm(df_position.index[1:]):
        # 更新当前时间
        context.current_dt = day + pd.Timedelta('09:30:00')
        # print(f'今天是：{context.current_dt}')
        # 获取昨日持仓
        position_yesterday = context.position_book.loc[[context.previous_dt], :]
        # 获取今日目标持仓股票
        target_stocks = df_position.loc[[day], :]
        # print(type(target_stocks))
        # 持仓股票数量
        stock_nums = 0
        # 获取今日股票价格
        for stock_code in target_stocks.columns:
            if target_stocks[stock_code].values[0] == 1:
                stock_nums += 1
                target_stocks[stock_code] *= \
                    read_from_database(stock_code, context.current_dt, context.current_dt)['open'].values[0]

        # print(target_stocks)
        # 计算每只股票的目标持仓金额
        target_value = context.portfolio.total_value * 0.8 / stock_nums
        # print(target_value)

        # # 计算每只股票的目标持仓数量
        target_stocks = target_stocks.apply(lambda x: target_value / x, axis=1).replace([np.inf, -np.inf], 0)
        # print(target_stocks)

        # print(f'目标持仓：{target_stocks.values}')
        # print(f'昨日持仓：{position_yesterday.values}')
        # # 根据仓位差异，计算目标交易量
        trade_stocks = pd.DataFrame(target_stocks.values - position_yesterday.values, columns=target_stocks.columns,
                                    index=target_stocks.index)
        # print(target_stocks)

        context.target_book = pd.concat([context.target_book, trade_stocks])

        # 按照交易量排序，先卖出后买入
        trade_stocks_sorted = trade_stocks.sort_values(by=day, axis=1)
        # print(trade_stocks_sorted)

        # 删除交易量为0的股票
        trade_stocks_sorted = trade_stocks_sorted.loc[:, trade_stocks_sorted.loc[day] != 0]
        # print(trade_stocks_sorted)

        # 获取交易股票及交易量列表
        trade_stocks = trade_stocks_sorted.columns.to_list()
        trade_volumes = list(trade_stocks_sorted.values[0])
        # print(f'交易列表：{trade_stocks}')
        # print(f'交易股数：{trade_volumes}')

        # 交易
        order(trade_stocks, trade_volumes, context.run_params.algo_type, context.run_params.algo_params,
              context.run_params.order_type)

        # 更新时间
        context.current_dt = day + pd.Timedelta('15:00:00')
        # print(context.current_dt)
        # 更新持仓记录
        context.position_book = pd.concat(
            [context.position_book, pd.DataFrame(context.portfolio.positions, index=[day])]).fillna(0)

        # 更新持仓价值
        context.portfolio.position_value = 0
        for stock_code, volume in context.portfolio.positions.items():
            price = read_from_database(stock_code, context.current_dt, context.current_dt)['close'].values[0]
            context.portfolio.position_value += price * volume

        # 更新总资产
        context.portfolio.total_value = context.portfolio.available_cash + context.portfolio.position_value
        context.total_value_book.loc[day, 'total_value'] = context.portfolio.total_value

        # 交易费用分析TCA
        for stock_code, target_volume in zip(trade_stocks, trade_volumes):
            # 获取今日交易记录
            df_trade_book = context.trade_book[stock_code]
            df_trade_today = df_trade_book.loc[df_trade_book.index.date == day.date(), :]

            # 计算平均成交价
            # print(stock_code, target_volume, day)
            avg_price = df_trade_book.groupby(df_trade_book.index.date).apply(
                lambda x: np.sum(x['price'] * x['volume'].abs()) / np.sum(x['volume'].abs()) if np.sum(
                    x['volume']) != 0 else 0).values[0]
            # 估算vwap
            vwap = calculate_history_vwap(stock_code, day, context.current_dt)

            # 计算今日成交量
            traded_volume = df_trade_today['volume'].sum()
            # print(traded_volume)
            # 计算交易手续费用
            fees = np.sum(df_trade_today['price'] * df_trade_today['volume'].abs() * context.run_params.commission_rate)

            # 获取今日开盘，收盘价
            open_time = day + pd.Timedelta(hours=9, minutes=30)
            close_time = day + pd.Timedelta(hours=15)
            today_open = read_from_database(stock_code, open_time, open_time)['open'].values[0]
            today_close = read_from_database(stock_code, close_time, close_time)['close'].values[0]

            # 计算交易成本
            trade_related_cost = (avg_price - today_open) * traded_volume
            opportunity_cost = (today_close - today_open) * (target_volume - traded_volume)

            # 总结分析结果
            tca_dict = {'avg_price': avg_price,
                        'vwap_price': vwap,
                        'target_volume': target_volume,
                        'traded_volume': traded_volume,
                        'trade_related_cost': trade_related_cost,
                        'today_open': today_open,
                        'today_close': today_close,
                        'opportunity_cost': opportunity_cost,
                        'commission_fees': fees}
            df_tca_today = pd.DataFrame(tca_dict, index=[day])

            context.tca_book[stock_code] = pd.concat([context.tca_book.get(stock_code, pd.DataFrame()), df_tca_today])

        # 今日结束，更新时间
        context.previous_dt = day

    # df_strategy_return = context.total_value_book.pct_change().fillna(0)
    # df_benchmark_return = pd.read_csv('database/000300.SH.csv')['pct_chg'].fillna(0)


# --------------------------------------------- GUI界面 ---------------------------------------------------
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading

# GUI主窗体
class TradingApp:
    def __init__(self, master, context):
        self.master = master
        self.context = context
        self.portfolio = self.context.portfolio

        master.title("算法交易系统")

        # 创建一个Notebook
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill="both")

        # 添加股票交易Tab
        self.stock_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stock_frame, text='股票交易')

        # 添加日志输出Tab
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text='交易日志输出')

        # 添加tca分析输出Tab
        self.tca_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tca_frame, text='tca分析')
        self.setup_search_frame(self.tca_frame)
        self.display_tca_analysis()


        # 当前持仓信息面板
        self.stock_info_frame = ttk.LabelFrame(self.stock_frame, text="当前持仓信息")
        self.stock_info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 交易操作面板
        # 投资组合信息
        self.portfolio_frame = ttk.LabelFrame(self.stock_frame, text="投资组合信息")
        self.portfolio_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.total_value_label = ttk.Label(self.portfolio_frame, text="总资产: ", anchor='w')
        self.total_value_label.pack(side=tk.TOP, fill=tk.X)
        self.available_cash_label = ttk.Label(self.portfolio_frame, text="可用资金: ", anchor='w')
        self.available_cash_label.pack(side=tk.TOP, fill=tk.X)
        self.position_value_label = ttk.Label(self.portfolio_frame, text="持仓市值: ", anchor='w')
        self.position_value_label.pack(side=tk.TOP, fill=tk.X)
        self.position_start_cash_label = ttk.Label(self.portfolio_frame, text="初始资金: ", anchor='w')
        self.position_start_cash_label.pack(side=tk.TOP, fill=tk.X)


        # 股票信息表
        self.tree = ttk.Treeview(self.stock_info_frame, columns=('Stock Code', 'Volume'), show='headings')
        self.tree.column('Stock Code', width=100, anchor='center')
        self.tree.heading('Stock Code', text='股票代码')

        self.tree.column('Volume', width=100, anchor='center')
        self.tree.heading('Volume', text='持仓数量')
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.load_and_display_positions()

        # 日志面板
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 交易按钮
        self.start_button = ttk.Button(self.portfolio_frame, text="开始策略", command=self.start_strategy)
        self.start_button.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.stop_button = ttk.Button(self.portfolio_frame, text="停止策略", command=self.stop_strategy)
        self.stop_button.pack(side=tk.TOP, fill=tk.X, expand=True)

        # 线程用于运行策略
        self.running = False
        self.strategy_thread = None

    def start_strategy(self):
        if not self.running:
            self.running = True
            self.strategy_thread = threading.Thread(target=self.run_strategy)
            self.strategy_thread.start()
            self.log_text.insert(tk.END, "策略已启动...\n")

    def stop_strategy(self):
        if self.running:
            self.running = False
            self.strategy_thread.join(timeout=1)  # 设置超时时间为1秒
            if self.strategy_thread.is_alive():
                self.log_text.insert(tk.END, "策略正在停止，请稍候...\n")
            else:
                self.log_text.insert(tk.END, "策略已停止...\n")

    def run_strategy(self):
        while self.running:
            run()
            self.update_portfolio_info()
            self.load_and_display_positions()
            self.display_trade_book()
            self.display_tca_analysis()

            if not self.running:
                break

    def update_portfolio_info(self):
        # 更新投资组合信息
        self.total_value_label.config(text=f"总资产: {self.portfolio.total_value:.2f}")
        self.available_cash_label.config(text=f"可用资金: {self.portfolio.available_cash:.2f}")
        self.position_value_label.config(text=f"持仓市值: {self.portfolio.position_value:.2f}")
        self.position_start_cash_label.config(text=f"初始资金: {self.portfolio.starting_cash:.2f}")


    def update_stock_info(self, stock_code, volume):
        # 查找树视图中是否已存在该股票代码
        for i in self.tree.get_children():
            if self.tree.item(i, 'values')[0] == stock_code:
                self.tree.item(i, values=(stock_code, volume))
                return
        # 如果不存在，则插入新数据
        self.tree.insert('', 'end', values=(stock_code, volume))

    def load_and_display_positions(self):
        # 清除Treeview中的现有数据
        for i in self.tree.get_children():
            self.tree.delete(i)

        # 加载新的持仓数据
        for stock_code, volume in self.portfolio.positions.items():
            self.update_stock_info(stock_code, volume)

    def display_trade_book(self):
        # 清空当前的日志面板内容
        self.log_text.delete('1.0', tk.END)

        # 创建一个空的DataFrame来存储所有有效交易
        all_trades = pd.DataFrame()

        for stock_code, trades in self.context.trade_book.items():
            # 筛选出 volume 不为0的行
            non_zero_trades = trades[trades['volume'] != 0]
            # 如果筛选后的DataFrame不为空
            if not non_zero_trades.empty:
                # 给每行加上股票代码
                non_zero_trades['stock_code'] = stock_code
                # 添加到 all_trades DataFrame
                all_trades = pd.concat([all_trades, non_zero_trades])

        # 如果合并后的DataFrame不为空，按时间进行排序
        if not all_trades.empty:
            all_trades.sort_index(inplace=True)
            # 格式化字符串并添加到日志面板中
            for index, row in all_trades.iterrows():
                self.log_text.insert(
                    tk.END,
                    f"{index.strftime('%Y-%m-%d %H:%M:%S')} - {row['stock_code']} - "
                    f"Price: {row['price']} - Volume: {row['volume']}\n"
                )

    def setup_search_frame(self, parent_frame):
        # 创建搜索框架
        search_frame = ttk.Frame(self.master)
        search_frame.pack(fill=tk.X, padx=10, pady=5)

        # 股票代码和日期的变量
        self.search_stock_code = tk.StringVar()
        self.search_date = tk.StringVar()

        # 创建股票代码标签和输入框
        search_code_label = ttk.Label(search_frame, text="股票代码:")
        search_code_label.pack(side=tk.LEFT, padx=(2, 0), pady=2)
        search_code_entry = ttk.Entry(search_frame, textvariable=self.search_stock_code, width=15)
        search_code_entry.pack(side=tk.LEFT, padx=2, pady=2)

        # 创建搜索日期标签和输入框
        search_date_label = ttk.Label(search_frame, text="搜索日期:")
        search_date_label.pack(side=tk.LEFT, padx=(2, 0), pady=2)
        search_date_entry = ttk.Entry(search_frame, textvariable=self.search_date, width=15)
        search_date_entry.pack(side=tk.LEFT, padx=2, pady=2)

        # 创建搜索按钮
        search_button = ttk.Button(search_frame, text="搜索", command=self.search_tca_data)
        search_button.pack(side=tk.LEFT, padx=2, pady=2)

        # 创建重置按钮
        reset_button = ttk.Button(search_frame, text="重置", command=self.reset_tca_data)
        reset_button.pack(side=tk.LEFT, padx=2, pady=2)

    def search_tca_data(self):
        # 获取用户输入的股票代码和日期
        stock_code = self.search_stock_code.get().strip().upper()
        search_date = self.search_date.get().strip()

        # 清空 Treeview
        for i in self.tca_tree.get_children():
            self.tca_tree.delete(i)

        # 搜索并展示数据
        for row in self.all_data_rows:
            # 检查股票代码和日期是否匹配用户输入
            if row[0] == stock_code and (search_date == "" or row[1] == search_date):
                self.tca_tree.insert('', 'end', values=row)

    def reset_tca_data(self):
        # 清空 Treeview
        self.tca_tree.delete(*self.tca_tree.get_children())

        # 重新加载所有TCA分析数据到Treeview
        for row in self.all_data_rows:
            self.tca_tree.insert('', 'end', values=row)

    def display_tca_analysis(self):
        # 首先清除tca_frame中可能存在的所有子组件
        for widget in self.tca_frame.winfo_children():
            widget.destroy()

        # 创建tca分析的Treeview控件
        tca_tree = ttk.Treeview(self.tca_frame, columns=('Stock Code', 'Date', 'Avg Price', 'VWAP', 'Target Volume', 'Traded Volume', 'Trade Related Cost', 'Today Open', 'Today Close', 'Opportunity Cost', 'Commission Fees'), show='headings')

        # 定义列标题和宽度
        columns = {
            'Stock Code': ('股票代码', 100),
            'Date': ('日期', 100),
            'Avg Price': ('平均价格', 100),
            'VWAP': ('加权平均价格', 100),
            'Target Volume': ('目标交易量', 100),
            'Traded Volume': ('实际交易量', 100),
            'Trade Related Cost': ('交易相关成本', 110),
            'Today Open': ('今开', 100),
            'Today Close': ('今收', 100),
            'Opportunity Cost': ('机会成本', 100),
            'Commission Fees': ('佣金费用', 100)
        }

        for col, (col_text, col_width) in columns.items():
            tca_tree.column(col, width=col_width, anchor='center')
            tca_tree.heading(col, text=col_text)

            # 整体的数据集，用于存放所有的行
        all_data_rows = []

        # 提取数据并格式化
        for stock_code, tca_data in self.context.tca_book.items():
            # 先按日期排序
            tca_data_sorted = tca_data.sort_index()
            for index, row in tca_data_sorted.iterrows():
                # 保持数据为两位小数，并在数据前加上股票代码
                formatted_row = [stock_code] + [index.strftime('%Y-%m-%d')] + [
                    '{:.2f}'.format(value) if isinstance(value, float) else value for value in row]
                all_data_rows.append(formatted_row)

        # 根据日期对所有行进行排序
        all_data_rows.sort(key=lambda x: x[1])

        # 插入排序后的数据到Treeview
        for row in all_data_rows:
            tca_tree.insert('', 'end', values=row)


        self.tca_tree = tca_tree  # 保存 Treeview 到类变量以便后续使用
        self.all_data_rows = all_data_rows  # 保存所有行的数据以供搜索

        # 将 Treeview 控件添加到 tca_frame 中并使其填充整个框架
        tca_tree.pack(fill=tk.BOTH, expand=True)



def main():
    global context  # 确保 context 已经被初始化
    # 初始化 context
    run_params = RunParams()
    portfolio = Portfolio(starting_cash=100_000_000)
    context = Context(run_params, portfolio)
    # 初始化 GUI
    root = tk.Tk()
    app = TradingApp(root, context)  # 传递 context 到 TradingApp
    root.mainloop()


if __name__ == "__main__":
    main()



# --------------------------------------------- 描述性统计 ---------------------------------------------------
# import quantstats as qs
# global context
# qs.extend_pandas()
#
# df_strategy_return = context.total_value_book['total_value'].pct_change().fillna(0)
# df_benchmark = pd.read_csv('/Users/tommylxt/Desktop/研一下/0001算法交易/003Project/data/000300.SH.csv', index_col='trade_date', parse_dates=True)
# df_benchmark_return = df_benchmark.loc[context.run_params.start_date:context.run_params.end_date, 'pct_chg'].fillna(0) / 100
# qs.reports.full(df_strategy_return, benchmark=df_benchmark_return)


