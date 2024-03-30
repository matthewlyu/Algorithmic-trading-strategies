# 导入Pandas库，用于数据处理
import pandas as pd

# 导入NumPy库，用于数值计算
import numpy as np

# 导入bz2库，用于读取压缩数据
import bz2

# 导入警告模块，用于忽略警告信息
import warnings
warnings.filterwarnings("ignore")

# 读取压缩数据文件，使用Pandas的read_pickle函数解析二进制数据
with bz2.open('./data/2022_600508_trade_pickle4.bz2', 'rb') as f:
    data_all = pd.read_pickle(f)

# 提取数据中的日期和时间信息，并分为不同的列
data_all[['trade_day', 'time']] = data_all['date'].astype(str).str.split(' ', expand=True)
# 创建新的DataFrame存储所需的列
data = pd.DataFrame()
data[['sec_code', 'trade_day', 'date_time', 'p_close', 'volume']] = data_all[['StockID', 'trade_day',
                                                                              'date', 'close', 'vol']]

data = data[:10000]


def mvwap_vol_trade(data, ratio):
    """
    根据历史成交量计算下一日总订单量。
    :param data, DataFrame: 待处理的原始数据框
    :param ratio, float: 当前计划交易量是前一日成交量的比例

    :return: data1, DataFrame: 处理后的数据框
    """
    # 格式化日期时间
    data['trade_day'] = pd.to_datetime(data['trade_day'], format='%Y-%m-%d')
    data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
    # 去除重复的数据，保留每个股票每个时间点的第一条记录
    data.drop_duplicates(subset=['sec_code', 'date_time'], keep='first', inplace=True)
    # 重置数据的索引
    data.reset_index(drop=True, inplace=True)
    # 根据股票代码和交易日期排序数据
    data.sort_values(by=['sec_code', 'date_time'], inplace=True)
    # 删除停牌交易日的观测，保留每个股票在每个交易日有交易的记录
    data = data[data.groupby(['sec_code', 'trade_day'])['volume'].apply(
        lambda x: pd.Series(np.full(len(x), x.sum() > 0))).values]
    # 计算当日预期总成交量，此处假定是市场前日成交量的ratio比例
    # 创建一个空的DataFrame用于存储计算后的数据
    output = pd.DataFrame()
    # 从原始数据中提取唯一的股票代码（sec_code）和交易日期（trade_day）组合，去除重复的组合
    output[['sec_code', 'trade_day']] = data[['sec_code', 'trade_day']].drop_duplicates()
    # 计算每个股票在每个交易日的交易量总和，然后乘以ratio得到下一日总订单量
    output['volume_trade'] = (data.groupby(['sec_code', 'trade_day'])['volume'].sum() * ratio).values
    # 将trade_day列向前移动，得到每条记录对应的下一交易日的日期
    output['trade_day'] = output.groupby('sec_code')['trade_day'].shift(-1)
    # 将计算得到的信息合并到原始数据中，使用sec_code和trade_day作为连接键
    data1 = pd.merge(data, output, on=['sec_code', 'trade_day'], how='left')
    return data1


def mvwap_vol_dis(data1, ndays, scaling_factor):
    """
    MVWAP（Modified Volume Weighted Average Price），成交量加权平均价格优化算法。在原始VWAP的基础之上有很多优化和改进的算法，
    最为常见的一种策略是根据市场实时价格和 VWAP_市场 的关系，对下单量的大小进行调整与控制，我们统一将这一类算法称为MVWAP。当市场实时价格小于
    此时的 VWAP_市场 时，在原有计划交易量的基础上进行放大，如果能够将放大的部分成交或部分成交，则有助于降低 VWAP_成交；反之，当市场实时价格
    大于此时的 VWAP_市场 时，在原有计划交易量的基础上进行缩减，也有助于降低 VWAP_成交，从而达到控制交易成本的目的。
    :param data1, DataFrame: 处理后的数据框
    :param ndays, int: 时间窗口长度
    :param scaling_factor, float: 缩放因子，用于调整下单量

    :return: data2, DataFrame: 处理后的数据框
    """
    time = data1['date_time'].apply(lambda x: x.strftime('%H:%M:%S'))
    data11 = pd.DataFrame()
    # 对于每一天的每一个时间段，预测可能会产生的成交量
    for i in range(len(time)):
        # 选择当前时间段的数据
        data1_i = data1[data1['date_time'].apply(lambda x: x.strftime('%H:%M:%S')) == time[i]]
        # 计算滚动平均成交量作为预测成交量
        volume_forecast = data1_i.groupby('sec_code')['volume'].rolling(ndays).mean()
        data1_i['volume_forcast'] = volume_forecast.reset_index(drop=True)
        # 将时间列向前平移一步，作为预测下一个时间段的数据
        data1_i['date_time'] = data1_i.groupby('sec_code')['date_time'].shift(-1)
        # 保留所需的列
        data1_i = data1_i[['sec_code', 'date_time', 'volume_forcast']]
        # 将当前时间段的数据追加到data11中
        data11 = pd.concat([data11, data1_i], axis=0)

    # 将预测成交量数据按股票代码和日期时间排序
    data11.sort_values(by=['sec_code', 'date_time'], inplace=True)
    # 将预测成交量数据与之前的数据合并
    data2 = pd.merge(data1, data11, on=['sec_code', 'date_time'], how='left')
    # 根据预测成交量数据，产生每一个时间段预测权重
    data2['weight_forcast'] = data2.groupby(['sec_code', 'trade_day'])['volume_forcast'].apply(
        lambda x: x / x.sum()).values
    # 计算实时VWAP
    data2['vwap_realtime'] = data2.groupby(['sec_code', 'trade_day']).apply(
        lambda x: ((x['p_close'] * x['volume']).cumsum() / x['volume'].cumsum())).values
    # 根据实时VWAP与实时价格之间的关系，调整下单量，如果实时价格较小，扩大下单量，反之，缩小下单量
    data2['scaling'] = data2.apply(
        lambda x: scaling_factor if x['p_close'] < x['vwap_realtime'] else 1 / scaling_factor, axis=1)
    # 将当日预期交易量按比例分配到每一个时间段
    data2['volume_trade_i'] = data2['volume_trade'] * data2['weight_forcast'] * data2['scaling']
    # 根据市场交易量，计算市场权重
    data2['weight_market'] = data2.groupby(['sec_code', 'trade_day'])['volume'].apply(lambda x: x / x.sum()).values
    # 计算每一个时间段真实交易量，即实际成交量和预测成交量的最小值
    data2['volume_real_i'] = data2.apply(lambda x: min(x['volume'], x['volume_trade_i']), axis=1)
    # 根据每个时间段真实交易量，计算真实权重
    data2['weight_real'] = data2.groupby(['sec_code', 'trade_day'])['volume_real_i'].apply(lambda x: x / x.sum()).values

    return data2


def mvwap_result(data2):
    """
    计算MVWAP（Modified Volume Weighted Average Price）算法执行效果的绩效评价指标。
    :param data2 DataFrame: MVWAP算法执行后的数据框，包括预测权重、市场权重、真实权重等信息。

    :return output DataFrame: 包含MVWAP算法绩效评价指标的数据框，如vdfa、oer、vwap_market、vwap_real和maper。
    """
    # 创建一个空的DataFrame用于存储计算后的结果
    output = pd.DataFrame()
    # 从处理后的数据中提取唯一的股票代码（sec_code）和交易日期（trade_day）组合，去除重复的组合
    output[['sec_code', 'trade_day']] = data2[['sec_code', 'trade_day']].drop_duplicates()
    # 计算绩效评价指标vdfa，用于衡量预期权重和市场权重之间的关系
    output['vdfa'] = data2.groupby(['sec_code', 'trade_day']).apply(
        lambda x: pow((x['weight_forcast'] - x['weight_market']), 2).sum()).values
    # 计算绩效评价指标oer，用于衡量真实成交的交易量和市场交易量之间的关系
    output['oer'] = data2.groupby(['sec_code', 'trade_day']).apply(
        lambda x: ((x['volume_real_i'].sum()) / x['volume_trade']).mean()).values
    # 根据市场权重和真实权重，分别计算市场vwap和真实vwap
    output['vwap_market'] = data2.groupby(['sec_code', 'trade_day']).apply(
        lambda x: (x['p_close'] * x['weight_market']).sum()).values
    output['vwap_real'] = data2.groupby(['sec_code', 'trade_day']).apply(
        lambda x: (x['p_close'] * x['weight_real']).sum()).values
    # 计算绩效评价指标maper，用于衡量真实vwap和市场vwap之间的关系
    output['maper'] = (output['vwap_real'] - output['vwap_market']) / output['vwap_market']
    return output


def mvwap(data, ndays, ratio, scaling_factor):
    """
    利用MVWAP（Modified Volume Weighted Average Price）算法进行算法交易拆单，减小市场冲击，并进行模拟交易。
    :param data DataFrame: 数据框，包括各标的的分钟收盘价、成交量等数据。
    :param ndays int: 时间窗口长度，默认值为60。
    :param ratio float: 阈值，默认值为0.6。
    :param scaling_factor: 缩放因子，默认值为1.2

    :return data2 DataFrame: 数据框，在data基础上添加模拟交易数据字段。
    :return output DataFrame: 包含VWAP算法绩效评价指标的数据框，如vdfa、oer、vwap_market、vwap_real和maper。
    """
    # 调用mvwap_vol_trade函数，得到处理后的数据data1
    data1 = mvwap_vol_trade(data, ratio)
    # 调用mvwap_vol_dis函数，得到处理后的数据data2
    data2 = mvwap_vol_dis(data1, ndays, scaling_factor)
    # 调用mvwap_result函数，得到回测结果output
    output = mvwap_result(data2)
    return data2, output


# 调用mvwap函数，传入原始数据，时间窗口长度，阈值和缩放因子，得到模拟交易数据和回测结果
data_mvwap, output_mvwap = mvwap(data, 60, 0.2, 1.2)
# 打印模拟交易数据和回测结果
print(data_mvwap, '\n', output_mvwap)
# 将output_vwap保存为CSV文件
output_mvwap.to_csv('result/output.csv', index=False)
