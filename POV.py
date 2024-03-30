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
data = data[:5000]


def pov_vol_trade(data, participation_rate):
    """
    根据历史成交量计算下一日总订单量。
    :param data, DataFrame: 待处理的原始数据框
    :param participation_rate, float: 当前计划交易量是市场前一日成交量的比例

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
    output['volume_trade'] = (data.groupby(['sec_code', 'trade_day'])['volume'].sum() * participation_rate).values
    # 将trade_day列向前移动，得到每条记录对应的下一交易日的日期
    output['trade_day'] = output.groupby('sec_code')['trade_day'].shift(-1)
    # 将计算得到的信息合并到原始数据中，使用sec_code和trade_day作为连接键
    data1 = pd.merge(data, output, on=['sec_code', 'trade_day'], how='left')
    return data1


def pov_vol_dis(data1, participation_rate):
    """
    PoV（Percent of Volume，比例成交量）是一种在算法交易中常用的执行策略。算法交易是通过计算机程序自动执行的交易策略，旨在实现最佳的交易
    执行，降低成本、提高交易速度以及提升交易效率。PoV策略的基本思想是将大单交易分散到多个较小的订单中，以减小市场冲击，避免价格波动对交易成本
    的影响。这种策略根据预先设定的参数，将大单交易分散到一定时间周期内的多个订单中，每个订单的成交量与上一时刻市场总成交量的比例保持在一个相对
    稳定的水平。
    :param data1, DataFrame: 处理后的数据框
    :param participation_rate, float: 当前计划交易量是市场前一日成交量的比例

    :return: data2, DataFrame: 处理后的数据框
    """
    data2 = data1.copy()
    # 计算市场某只股票上一时刻的成交量
    data2['volume_history'] = data2.groupby(['sec_code', 'trade_day'])['volume'].shift(-1)
    # 根据市场成交量，按照固定比例进行下单
    data2['volume_trade_i'] = data2['volume_history'] * participation_rate
    # 根据市场成交量，计算市场权重
    data2['weight_market'] = data2.groupby(['sec_code', 'trade_day'])['volume'].apply(lambda x: x / x.sum()).values
    # 计算每一个时间段真实交易量，即实际成交量和预测成交量的最小值
    data2['volume_real_i'] = data2.apply(lambda x: min(x['volume'], x['volume_trade_i']), axis=1)
    # 根据每个时间段真实成交量，计算真实权重
    data2['weight_real'] = data2.groupby(['sec_code', 'trade_day'])['volume_real_i'].apply(lambda x: x / x.sum()).values
    return data2


def pov_result(data2):
    """
    计算绩效评价指标，判断策略执行效果
    :param data2, DataFrame: 处理后的数据框

    :return: output, DataFrame: 绩效评价指标
    """
    # 创建一个空的DataFrame用于存储计算后的结果
    output = pd.DataFrame()
    # 从处理后的数据中提取唯一的股票代码（sec_code）和交易日期（trade_day）组合，去除重复的组合
    output[['sec_code', 'trade_day']] = data2[['sec_code', 'trade_day']].drop_duplicates()
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


def pov(data, participation_rate):
    """
    根据市场参与率执行 POV (Percentage of Volume) 算法交易策略
    :param data, DataFrame: 原始数据框，包括各标的的分钟收盘价、成交量等数据
    :param participation_rate, float: 市场参与率，表示当前计划交易量是市场前一日成交量的比例

    :return: data2, DataFrame: 在data基础上添加模拟交易数据字段
    :return: output, DataFrame: 回测结果，包括POV和相关指标数据
    """
    # 调用pov_vol_trade函数，得到处理后的数据data1
    data1 = pov_vol_trade(data, participation_rate)
    # 调用pov_vol_dis函数，得到处理后的数据data2
    data2 = pov_vol_dis(data1, participation_rate)
    # 调用pov_result函数，得到回测结果output
    output = pov_result(data2)
    return data2, output


# 调用pov函数，传入原始数据，参与度，得到模拟交易数据和回测结果
data_pov, output_pov = pov(data, 0.2)
# 打印模拟交易数据和回测结果
print(output_pov)
# 将output_vwap保存为CSV文件
output_pov.to_csv('result/output.csv', index=False)

