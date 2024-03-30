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


def twap_vol_trade(data, ratio):
    """
    根据历史交易量计算下一日总订单量
    :param data, DataFrame: 待处理的原始数据框
    :param ratio, float: 当前计划交易量是前一日交易量的比例

    :return: data1, DataFrame: 第一次处理后的数据
    """
    # 格式化日期时间列
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
    # 计算日交易量，此处假定是市场前日交易量的ratio比例
    # 创建一个空的DataFrame用于存储计算后的数据
    output = pd.DataFrame()
    # 从原始数据中提取唯一的股票代码（sec_code）和交易日期（trade_day）组合，去除重复的组合
    output[['sec_code', 'trade_day']] = data[['sec_code', 'trade_day']].drop_duplicates()
    # 重置索引，确保索引是连续的整数序列
    output.reset_index(drop=True, inplace=True)
    # 计算每个股票在每个交易日的交易量总和，然后乘以ratio得到下一日总订单量
    output['volume_trade'] = (data.groupby(['sec_code', 'trade_day'])['volume'].sum() * ratio).values
    # 将trade_day列向前移动，得到每条记录对应的下一交易日的日期
    output['trade_day'] = output.groupby('sec_code')['trade_day'].shift(-1)
    # 将计算得到的信息合并到原始数据中，使用sec_code和trade_day作为连接键
    data1 = pd.merge(data, output, on=['sec_code', 'trade_day'], how='left')
    return data1


def twap_vol_dis(data1):
    """
    TWAP（Time Weighted Average Price），时间加权平均价格算法，是一种最简单的传统算法交易策略。
    该模型将交易时间进行均匀分割，并在每个分割节点上将拆分的订单进行提交。例如，可以将某个交易日的交易时间平均分为N段，
    TWAP策略会将该交易日需要执行的订单均匀分配在这N个时间段上去执行，从而使得交易均价跟踪TWAP。
    :param data1, DataFrame: 经过twap_vol_trade处理后的数据框

    :return: data2, DataFrame: 处理后的数据框
    """
    # 根据TWAP策略进行拆单交易
    data2 = data1.copy()
    # 根据时间长度，产生平均分配的权重，初始化为1
    data2['weight_forcast'] = 1
    # 对每个股票在每个交易日的权重进行标准化，确保权重和为1
    data2['weight_forcast'] = data2.groupby(['sec_code', 'trade_day'])['weight_forcast'].transform(
        lambda x: x / x.sum())
    # 重置索引，确保索引是连续的整数序列
    data2.reset_index(drop=True, inplace=True)
    # 将当日预期交易量平均分配到每一个时间段
    data2['volume_trade_i'] = data2['volume_trade'] * data2['weight_forcast']
    # 计算市场中每个交易日每个时间段的交易量占比
    data2['weight_market'] = data2.groupby(['sec_code', 'trade_day'])['volume'].transform(lambda x: x / x.sum())
    # 计算当日每一个交易段真实会产生的交易量，取实际交易量和预期交易量的较小值
    data2['volume_real_i'] = data2.apply(lambda x: min(x['volume'], x['volume_trade_i']), axis=1)
    # 计算每个股票在每个交易日每个时间段的交易量占比，确保权重和为1
    data2['weight_real'] = data2.groupby(['sec_code', 'trade_day'])['volume_real_i'].transform(lambda x: x / x.sum())
    return data2


def twap_result(data2):
    """
    计算绩效评价指标，判断策略执行效果。
    :param data2, DataFrame: 经过twap_vol_dis处理后的数据框

    :return: output, DataFrame: 回测结果数据框
    """
    # 创建一个空的DataFrame用于存储计算后的结果
    output = pd.DataFrame()
    # 从处理后的数据中提取唯一的股票代码（sec_code）和交易日期（trade_day）组合，去除重复的组合
    output[['sec_code', 'trade_day']] = data2[['sec_code', 'trade_day']].drop_duplicates()
    # 重置索引，确保索引是连续的整数序列
    output.reset_index(drop=True, inplace=True)
    # 计算绩效评价指标vdfa，用于衡量预期权重和市场权重之间的关系
    output['vdfa'] = data2.groupby(['sec_code', 'trade_day']).apply(
        lambda x: pow((x['weight_forcast'] - x['weight_market']), 2).sum()).values
    # 计算绩效评价指标oer，用于衡量真实成交的交易量和市场交易量之间的关系
    output['oer'] = data2.groupby(['sec_code', 'trade_day']).apply(
        lambda x: ((x['volume_real_i'].sum()) / x['volume_trade']).mean()).values
    # 计算当日的twap
    output['twap_real'] = data2.groupby(['sec_code', 'trade_day']).apply(
        lambda x: (x['p_close'] * (1 / len(x['p_close']))).sum()).values

    return output


def twap(data, ratio):
    """
    利用算法交易拆单进行交易，减小市场冲击，并进行模拟交易。
    :param data, DataFrame: 包括各标的的收盘价、交易量等数据的原始数据框
    :param ratio, Float: 阈值，默认值为0.6

    :return: data2, DataFrame: 在data基础上添加模拟交易数据字段的数据框
    :return: output, DataFrame: 回测结果，包括twap和指标数据的数据框
    """
    # 调用twap_vol_trade函数，得到处理后的数据data1
    data1 = twap_vol_trade(data, ratio)
    # 调用twap_vol_dis函数，得到处理后的数据data2
    data2 = twap_vol_dis(data1)
    # 调用twap_result函数，得到回测结果output
    output = twap_result(data2)
    return data2, output


# 调用twap函数，传入原始数据和阈值0.6，得到模拟交易数据和回测结果
data_twap, output_twap = twap(data, 0.6)
# 打印模拟交易数据和回测结果
print(data_twap, '\n', output_twap)
# 将output_twap保存为CSV文件
output_twap.to_csv('result/output.csv', index=False)

