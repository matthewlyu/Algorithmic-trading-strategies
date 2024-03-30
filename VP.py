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
data = data[:20000]


def vp_vol_trade(data, participation_rate):
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


def vp_vol_dis(data1, participation_rate, ndays):
    """
    VP（Volume Participation），固定百分比成交策略，与VWAP策略类似，都是跟踪市场真实成交量的变化，从而制定相应的下单策略。所不同的是，
    VWAP是在确定某个交易日需要成交数量或成交金额的基础上，对该订单进行拆分交易；而VP则是确定一个固定的跟踪比例，根据市场真实的分段成交量，
    按照该固定比例进行下单。
    :param data1, DataFrame: 处理后的数据框
    :param participation_rate, float: 当前计划交易量是市场前一日成交量的比例
    :param ndays, int: 滚动平均窗口长度

    :return: data2, DataFrame: 处理后的数据框
    """
    # 提取时间列，转换为字符串格式
    time = data1['date_time'].apply(lambda x: x.strftime('%H:%M:%S'))
    # 创建一个新的空的数据框用于保存数据
    data11 = pd.DataFrame()
    # 对于每一天的每一个时间段，预测可能会产生的成交量
    for i in range(len(time)):
        # 选择当前时间段的数据
        data1_i = data1[data1['date_time'].apply(lambda x: x.strftime('%H:%M:%S')) == time[i]]
        # 计算滚动平均成交量作为预测成交量
        volume_forecast = data1_i.groupby('sec_code')['volume'].rolling(ndays).mean()
        data1_i['volume_forcast'] = volume_forecast.values
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
    # 根据预测成交量，按照固定比例进行下单
    data2['volume_trade_i'] = data2['volume_forcast'] * participation_rate
    # 根据市场成交量，计算市场权重
    data2['weight_market'] = data2.groupby(['sec_code', 'trade_day'])['volume'].apply(lambda x: x / x.sum()).values
    # 计算每一个时间段真实交易量，即实际成交量和预测成交量的最小值
    data2['volume_real_i'] = data2.apply(lambda x: min(x['volume'], x['volume_trade_i']), axis=1)
    # 根据每个时间段真实成交量，计算真实权重
    data2['weight_real'] = data2.groupby(['sec_code', 'trade_day'])['volume_real_i'].apply(lambda x: x / x.sum()).values
    return data2


def vp_result(data2):
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


def vp(data, participation_rate, ndays):
    """
    根据成交量预测执行 VP (Volume Prediction) 算法交易策略
    :param data, DataFrame: 原始数据框，包括各标的的分钟收盘价、成交量等数据
    :param participation_rate, float: 当前计划交易量是市场前一日成交量的比例
    :param ndays, int: 滚动平均窗口长度

    :return: data2, DataFrame: 在data基础上添加模拟交易数据字段
    :return: output, DataFrame: 回测结果，包括VP和相关指标数据
    """
    # 调用vp_vol_trade函数，得到处理后的数据data1
    data1 = vp_vol_trade(data, participation_rate)
    # 调用vp_vol_dis函数，得到处理后的数据data2
    data2 = vp_vol_dis(data1, participation_rate, ndays)
    # 调用vp_result函数，得到回测结果output
    output = vp_result(data2)
    return data2, output


# 调用vp函数，传入原始数据，参与度和时间窗口长度，得到模拟交易数据和回测结果
data_vp, output_vp = vp(data, 0.2, 2)
# 打印模拟交易数据和回测结果
print(output_vp)
# 将output_vp保存为CSV文件
output_vp.to_csv('result/output.csv', index=False)
