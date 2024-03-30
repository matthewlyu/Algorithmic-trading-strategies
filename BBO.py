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

# 创建一个空的DataFrame用于存储订单簿数据
order_book_df = pd.DataFrame()
order_book_df[['date_time']] = data_all[['date']]

# 将买卖价和对应的交易量添加到DataFrame中
for i in range(1, 6):
    order_book_df[[f'buy{i}', f'bc{i}', f'sale{i}', f'sc{i}']] = data_all[[f'buy{i}', f'bc{i}', f'sale{i}', f'sc{i}']]


# BBO策略
def bbo_strategy(order_book_df, trade_volume):
    """
    BBO（Best Bid and Offer）是一种算法交易中的最优出价策略，用于确定买入和卖出的最佳价格。BBO策略基于市场上的最佳买价（Best Bid）
    和最佳卖价（Best Offer）。最佳买价是指市场上最高的买入价格，而最佳卖价是指市场上最低的卖出价格。BBO策略的目标是在买入时以最低的价格买入，
    并在卖出时以最高的价格卖出。这样可以最大程度地减少交易成本，并获得更好的交易执行。

    :param order_book_df: DataFrame，包含订单簿数据的DataFrame
    :param trade_volume: int，交易量

    :return trades: list，包含交易信息的列表
    """
    # 初始化一个空的交易记录列表
    trades = []
    # 迭代订单簿DataFrame的每一行
    for _, row in order_book_df.iterrows():
        # 获取当前行的时间信息
        t = row['date_time']
        # 获取最佳买价和对应的买量
        best_bid, bid_volume = row['buy1'], row['bc1']
        # 获取最佳卖价和对应的卖量
        best_ask, ask_volume = row['sale1'], row['sc1']

        # 检查是否有足够的交易量以执行交易
        if bid_volume >= trade_volume and ask_volume >= trade_volume:
            # 买入
            buy_price = best_bid
            buy_volume = trade_volume

            # 卖出
            sell_price = best_ask
            sell_volume = trade_volume

            # 更新交易记录
            trades.append((t, buy_price, buy_volume, sell_price, sell_volume))
        else:
            # 如果买卖量不足，添加空值到交易记录
            trades.append((t, np.nan, np.nan, np.nan, np.nan))
    return trades


# 执行BBO策略，设置交易量为1
trade_volume = 1
# 执行BBO策略，获取交易记录
trades = bbo_strategy(order_book_df, trade_volume)

# 将交易数据转换为DataFrame
trades_df = pd.DataFrame(trades, columns=['date_time', 'buy_price', 'buy_volume', 'sell_price', 'sell_volume'])
# 将交易数据保存为CSV文件
trades_df.to_csv('result/trades.csv', index=False)

