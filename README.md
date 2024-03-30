
# TWAP（Time Weighted Average Price），时间加权平均价格算法

TWAP（Time Weighted Average Price），时间加权平均价格算法，是一种最简单的传统算法交易策略。
该模型将交易时间进行均匀分割，并在每个分割节点上将拆分的订单进行提交。
例如，可以将某个交易日的交易时间平均分为N段，TWAP策略会将该交易日需要执行的订单均匀分配在这N个时间段上去执行，
从而使得交易均价跟踪TWAP。

$$
T W A P=\frac{\sum_{t=1}^N \text { price }_t}{N}
$$

TWAP策略设计的目的是在使交易对市场影响最小化的同时提供一个较低的平均成交价格，从而达到减小交易成本的目的。
在分时成交量无法准确估计的情况下，该模型可以较好地实现算法交易的基本目的。

参考文档：《广发证券——算法交易系列研究之二传统算法交易策略中的相关参数研究》

博客园https://www.cnblogs.com/timlong/p/6701441.html

参考代码：万矿https://www.windquant.com/qntcloud/article?dc74bf11-0bed-4a77-9c86-47d4766b7478