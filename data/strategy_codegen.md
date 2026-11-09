# 日线回测规则知识库

## 日线数据和回测口径

股票使用 BaoStock 前复权日线，字段为 open、high、low、close、volume。规则以当日收盘时能看到的数据为准。市价订单在同一标的下一可交易日开盘模拟成交。基准只用于比较，停牌日线不参与信号和持仓计数。策略规则只声明交易条件，日期、标的、资金和手续费由运行配置提供。

参考资料 https://www.backtrader.com/docu/order/ 和 https://www.backtrader.com/docu/order-creation-execution/order-creation-execution/

## RSI 阈值和均线交叉

RSI 条件写成 kind 为 rsi，填 period、op、threshold。阈值以 0 到 100 表示。RSI 在 14 日低于 30 买入、高于 70 卖出，可写为 {"entry":{"kind":"rsi","period":14,"op":"lt","threshold":30},"exit":{"kind":"rsi","period":14,"op":"gt","threshold":70},"position_fraction":0.1}。程序使用 RSI_Safe 计算。

简单均线交叉写成 kind 为 ma_cross，填 fast、slow、direction。up 表示昨日快线不高于慢线且今日快线高于慢线，down 表示昨日快线不低于慢线且今日快线低于慢线。比较运算 lt、lte、gt、gte 分别是小于、小于等于、大于、大于等于。

## 成交量、价格突破和条件分组

volume_ratio 比较当日成交量除以前 period 根可交易日线的均量，基准窗口排除当天。前 20 日均量的至少 1.5 倍写成 {"kind":"volume_ratio","period":20,"op":"gte","threshold":1.5}。历史均量为零时条件不触发。

breakout 的 direction 为 above 时，要求当日收盘价高于前 period 根日线最高价；below 时要求低于最低价。all 和 any 通过 conditions 列表嵌套，分别表示全部满足和任一满足。例如放量并且 RSI 或均线上穿，将放量与一个 any 子组放在 all 中。

## 仓位和退出

position_fraction 表示每只股票目标建仓金额占账户权益的比例，并受可用现金约束。stop_loss 和 take_profit 使用相对实际买入成交价的小数比例。max_hold_days 从实际成交的日线算第 1 日。指标退出和这些风控条件任一满足即发出卖出信号。用户明确不卖出时用 hold_to_end，期末按可用收盘价估值。
