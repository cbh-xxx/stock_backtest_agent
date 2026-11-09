import os
import sys
import pandas as pd
from datetime import datetime
import re
import backtrader as bt
# 添加项目根目录到 Python 路径，解决直接运行时的导入问题
current_dir = os.path.dirname(os.path.dirname(__file__))
project_root = os.path.dirname(current_dir)  # Agent 项目目录sys.path.insert(0, project_root)
sys.path.insert(0, project_root)

from utils.logger_handler import logger
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_strategy_generator_prompts

rag = RagSummarizeService()

# @tool(description="获取完整股票池和市场指数的历史数据")
def get_complete_market_data(start_date: str = None, end_date: str = None) -> dict:
    """
    从本地文件获取完整股票池和市场指数的历史数据

    参数:
    - start_date: 开始日期 (YYYY-MM-DD格式，可选)
    - end_date: 结束日期 (YYYY-MM-DD格式，可选)

    返回:
    - 包含股票数据和指数数据的字典，格式为:
      {
          "stocks": {stock_code: pandas.DataFrame},
          "indices": {index_code: pandas.DataFrame}
      }
    """
    # 读取股票数据
    stocks_path = get_abs_path("stock_data/stocks_all.csv")
    indices_path = get_abs_path("stock_data/indices_all.csv")

    if not os.path.exists(stocks_path):
        raise FileNotFoundError(f"股票数据文件{stocks_path}不存在")
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"指数数据文件{indices_path}不存在")

    # 读取CSV文件
    stocks_df = pd.read_csv(stocks_path)
    indices_df = pd.read_csv(indices_path)

    # 确保日期列是datetime类型
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    indices_df['date'] = pd.to_datetime(indices_df['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # 日期过滤
    stocks_df = stocks_df[stocks_df['date'] >= start_date]
    indices_df = indices_df[indices_df['date'] >= start_date]
    stocks_df = stocks_df[stocks_df['date'] <= end_date]
    indices_df = indices_df[indices_df['date'] <= end_date]

    # 按股票代码分组
    stocks_data = {}
    for code, group in stocks_df.groupby('code'):
        stocks_data[code] = group.set_index('date')

    # 按指数代码分组
    indices_data = {}
    for code, group in indices_df.groupby('code'):
        indices_data[code] = group.set_index('date')

    return {
        "stocks": stocks_data,
        "indices": indices_data
    }


# @tool(description="基于向量库匹配策略并生成Backtrader代码，以纯字符串形式返回")
def parse_strategy_with_rag(strategy_description: str) -> str:
    try:
        # 参数验证
        if not strategy_description or not isinstance(strategy_description, str):
            raise ValueError("策略描述不能为空且必须是字符串")

        context_docs = rag.retriever_docs(strategy_description)
        if not context_docs:
            logger.warning("未找到相关策略参考资料")

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        prompt=load_strategy_generator_prompts()

        # 构建完整的prompt
        prompt_text = f"""用户输入的策略需求：{strategy_description}
        检索到的相关资料：{context}，
        请基于以下要求生成满足用户需求的Backtrader策略代码：{prompt}
        """
        # print(prompt_text)

        # 复用项目的 DeepSeek 模型，并提取正文供回测流程解析日期和代码块。
        return (chat_model | StrOutputParser()).invoke(prompt_text).strip()
    except Exception as e:
        logger.error(f"[parse_strategy_with_rag]生成策略代码出错，{str(e)}")
        raise e
    

@tool(description="基于向量库匹配策略并生成Backtrader代码，获取数据并执行代码，生成分析报告，以纯字符串形式返回分析报告")
def execute_backtest(strategy_description: str) -> str:
    """
    执行Backtrader回测并返回分析结果
    
    参数:
    - strategy_description: 策略描述字符串
    - whole_data: 包含股票和指数数据的字典
    
    返回:
    - 回测结果分析字符串
    """
    strategy_code=parse_strategy_with_rag(strategy_description)
    print(f"生成的策略代码:\n{strategy_code}")
    try:
        # 提取回测时间范围

        pattern = r'\b\d{4}-\d{2}-\d{2}\b'
        dates = re.findall(pattern, strategy_code)
        # 假设第一个日期是开始，最后一个日期是结束
        start_date = dates[0]
        end_date = dates[1]
        whole_data = get_complete_market_data(start_date=start_date, end_date=end_date)

        # 提取代码块
        code_start = '2024-01-01'
        code_end = '2024-03-31'   
        code_start = strategy_code.find("```python")
        code_end = strategy_code.rfind("```")
        if code_start == -1 or code_end == -1:
            raise ValueError("无效的代码格式")
        
        pure_code = strategy_code[code_start + 9:code_end].strip()
        
        # 动态执行策略代码
        exec(pure_code, globals())
        
        # 创建回测引擎
        cerebro = bt.Cerebro()
        
        # 添加股票数据
        stock_data = whole_data.get("stocks", {})
        for code, df in stock_data.items():
            # 数据验证
            if df.empty:
                logger.warning(f"股票 {code} 数据为空，跳过")
                continue
                
            # 确保数据有足够的长度
            if len(df) < 30:
                logger.warning(f"股票 {code} 数据不足，需要至少30个数据点，实际只有 {len(df)} 个")
                continue
                
            # 确保列名正确
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"股票 {code} 缺少必要的列")
                continue
            
            # 添加数据到回测引擎
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data, name=code)
        
        # 添加指数数据
        indices_data = whole_data.get("indices", {})
        for code, df in indices_data.items():
            if df.empty:
                continue
                
            # df['date'] = pd.to_datetime(df['date'])
            # df.set_index('date', inplace=True)
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data, name=code)
        
        # 添加策略
        cerebro.addstrategy(MyStrategy)
        
        # 设置初始资金和交易参数
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.001)  # 0.1%手续费
        
        # 添加分析器
        # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')


        # 运行回测
        results = cerebro.run()

        # 获取策略实例
        strategy_instance = results[0]
        
        # 获取分析结果
        # sharpe_analysis = strategy_instance.analyzers.sharpe.get_analysis()
        drawdown_analysis = strategy_instance.analyzers.drawdown.get_analysis()
        trade_analysis = strategy_instance.analyzers.trades.get_analysis()


        # 计算最大回撤
        max_drawdown = 0.0
        if isinstance(drawdown_analysis, dict):
            max_drawdown = drawdown_analysis.get('max', {}).get('drawdown', 0.0)
        elif hasattr(drawdown_analysis, 'max') and hasattr(drawdown_analysis.max, 'drawdown'):
            max_drawdown = drawdown_analysis.max.drawdown

        # 计算交易统计
        total_trades = 0
        won_trades = 0
        if isinstance(trade_analysis, dict):
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            won_trades = trade_analysis.get('won', {}).get('total', 0)
        elif hasattr(trade_analysis, 'total') and hasattr(trade_analysis.total, 'total'):
            total_trades = trade_analysis.total.total
            won_trades = trade_analysis.won.total if hasattr(trade_analysis.won, 'total') else 0

        # 计算胜率
        win_rate = 0.0
        if total_trades > 0:
            win_rate = (won_trades / total_trades) * 100
        
# 生成详细分析报告
        analysis = f"""
回测完成！
----------------------------
资金情况:
初始资金: ¥{cerebro.broker.startingcash:,.2f}
最终资金: ¥{cerebro.broker.getvalue():,.2f}
总收益: ¥{cerebro.broker.getvalue() - cerebro.broker.startingcash:,.2f}
总收益率: {(cerebro.broker.getvalue() - cerebro.broker.startingcash) / cerebro.broker.startingcash * 100:.2f}%
----------------------------
交易统计:
总交易次数: {trade_analysis.total.total}
盈利交易: {trade_analysis.won.total}
亏损交易: {trade_analysis.lost.total}
胜率: {trade_analysis.won.total / trade_analysis.total.total * 100:.1f}% (如果有交易)
----------------------------
回测详情:
回测日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
交易标的数量: {len(stock_data)}
基准指数数量: {len(indices_data)}
----------------------------
详细分析:
最大回撤: {drawdown_analysis.max.drawdown:.2f}%
----------------------------
注: 以上回测结果为模拟结果，实际投资需谨慎。
"""
# 夏普比率: {sharpe_analysis.sharperatio:.3f}

        return analysis
        
    except Exception as e:
        return f"回测执行失败: {str(e)}"


if __name__ == "__main__":

#     strategy_code='''
# ### 回测时间范围
# 开始时间：2024-01-01
# 结束时间：2024-03-31
#
# ### 代码块
# ```python
# import backtrader as bt
#
# class MyStrategy(bt.Strategy):
#     params = (
#         ('period', 20),  # 计算最高价的周期
#         ('buy_percentage', 0.05),  # 买入时使用的资金比例
#         ('stop_loss', 0.08),  # 止损百分比
#         ('take_profit', 0.10),  # 止盈百分比
#         ('max_hold_days', 10)  # 最大持有天数
#     )
#
#     def __init__(self):
#         self.indicators = {}
#         for d in self.datas:
#             if len(d.lines.lines[0].array) >= self.params.period:  # 确保至少有params.period个数据点
#                 self.indicators[d._name] = {
#                     'high_20': bt.indicators.Highest(d.high, period=self.params.period)
#                 }
#             else:
#                 self.indicators[d._name] = None
#
#         self.buy_prices = {}  # 用于存储每只股票的买入价格
#         self.hold_days = {}  # 用于跟踪每只股票的持有天数
#
#     def next(self):
#         for d in self.datas:
#             # 检查指标是否可用
#             if self.indicators[d._name] is None or len(d.lines.lines[0].array) < self.params.period:
#                 continue  # 跳过数据不足的股票
#
#             high_20 = self.indicators[d._name]['high_20']
#
#             # 检查成交量放大和价格突破条件
#             if d.volume[0] > 1.1 * d.volume[-1] and d.close[0] > high_20[-1]:
#                 # 检查是否已经持有该股票
#                 if not self.getposition(d).size:
#                     # 计算买入金额
#                     buy_amount = self.broker.getvalue() * self.params.buy_percentage
#                     size = int(buy_amount / d.close[0])  # 计算买入数量
#                     # 买入股票
#                     self.buy(data=d, size=size)
#                     # 记录买入价格和持有天数
#                     self.buy_prices[d._name] = d.close[0]
#                     self.hold_days[d._name] = 0
#                     self.log(f'BUY  {d._name} at {d.close[0]:.2f}, size: {size:.2f}')
#
#             # 检查止损、止盈或最大持有天数
#             if self.getposition(d).size:
#                 current_price = d.close[0]
#                 buy_price = self.buy_prices[d._name]
#                 hold_days = self.hold_days[d._name]
#
#                 # 检查止损条件
#                 if (current_price - buy_price) / buy_price <= -self.params.stop_loss:
#                     self.sell(data=d, size=self.getposition(d).size)
#                     self.log(f'SELL {d._name} at {current_price:.2f} (STOP LOSS)')
#                 # 检查止盈条件
#                 elif (current_price - buy_price) / buy_price >= self.params.take_profit:
#                     self.sell(data=d, size=self.getposition(d).size)
#                     self.log(f'SELL {d._name} at {current_price:.2f} (TAKE PROFIT)')
#                 # 检查最大持有天数
#                 elif hold_days >= self.params.max_hold_days:
#                     self.sell(data=d, size=self.getposition(d).size)
#                     self.log(f'SELL {d._name} at {current_price:.2f} (MAX HOLD)')
#                 # 更新持有天数
#                 else:
#                     self.hold_days[d._name] += 1
#
#     def log(self, txt, dt=None):
#         """简单的日志函数，可打印交易信息"""
#         dt = dt or self.datas[0].datetime.date(0)
#         print(f'{dt.isoformat()} - {txt}')
# ```
#
# ### 代码说明
# 1. **初始化** (`__init__` 方法):
#    - 初始化 `indicators` 字典，用于存储每只股票的最高价指标。
#    - 初始化 `buy_prices` 和 `hold_days` 字典，分别用于存储每只股票的买入价格和持有天数。
#
# 2. **主逻辑** (`next` 方法):
#    - 遍历所有股票数据，检查每只股票的数据是否足够。
#    - 检查成交量放大和价格突破条件，如果满足则买入股票。
#    - 检查已持有的股票是否达到止损、止盈或最大持有天数条件，如果满足则卖出股票。
#
# 3. **日志记录** (`log` 方法):
#    - 提供一个简单的日志函数，用于记录交易信息。'''


    strategy_code=parse_strategy_with_rag("我想测试一个基于成交量放量的突破策略。具体来说，我希望策略能够监控每只股票的成交量变化，当某只股票的成交量突然放大，并且当天的收盘价也突破了最近20个交易日的最高价时，就买入这只股票。买入时使用账户总资金的5%，并且设置止损条件，如果股价从买入价下跌了8%就卖出止损。如果没有触发止损，就持有股票直到股价上涨了10%时止盈卖出。如果股价既没有达到止损也没有达到止盈，就持有最多10个交易日，无论盈亏都卖出。回测的时间范围是从2023年6月1日到2023年12月31日。这个策略应该同时监控多只股票，但每只股票独立计算信号和持仓。")
    print(strategy_code)
    print(execute_backtest("我想测试一个基于成交量放量的突破策略。具体来说，我希望策略能够监控每只股票的成交量变化，当某只股票的成交量突然放大，并且当天的收盘价也突破了最近20个交易日的最高价时，就买入这只股票。买入时使用账户总资金的5%，并且设置止损条件，如果股价从买入价下跌了8%就卖出止损。如果没有触发止损，就持有股票直到股价上涨了10%时止盈卖出。如果股价既没有达到止损也没有达到止盈，就持有最多10个交易日，无论盈亏都卖出。回测的时间范围是从2023年6月1日到2023年12月31日。这个策略应该同时监控多只股票，但每只股票独立计算信号和持仓。"))


    
