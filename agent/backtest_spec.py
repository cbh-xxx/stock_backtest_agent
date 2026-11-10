from datetime import date


DEFAULT_STOCKS = ("sh.600000", "sh.600519", "sz.000001")
EARLIEST_DATE = date(1990, 1, 1)
BENCHMARK_CODES = {
    "上证指数": "sh.000001",
    "沪深300": "sh.000300",
    "中证500": "sh.000905",
    "深证成指": "sz.399001",
}


class BacktestInputError(ValueError):
    pass
