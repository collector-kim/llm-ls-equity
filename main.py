import pandas as pd
from extractor.price_extractor import PriceExtractor
from extractor.news_extractor import NewsExtractor
from extractor.earnings_call_extractor import EarningsCallExtractor
from extractor.financial_statement_extractor import FinancialStatementExtractor

from llm.deep_seek import LLMForFinance
from llm import *

if __name__ == "__main__":
    price_extractor = PriceExtractor()
    news_extractor = NewsExtractor()
    earnings_call_extractor = EarningsCallExtractor()
    financial_statement_extractor = FinancialStatementExtractor()

    # print(earnings_call_extractor.get_previous_quarters_transcripts_json('AAPL', 2025, 'Q1'))

    # print(financial_statement_extractor.get_previous_quarters_statements_json('PYPL', 2024, 'Q4', 2))
    
    # price_universe = price_extractor.show_universe(start_date="2024-02-20", end_date="2025-02-14", min_cnt=50)
    
    # # print(news_extractor.extract_news("PYPL", "2024-10-02", "2025-01-30"))

    # news_universe = news_extractor.show_universe(start_date="2024-02-20", end_date="2025-02-14", min_cnt=None)
    # news_tickers = [data['ticker'] for data in news_universe]

    # # # print(price_extractor.ticker_statistics('PYPL',start_date="2024-02-20", end_date="2025-02-14"))

    model = LLMForFinance()

    # # prediction = model.forecast_price_data('PYPL', "2024-10-21", "2025-01-30")
    # # prediction.to_csv('PYPL_FORECAST_2024102120250130.csv', index=False)
    # predictions = model.forecast_tickers_price_data(tickers, "2024-02-20", "2025-02-14")
    # predictions.to_csv('ALL_FORECAST.csv', index=False)

    # analysis = model.analyze_tickers_sentiments(news_tickers, '2024-02-20', '2025-02-14')
    # analysis.to_csv('SENTIMENT_SCORING.csv', index=False)

    # data = model.forecast_price_data('PYPL', '2024-10-20', '2025-02-02')
    # data = model.forecast_tickers_price_data(news_tickers, '2024-02-20', '2025-02-14')
    # data.to_csv('SENTIMENT_PRICE.csv', index=False)

    # data = model.estimate_stock_ticker('PYPL', '2024-10-20', '2025-02-14')
    # data = model.estimate_tickers(news_tickers, '2024-02-20', '2025-02-14')

    
    tickers = financial_statement_extractor.get_tickers()
    
    earnings_estimate = []


    earnings_estimate.append(model.estimate_ticker_earnings('HIMS', 2024, 'Q4'))
    
    res_df = pd.concat(earnings_estimate)
    res_df.to_csv('EARNINGS_EST.csv', index=False)


    # data.to_csv('TICKER_ESTIMATE_PRICE.csv', index=False)
