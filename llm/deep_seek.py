import json
import os
import re
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from dotenv import load_dotenv
from openai import OpenAI
from typing import Any

from extractor.price_extractor import PriceExtractor
from extractor.news_extractor import NewsExtractor
from extractor.earnings_call_extractor import EarningsCallExtractor
from extractor.financial_statement_extractor import FinancialStatementExtractor

from . import *

load_dotenv()

class LLMForFinance:
    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.1, stream: bool = False, max_workers = 10) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY",""), 
                            base_url=os.getenv("DEEPSEEK_URL", ""))
        self.model = model
        self.temperature = temperature        
        self.stream = stream

        self.price_extractor = PriceExtractor()
        self.news_extractor = NewsExtractor()
        self.earnings_extractor = EarningsCallExtractor()
        self.financial_statement_extractor = FinancialStatementExtractor()

        self.max_workers = max_workers
    
    def load_template(self, template: LLMTemplate) -> None:
        match template:
            case LLMTemplate.PRICE:
                self.template = PRICE_TEMPLATE
            case LLMTemplate.PRICE_NEWS:
                self.template = PRICE_SENTIMENT_TEMPLATE
            case LLMTemplate.NEWS:
                self.template = NEWS_TEMPLATE
            case LLMTemplate.ESTIMATE_TICKER:
                self.template = TICKER_TEMPLATE
            case LLMTemplate.ESTIMATE_EARNINGS:
                self.template = EARNINGS_TEMPLATE
            case _:
                raise Exception("Invalid Template Type")

    def analyze_sentiment(self, 
                          ticker: str,
                          date: str,
                          headline: str,
                          summary) -> dict[str, Any]:
        def parse_sentiment_line(line: str) -> dict:
            line = line.strip()
            if not line.startswith("###!SENTIMENT!###"):
                raise ValueError("Line does not start with '###!SENTIMENT!###'")

            payload = line.replace("###!SENTIMENT!###", "").strip()
            parts   = [p.strip() for p in payload.split("|")]

            if len(parts) != 3:
                raise ValueError("Expected exactly three fields separated by '|'")

            score_str = parts[0]
            if not re.fullmatch(r"-?\d+", score_str):
                raise ValueError(f"Score is not an integer: {score_str}")

            score       = int(score_str)
            confidence  = parts[1].title()
            reason      = parts[2]

            if confidence not in {"High", "Medium", "Low"}:
                raise ValueError(f"Confidence must be High, Medium, or Low (got {confidence})")

            return {
                "score": score,
                "confidence": confidence,
                "reason": reason,
            }

        prompt = self.template.format(news_data=str({
            "ticker": ticker,
            "headline": headline,
            "summary": summary
        }))
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional financial analyst."},
                {"role": "user", "content": prompt}
            ],
            stream=self.stream,
            temperature=self.temperature
        )
        result = response.choices[0].message.content
        
        sentiment = parse_sentiment_line(result)

        sentiment.update({
                'ticker': ticker,
                "headline": headline,
                "summary": summary,
                "date": date
            })
        
        print(sentiment)

        return sentiment
        
    def analyze_ticker_sentiments(self,
                                  ticker: str,
                                  start_date: str,
                                  end_date: str) -> pd.DataFrame:
        self.load_template(LLMTemplate.NEWS)

        results = []

        extracted_data = self.news_extractor.extract_news_json(ticker, start_date, end_date, include_ticker=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_item = {
                pool.submit(
                    self.analyze_sentiment,
                    item["ticker"],
                    item["date"],
                    item["headline"],
                    item["summary"]
                ): item
                for item in extracted_data
            }

            for fut in as_completed(future_to_item):
                item = future_to_item[fut]
                try:
                    sentiment_dict = fut.result()      # returns dict from analyze_sentiment
                    results.append(sentiment_dict)
                except Exception as exc:
                    print(f"[{item['ticker']} | {item['date']}] sentiment failed: {exc}")

        return pd.DataFrame(results)
    
    def analyze_tickers_sentiments(self,
                                   tickers: list[str],
                                   start_date: str,
                                   end_date) -> pd.DataFrame:
        results = []

        for ticker in tickers:
            df = self.analyze_ticker_sentiments(ticker, start_date, end_date)
            results.append(df)

        return pd.concat(results)

    def forecast_price_data(self, 
                            ticker: str,
                            start_date: str,
                            end_date: str, 
                            window_size = 30, 
                            with_news=False) -> pd.DataFrame:
        
        if not with_news:
            self.load_template(LLMTemplate.PRICE)
        else:
            self.load_template(LLMTemplate.PRICE_NEWS)

        def extract_price(text: str) ->  float | None:
            pattern = r'###!PRICE!###\s*(\d+\.?\d*)'
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
            return None

        extracted_data = self.price_extractor.extract_ticker_price_json(ticker, start_date, end_date)
        extracted_data = json.loads(extracted_data)

        # extracted_news_data = self.news_extractor.extract_news_json(ticker, start_date, end_date, include_ticker=True)

        # Below CSV for LLM output but could use above methid instead.
        sentiment_df = pd.read_csv('SENTIMENT_SCORING.csv')
        sentiment_df = sentiment_df[sentiment_df['ticker'] == ticker]

        price_data = [{'close': data['close'], 'date': data['date']} for data in extracted_data]

        predictions = []
        
        for i in range(window_size, len(price_data)):
            window = price_data[i - window_size:i]
            last_data = window[-1]
            start_data = window[0]

            date_start = start_data['date']
            date_end = last_data['date']

            sentiment_data = sentiment_df[(sentiment_df['date'] >= date_start) & (sentiment_df['date'] <= date_end)]
            sentiment_data = sentiment_data[['date', 'score', 'confidence']].to_dict(orient="records")

            prompt = self.template.format(price_data=str(window), sentiment_data=str(sentiment_data))
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                stream=self.stream,
                temperature=self.temperature
            )
            result = response.choices[0].message.content
            price = extract_price(result)

            last_close = last_data['close']
            last_date = last_data['date']

            print(ticker, price, last_date, last_close)

            predictions.append({
                'estimated_price': price,
                'last_date': last_date,
                'last_close': last_close,
            })

        df = pd.DataFrame(predictions)
        df = df.copy()
        
        df['last_date'] = pd.to_datetime(df['last_date'])
        df['estimated_date'] = df['last_date'].shift(-1)
        
        df.loc[df['estimated_date'].isna(), 'estimated_date'] = df['last_date'].iloc[-1] + pd.offsets.BDay(1)

        return df


    def forecast_tickers_price_data(self, 
                        tickers: list[str],
                        start_date: str,
                        end_date: str, 
                        window_size = 30) -> pd.DataFrame:
        frames = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_ticker = {
                pool.submit(
                    self.forecast_price_data,
                    t,
                    start_date,
                    end_date,
                    window_size
                ): t
                for t in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    df["ticker"] = ticker     # tag the ticker label
                    frames.append(df)
                except Exception as e:
                    print(f"[{ticker}] forecast failed: {e}")

        return pd.concat(frames, ignore_index=True)
    
    def estimate_stock_ticker(
                self,
                ticker: str,
                start_date: str,
                end_date: str,
                window_size: int = 30,
            ):
        
        self.load_template(LLMTemplate.ESTIMATE_TICKER)
        
        def extract_ticker(text: str) -> str | None:
            pattern = r'###!TICKER!###\s*([A-Z]{1,5})'
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            return None
        
        extracted_data = self.price_extractor.extract_ticker_price_json(ticker, start_date, end_date)
        extracted_data = json.loads(extracted_data)

        price_data = [{'close': data['close'], 'date': data['date']} for data in extracted_data]

        predictions = []
        
        for i in range(window_size, len(price_data)):
            window = price_data[i - window_size:i]
            last_data = window[-1]
            start_data = window[0]
            print(last_data)

            prompt = self.template.format(price_data=str(window))
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                stream=self.stream,
                temperature=self.temperature
            )
            result = response.choices[0].message.content
            ticker_estimate = extract_ticker(result)

            last_close = last_data['close']
            last_date = last_data['date']

            print(ticker, ticker_estimate, last_date, last_close)

            predictions.append({
                'estimated_ticker': ticker_estimate,
                'last_date': last_date,
                'last_close': last_close,
            })
            
        df = pd.DataFrame(predictions)
        df = df.copy()
        
        df['last_date'] = pd.to_datetime(df['last_date'])
        df['estimated_date'] = df['last_date'].shift(-1)
        
        df.loc[df['estimated_date'].isna(), 'estimated_date'] = df['last_date'].iloc[-1] + pd.offsets.BDay(1)

        return df
    
    def estimate_tickers(self, 
                    tickers: list[str],
                    start_date: str,
                    end_date: str, 
                    window_size = 30) -> pd.DataFrame:
        frames = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_ticker = {
                pool.submit(
                    self.estimate_stock_ticker,
                    t,
                    start_date,
                    end_date,
                    window_size
                ): t
                for t in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    df["ticker"] = ticker     # tag the ticker label
                    frames.append(df)
                except Exception as e:
                    print(f"[{ticker}] forecast failed: {e}")

        return pd.concat(frames, ignore_index=True)
    
    def estimate_ticker_earnings(
            self,
            ticker: str,
            year: int,
            quarter: str
    ) -> pd.DataFrame:
        self.load_template(LLMTemplate.ESTIMATE_EARNINGS)
        
        def get_quarter_date_range(year: int, quarter: str) -> tuple[str, str]:
            """Returns the (start_date, end_date) of a given quarter in 'YYYY-MM-DD' format."""
            if quarter == 'Q1':
                start = date(year, 1, 1)
                end = date(year, 3, 31)
            elif quarter == 'Q2':
                start = date(year, 4, 1)
                end = date(year, 6, 30)
            elif quarter == 'Q3':
                start = date(year, 7, 1)
                end = date(year, 9, 30)
            elif quarter == 'Q4':
                start = date(year, 10, 1)
                end = date(year, 12, 31)
            else:
                raise ValueError(f"Invalid quarter: {quarter}")

            return start.isoformat(), end.isoformat()


        def parse_estimated_earnings(result: str) -> tuple[int, float]:
            pattern = r"###EARNINGS###\s*(\d+)\s*\|\s*([0-9.]+)"
            match = re.search(pattern, result)
            
            if match:
                revenue = int(match.group(1))
                eps = float(match.group(2))
                return revenue, eps
            else:
                raise ValueError("Earnings result not found or improperly formatted.")
        
        start_date, end_date = get_quarter_date_range(year, quarter)

        prev_earnings_call = self.earnings_extractor.get_previous_quarters_transcripts_json(ticker, year, quarter, 1)
        prev_earnings = self.financial_statement_extractor.get_previous_quarters_statements_json(ticker, year, quarter, 2)
        news_data = self.news_extractor.extract_news_json(ticker, start_date, end_date, include_ticker=True)

        prompt = self.template.format(prev_earnings_call=str(prev_earnings_call), prev_financials=str(prev_earnings), news_data=str(news_data))
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional financial analyst."},
                {"role": "user", "content": prompt}
            ],
            stream=self.stream,
            temperature=self.temperature
        )
        result = response.choices[0].message.content

        revenue, eps = parse_estimated_earnings(result)

        df = pd.DataFrame([
            {
                'ticker': ticker,
                'est_revenue': revenue,
                'est_eps': eps,
                'year': year,
                'quarter': quarter
            }
        ])
    
        return df