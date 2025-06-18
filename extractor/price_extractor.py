import datetime
import pandas as pd

from typing import Any

class PriceExtractor:
    def __init__(self):
        self.data = pd.read_csv("./data/stock_price_history.csv")        
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data.sort_values('date', ascending=True, inplace=True)
        self.data.drop(columns=['uuid', 'open', 'high', 'low', 'stock_splits', 'dividends'], inplace=True)
        self.data['date'] = self.data['date'].dt.strftime("%Y-%m-%d")

    def ticker_statistics(self, ticker: str, start_date: str, end_date: str) -> dict[Any]:
        mask = (
            (self.data["ticker"] == ticker)
            & (self.data["date"] >= start_date)
            & (self.data["date"] <= end_date)
        )

        df = self.data.loc[mask, ["date", 'close']].sort_values("date").copy()

        if df.empty:
            raise ValueError(f"No data for {ticker} between {start_date} and {end_date}")

        df["ret"] = df['close'].pct_change()

        min_price = float(df['close'].min())
        max_price = float(df['close'].max())
        min_price_date = df.loc[df['close'].idxmin(), "date"]
        max_price_date = df.loc[df['close'].idxmax(), "date"]

        max_drop_pct  = float(df["ret"].min() * 100)
        max_gain_pct  = float(df["ret"].max() * 100)
        vol_pct       = float(df["ret"].std(ddof=1) * 100)
        var_95_pct    = float(df["ret"].quantile(0.05) * 100)

        return {
            "ticker": ticker,
            "min_price": round(min_price, 4),
            "min_price_date": min_price_date,
            "max_price": round(max_price, 4),
            "max_price_date": max_price_date,
            "max_single_day_drop_pct": round(max_drop_pct, 4),
            "max_single_day_gain_pct": round(max_gain_pct, 4),
            "volatility_pct": round(vol_pct, 4),
            "var_95_pct": round(var_95_pct, 4),
        }
    
    def show_universe(self, start_date: str | None = None, end_date: str | None = None, min_cnt: int | None = 50) -> list[dict[str, Any]]:
        summary = self.data.copy()
        
        if start_date:
            summary = summary[summary['date'] >= start_date]
        
        if end_date:
            summary = summary[summary['date'] <= end_date]        
        
        summary = (
            summary.groupby("ticker")["date"]
            .agg(
                start_date="min",
                end_date="max",
                days="nunique"
            )
            .reset_index()
        )

        if min_cnt:
            summary = summary[summary['days'] >= min_cnt]

        result = summary.to_dict("records")

        return result

    def extract_ticker_price(self, ticker: str, start_date: str, end_date: str):
        df = self.data[self.data['ticker'] == ticker]
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        return df
    
    def extract_tickers_price(self, tickers: list[str], start_date: str, end_date: str):
        df = self.data[self.data['ticker'].isin(tickers)]
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return df
    
    def extract_ticker_price_json(self, 
                           ticker: str, 
                           start_date: str, 
                           end_date: str) -> list[dict[str, Any]]:
        df = self.extract_ticker_price(ticker, start_date, end_date)
        df.drop(columns=['ticker', 'volume'], inplace=True)
        return df.to_json(orient="records")
    
    def extract_tickers_price_json(self, 
                                   tickers: list[str], 
                                   start_date: str, 
                                   end_date: str) -> dict[str, list[dict[str, Any]]]:
        
        df = self.data[self.data['ticker'].isin(tickers)]
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        
        result = {}
        
        for ticker_symbol, grp in df.groupby('ticker'):
            grp = grp.drop(columns=['ticker', 'volume'])
            result[ticker_symbol] = grp.to_dict(orient="records")

        return result
    