import pandas as pd
from typing import Any, List, Dict


class NewsExtractor:
    def __init__(self) -> None:
        self.data = pd.read_csv("./data/news_history.csv")
        self.data["date"] = pd.to_datetime(self.data["datetime"], errors="coerce")
        self.data.sort_values("date", ascending=True, inplace=True)
        keep_cols = {"date", "headline", "summary", "ticker"}
        drop_cols = [c for c in self.data.columns if c not in keep_cols]
        self.data.drop(columns=drop_cols, inplace=True, errors="ignore")
        self.data["date"] = self.data["date"].dt.strftime("%Y-%m-%d")

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

    def extract_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ):
        """
        Return a DataFrame of raw news rows for one ticker (or all tickers if None),
        filtered by an optional date window.
        """
        df = self.data
        df = df[df["ticker"] == ticker]
        df = df[df["date"] >= start_date]
        df = df[df["date"] <= end_date]

        return df

    def extract_news_json(
        self,
        ticker: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_ticker = False
    ) -> List[Dict[str, Any]]:
        """
        Same as extract_news, but returned as a list of dictionaries
        (each dict represents one news item).
        """
        cols = ["date", "headline", "summary"]
        df = self.extract_news(ticker, start_date, end_date)
        
        if include_ticker:
            cols.append("ticker")

        df = df[cols]  
        return df.to_dict(orient="records")

    def extract_news_for_tickers(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Return a DataFrame with news for all supplied tickers over a date window.
        """
        df = self.data[self.data["ticker"].isin(tickers)]
        df = df[df["date"] >= start_date]
        df = df[df["date"] <= end_date]
        
        return df

    def extract_news_json_for_tickers(
        self,
        tickers: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Produce `{ticker: [ {date, headline, summary}, … ]}` for each ticker.
        """
        df = self.extract_news_for_tickers(tickers, start_date, end_date)
        result = {}

        for ticker_symbol, grp in df.groupby("ticker"):
            grp = grp[["date", "headline", "summary"]]
            result[ticker_symbol] = grp.to_dict(orient="records")

        return result