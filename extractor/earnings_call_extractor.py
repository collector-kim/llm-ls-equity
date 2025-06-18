import pandas as pd

class EarningsCallExtractor:
    def __init__(self) -> None:
        self.data = pd.read_csv('./data/earnings_transcripts.csv')
        self.data.drop(columns=['transcript_split'], inplace=True)
        self.data.rename(columns=lambda x: x.strip().lower(), inplace=True)

    def get_previous_quarters_transcripts_df(self, ticker: str, current_year: int, current_quarter: str, n_quarters: int = 2) -> pd.DataFrame:
        df = self.data[self.data['ticker'] == ticker].copy()

        quarter_order = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        df['quarter_num'] = df['quarter'].map(quarter_order)

        df['year'] = df['year'].astype(int)
        df['quarter_num'] = df['quarter_num'].astype(int)

        df['sort_key'] = df['year'] * 10 + df['quarter_num']
        current_key = current_year * 10 + quarter_order[current_quarter]
        df_filtered = df[df['sort_key'] < current_key]
        df_filtered = df_filtered.sort_values('sort_key', ascending=False).head(n_quarters)

        return df_filtered.drop(columns=['quarter_num', 'sort_key'])

    def get_previous_quarters_transcripts_json(self, ticker: str, current_year: int, current_quarter: str, n_quarters: int = 2) -> list[dict]:
        df = self.get_previous_quarters_transcripts_df(ticker, current_year, current_quarter, n_quarters)
        return df.to_dict(orient='records')