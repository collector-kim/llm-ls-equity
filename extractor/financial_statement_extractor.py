import pandas as pd

class FinancialStatementExtractor:
    def __init__(self) -> None:
        self.data = pd.read_csv('./data/financial_statement_history.csv')
        self.data.rename(columns=lambda x: x.strip().lower(), inplace=True)

    def get_tickers(self) -> list[str]:
        return list(self.data['ticker'].unique())

    def get_previous_quarters_statements_df(
        self, ticker: str, current_year: int, current_quarter: str, n_quarters: int = 2
    ) -> pd.DataFrame:
        df = self.data[self.data['ticker'] == ticker].copy()

        quarter_order = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        df['quarter_num'] = df['quarter'].map(quarter_order)
        df['year'] = df['year'].astype(int)
        df['quarter_num'] = df['quarter_num'].astype(int)
        df['sort_key'] = df['year'] * 10 + df['quarter_num']

        current_key = current_year * 10 + quarter_order[current_quarter]
        df = df[df['sort_key'] < current_key]

        recent_quarters = (
            df[['year', 'quarter', 'sort_key']]
            .drop_duplicates()
            .sort_values('sort_key', ascending=False)
            .head(n_quarters)
        )

        df_filtered = df.merge(recent_quarters[['year', 'quarter']], on=['year', 'quarter'], how='inner')
        df_filtered = df_filtered.drop(columns=['quarter_num', 'sort_key'])

        return df_filtered.sort_values(['quarter', 'year'], ascending=False)


    def get_previous_quarters_statements_json(
        self, ticker: str, current_year: int, current_quarter: str, n_quarters: int = 2
    ) -> list[dict]:
        df = self.get_previous_quarters_statements_df(ticker, current_year, current_quarter, n_quarters)
        grouped = df.groupby(['year', 'quarter'])

        result = []
        for (year, quarter), group in grouped:
            date = group['date'].iloc[0]
            bs_metrics = group[group['financial_statement'] == 'balance_sheet'][['metric', 'value']]
            is_metrics = group[group['financial_statement'] == 'income_statement'][['metric', 'value']]

            result.append({
                "year": int(year),
                "quarter": quarter,
                "date": date,
                "balance_sheet": dict(zip(bs_metrics['metric'], bs_metrics['value'])),
                "income_statement": dict(zip(is_metrics['metric'], is_metrics['value'])),
            })

        return result