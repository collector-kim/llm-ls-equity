from enum import Enum

class LLMTemplate:
    PRICE = "PRICE"
    PRICE_NEWS = "PRICE_NEWS"
    NEWS = "NEWS"
    PRICE_NEWS_EARNINGS_CALL = "PRICE_NEWS_EARNINGS_CALL"
    ESTIMATE_TICKER = "ESTIMATE_TICKER"
    ESTIMATE_EARNINGS = "ESTIMATE_EARNINGS"

PRICE_TEMPLATE = """
You are a professional financial analyst.

Below is **30 days** of daily closing price data for a stock, in JSON format. Each record contains:
- "date" (YYYY-MM-DD)
- "close" (float, closing price for that day)

```json
{price_data}

Based only on the trend and information in the above data, predict the next business day's closing price.
Instructions:
Only consider the numerical trend in the last 30 days.

DO NOT use any external knowledge or past training data.

Ensure your prediction is a reasonable continuation based on the most recent values.

Do not deviate more than ±5% from the most recent close price unless there is a clear pattern.

Output must be in the format:

###!PRICE!### <predicted_close_price> """

PRICE_SENTIMENT_TEMPLATE = """
You are a professional financial analyst.

Below is **30 days** of daily closing price data for a stock, in JSON format. Each record contains:
- "date" (YYYY-MM-DD)
- "close" (float, closing price for that day)

```json
{price_data}

You are also provided with the daily news-sentiment summary you already
computed elsewhere, in JSON format.  Each record contains:
	•	“date”       (YYYY-MM-DD)        ← publication date (or end-of-day bucket)
	•	“score”      (integer 1-10)      ← overall sentiment score (5 being irrelavant / neutral)
	•	“confidence” (“High”/“Medium”/“Low”)

```json
{sentiment_data}

Instructions:
• Base your prediction only on the information above
– the 30-day price trend and the sentiment summaries.
• If the sentiment points to a significant upward or downward move, you may
adjust your forecast accordingly; otherwise, rely mainly on the price trend.
• Do not deviate more than ±5 % from the most recent close unless the
sentiment shows a clear catalyst with High confidence.
• No outside knowledge or training data.

Output must be in the format:
###!PRICE!### <predicted_close_price>"""

NEWS_TEMPLATE  = """
You are a professional financial-markets analyst.

Below is a set of recent news items about **one** stock, supplied in JSON
format. Each record contains:

- "ticker"   (string, the company’s symbol)
- "date"     (YYYY-MM-DD, publication date)
- "headline" (string)
- "summary"  (string: 1-3-sentence abstract of the article)

```json
{news_data}

Your task
	1.	Read only the article above; do not use any external information.
	2.	Judge the net tone, relevance, and likely market impact on the ticker.
	3.	Produce three items:

• Sentiment score – integer 1 (very negative) … 10 (very positive)
(use 5 if the item is neutral or irrelevant).
• Confidence level – High, Medium, or Low based on how clear,
consistent, and forceful the signal is.
• Reason for the chosen confidence – one concise phrase or sentence
explaining why the confidence is High/Medium/Low
(e.g., “single source, mixed tone” or “multiple upbeat metrics”).

Scoring guide

Score 8–10  → clear positive catalyst (earnings beat, major deal, favourable ruling).
Score 6–7   → moderately positive overall news flow.
Score 5 → neutral or news appears unrelated / immaterial to fundamentals.
Score 3–4   → moderately negative.
Score 1–2   → clear negative catalyst (earnings miss, litigation, leadership scandal).

Confidence guide

High   → several consistent, highly relevant articles pointing the same way.
Medium → mixed signals or limited but clear relevance.
Low    → few articles, conflicting signals, or weak relevance.

Output format

Return exactly one line:

###!SENTIMENT!### <score_integer> | <confidence: High/Medium/Low> | <reason>
"""


TICKER_TEMPLATE = """
You are a professional equity analyst.

Below is window of daily closing price data for a single US-listed stock, formatted in JSON. Each record includes:
- "date" (YYYY-MM-DD)
- "close" (float, closing price for that day)

```json
{price_data}
Based only on the price pattern and magnitude over the 30-day window, guess the most likely stock ticker this data belongs to. The stock is listed on a US exchange (e.g., NASDAQ, NYSE).

Instructions:

Use historical price movement, trends, and approximate value ranges to inform your guess.

Consider overall volatility, directional patterns, and levels as possible clues.

If multiple tickers could fit, choose the most probable based on the shape and scale of the trend.

Do not hallucinate prices or rely on memorized company events.

Output your answer in the format:
###!TICKER!### <uppercase_ticker_symbol>
"""

EARNINGS_TEMPLATE = """
You are a professional equity analyst.

Your goal is to forecast **next quarter's Revenue and EPS (Earnings Per Share)** for a public company, based on the provided data.

---

### DATA FORMAT

You are provided with three datasets:

1. **Historical Financial Statements** – last **2 quarters**  
Each item is a dictionary with:
- `"year"` (int)
- `"quarter"` (str: one of "Q1", "Q2", "Q3", "Q4")
- `"date"` (str, YYYY-MM-DD, end of fiscal quarter)
- `"balance_sheet"` (dict: financial metrics as key-value pairs in thousands USD)
- `"income_statement"` (dict: financial metrics as key-value pairs in thousands USD)

```json
{prev_financials}

2. **Previous Quarter's Earnings Call Transcript** – 1 item
Each item is a dictionary with:
- `"year"` (int)
- `"quarter"` (str: one of "Q1", "Q2", "Q3", "Q4")
- `"transcript"` (str, full earnings call transcript text)

```json
{prev_earnings_call}

3. **News Sentiment Articles** – current quarter
Each item contains:
- `"ticker"` (str, company ticker)
- `"date"` (str, YYYY-MM-DD)
- `"headline"` (str, article headline)
- `"summary"` (str, 1–3 sentence summary)

Note: Not all news may be relevant or material to this quarter's earnings. Use your judgment and give more weight to articles that appear financially impactful or mention guidance, performance, or strategy.
```json
{news_data}

**INSTRUCTIONS**
Using the datasets above:
- Analyze revenue and EPS trends from the last 2 quarters.
- Use balance sheet changes (e.g. cash, working capital, debt, equity) to understand financial momentum or stress.
- Use previous earnings call to assess management tone, forward guidance, macro commentary, and new initiatives.
- Use recent news, but recognize that some articles may be irrelevant or only marginally connected to earnings—apply discretion accordingly.
- Assume all financial values are in thousands USD (e.g. Revenue of 45678 means $45.678M).
- If data is insufficient, provide a conservative estimate based on available trends.

Do not fabricate any numbers. Only rely on the data provided.

**OUTPUT FORMAT (MANDATORY)**
Your final estimate should appear exactly in this format:

###EARNINGS### <estimated_revenue_in_thousands> | <estimated_eps>

Example:
###EARNINGS### 52480000 | 0.88

Do not include commentary or explanations after this line.
"""