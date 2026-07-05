# SentiScrape

A web scraper + sentiment analysis pipeline that pulls live news headlines about any brand or product and scores them for sentiment — with built-in support for comparing multiple brands side by side.

---

## What it does

Give it one or more search terms (a company, product, or brand name) and it will:

1. **Scrape** recent news headlines matching each query from Google News' public RSS feed
2. **Clean** the text (strip URLs, HTML/markdown artifacts, extra whitespace)
3. **Score** each headline's sentiment using VADER (positive / neutral / negative + a compound score from -1 to +1)
4. **Output** a ranked CSV of every headline scored, plus comparison charts when multiple brands are queried

No API key, no login, no developer account required — it hits a public XML feed directly.

## Example output

Comparing sentiment across three companies from 90 live headlines:

```
Havells        (n=30, avg compound score=+0.094)
  positive:  10  (33.3%)
  neutral :  14  (46.7%)
  negative:   6  (20.0%)

Bajaj          (n=30, avg compound score=+0.150)
  positive:  12  (40.0%)
  neutral :  17  (56.7%)
  negative:   1  ( 3.3%)

Philips India  (n=30, avg compound score=+0.197)
  positive:  17  (56.7%)
  neutral :  10  (33.3%)
  negative:   3  (10.0%)
```

Plus two charts: a grouped bar chart of sentiment mix per brand, and an average-sentiment-score comparison (green above zero, red below) — the faster read for "who's ahead."

## Usage

```bash
pip install requests vaderSentiment pandas matplotlib

# Single brand
python sentiscrape.py --query "Havells" --limit 50

# Compare multiple brands in one run
python sentiscrape.py --query "Havells" "Bajaj" "Philips India" --limit 30

# Custom output filename
python sentiscrape.py --query "Havells" --limit 50 --out havells_sentiment.csv
```

**Arguments:**

| Flag | Description |
|---|---|
| `--query` | One or more search terms (required). Multiple terms trigger comparison mode. |
| `--limit` | Max articles to fetch per query (default 50). |
| `--out` | Output CSV filename (default `sentiment_results.csv`). |

## Output files

* `<name>.csv` — every headline with its query, source publication, date, sentiment label, compound score, and article link
* `<name>_chart.png` — sentiment distribution bar chart (single-query mode)
* `<name>_comparison.png` — sentiment mix + average score comparison across brands (multi-query mode)

## How the sentiment scoring works

Each cleaned headline is passed through VADER's `polarity_scores()`, which returns a compound score in `[-1, +1]`:

* `compound >= 0.05` → **positive**
* `compound <= -0.05` → **negative**
* otherwise → **neutral**

VADER was chosen over a transformer-based model for speed and zero setup cost (no model download, no GPU) — appropriate for headline-length text where the marginal accuracy gain from a larger model is small relative to the added complexity.

## Tech stack

| Component | Tool |
|---|---|
| Scraping | `requests` against Google News RSS |
| Parsing | `xml.etree.ElementTree` |
| Sentiment | `vaderSentiment` |
| Data handling | `pandas` |
| Visualization | `matplotlib` |


## Running locally

```bash
git clone https://github.com/rrj2312/SentiScrape.git
cd SentiScrape
pip install requests vaderSentiment pandas matplotlib
python sentiscrape.py --query "your brand here" --limit 50
```

---

Built as a scraping + NLP portfolio project, and as a real-time case study in what happens when your first data source blocks you — the pivot from Reddit to Google News RSS happened mid-build, not in planning.
