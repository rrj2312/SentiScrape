"""
SentiScrape — Web Scraper + Sentiment Analysis Pipeline
---------------------------------------------------------
Scrapes recent news headlines about a brand/product from Google News' public
RSS feed (no login, no API key, no developer account needed), cleans the
text, runs sentiment analysis (VADER), and outputs a ranked CSV + summary
stats + comparison charts across one or more queries.

Usage:
    python sentiscrape.py --query "Havells" --limit 50
    python sentiscrape.py --query "Havells" "Bajaj" "Philips India" --limit 30

Requires: requests, vaderSentiment, pandas, matplotlib
    pip install requests vaderSentiment pandas matplotlib
"""

import argparse
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


HEADERS = {"User-Agent": "Mozilla/5.0 (sentiscrape educational sentiment analysis project)"}


def scrape_news(query: str, limit: int = 50, region: str = "IN", lang: str = "en") -> list[dict]:
    """
    Scrapes recent news article headlines matching a query from Google News'
    public RSS feed. This is a plain XML feed — no auth, no API key, no
    developer account required.
    Returns a list of dicts: {title, source, created, link}
    """
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": lang, "gl": region, "ceid": f"{region}:{lang}"}

    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    items = root.findall(".//item")[:limit]

    articles = []
    for item in items:
        raw_title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pubdate_raw = (item.findtext("pubDate") or "").strip()

        # Google News titles are formatted "Headline - Source Name"; split them apart
        if " - " in raw_title:
            headline, source = raw_title.rsplit(" - ", 1)
        else:
            headline, source = raw_title, "Unknown"

        try:
            created = parsedate_to_datetime(pubdate_raw).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            created = ""

        articles.append({
            "title": headline.strip(),
            "source": source.strip(),
            "created": created,
            "link": link,
        })
    return articles


def clean_text(text: str) -> str:
    """Strip URLs, markdown/HTML artifacts, and excess whitespace."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\*\_\~\^]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> dict:
    """Runs VADER sentiment analysis. Returns compound score + label."""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"compound": compound, "label": label,
            "pos": scores["pos"], "neu": scores["neu"], "neg": scores["neg"]}


def fetch_and_analyze(query: str, limit: int = 50,
                       analyzer: SentimentIntensityAnalyzer = None) -> pd.DataFrame:
    """Scrapes + analyzes one query. Returns a DataFrame tagged with the query name."""
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()

    print(f"  Scraping news for '{query}'...")
    articles = scrape_news(query, limit)
    if not articles:
        print(f"    -> No articles found for '{query}'. Skipping.")
        return pd.DataFrame()

    print(f"    -> {len(articles)} articles retrieved")

    rows = []
    for a in articles:
        cleaned = clean_text(a["title"])
        sentiment = analyze_sentiment(cleaned, analyzer)
        rows.append({
            "query": query,
            "title": a["title"],
            "source": a["source"],
            "created": a["created"],
            "sentiment_label": sentiment["label"],
            "compound_score": sentiment["compound"],
            "link": a["link"],
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, label: str):
    counts = df["sentiment_label"].value_counts()
    total = len(df)
    avg = df["compound_score"].mean()
    print(f"\n{label} (n={total}, avg compound score={avg:+.3f})")
    for lbl in ["positive", "neutral", "negative"]:
        n = counts.get(lbl, 0)
        print(f"  {lbl:8s}: {n:3d}  ({n/total*100:.1f}%)")


def run_pipeline(queries: list[str], limit: int = 50, outfile: str = "sentiment_results.csv"):
    analyzer = SentimentIntensityAnalyzer()
    print(f"[1/3] Scraping and analyzing {len(queries)} quer{'y' if len(queries)==1 else 'ies'}...")

    all_dfs = []
    for i, q in enumerate(queries):
        df = fetch_and_analyze(q, limit, analyzer)
        if not df.empty:
            all_dfs.append(df)
        if i < len(queries) - 1:
            time.sleep(1)  # be polite between requests

    if not all_dfs:
        print("No data retrieved for any query. Try broader search terms.")
        return

    combined = pd.concat(all_dfs, ignore_index=True).sort_values(["query", "compound_score"])

    print(f"\n[2/3] Saving combined results to {outfile}...")
    combined.to_csv(outfile, index=False)

    print("\n[3/3] Summary per query:")
    for q in queries:
        subset = combined[combined["query"] == q]
        if subset.empty:
            continue
        print_summary(subset, q)
        top_pos = subset.nlargest(1, "compound_score").iloc[0]
        top_neg = subset.nsmallest(1, "compound_score").iloc[0]
        print(f"  most positive: [{top_pos.compound_score:+.2f}] {top_pos.title[:70]}")
        print(f"  most negative: [{top_neg.compound_score:+.2f}] {top_neg.title[:70]}")

    if len(queries) == 1:
        _plot_single(combined, queries[0], outfile)
    else:
        _plot_comparison(combined, queries, outfile)


def _plot_single(df: pd.DataFrame, query: str, outfile: str):
    counts = df["sentiment_label"].value_counts().reindex(["positive", "neutral", "negative"])
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#4CAF50", "#9E9E9E", "#F44336"])
    ax.set_title(f"Sentiment Distribution: '{query}'")
    ax.set_ylabel("Number of articles")
    plt.tight_layout()
    chart_file = outfile.replace(".csv", "_chart.png")
    plt.savefig(chart_file, dpi=150)
    print(f"\nChart saved to {chart_file}")


def _plot_comparison(df: pd.DataFrame, queries: list[str], outfile: str):
    """Two comparison charts: (1) sentiment mix per query, (2) average compound score per query."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    pivot = df.groupby(["query", "sentiment_label"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    pivot = pivot.reindex(queries)
    pivot.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#9E9E9E", "#F44336"])
    axes[0].set_title("Sentiment Mix by Query")
    axes[0].set_ylabel("Number of articles")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="")

    avg_scores = df.groupby("query")["compound_score"].mean().reindex(queries)
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in avg_scores]
    avg_scores.plot(kind="bar", ax=axes[1], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Average Sentiment Score by Query")
    axes[1].set_ylabel("Avg compound score (-1 to +1)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    chart_file = outfile.replace(".csv", "_comparison.png")
    plt.savefig(chart_file, dpi=150)
    print(f"\nComparison chart saved to {chart_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape news headlines + compare sentiment across one or more queries.")
    parser.add_argument("--query", required=True, nargs="+",
                         help="One or more search terms, e.g. --query Havells 'Philips India' Bajaj")
    parser.add_argument("--limit", type=int, default=50, help="Max articles to fetch per query (default 50)")
    parser.add_argument("--out", default="sentiment_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    run_pipeline(args.query, args.limit, args.out)