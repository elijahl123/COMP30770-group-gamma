import pandas as pd
import numpy as np

FILE = "titles_cleaned.csv"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)  # remove ALL whitespace inside names too
    )
    return df

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "")
        if key in norm_map:
            return norm_map[key]
    return None

def load_sentiment_only(path: str) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    df = clean_columns(df)

    sent_title = find_col(df, ["SentimentTitle", "sentiment_title", "SentTitle"])
    sent_head  = find_col(df, ["SentimentHeadline", "sentiment_headline", "SentHeadline"])

    if sent_title is None or sent_head is None:
        raise KeyError(
            f"Couldn't find sentiment columns.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Expected something like SentimentTitle and SentimentHeadline."
        )

    df[sent_title] = pd.to_numeric(df[sent_title], errors="coerce")
    df[sent_head]  = pd.to_numeric(df[sent_head], errors="coerce")

    keep = [sent_title, sent_head]

    for optional in ["Title", "Headline", "Topic", "Source", "IDLink"]:
        col = find_col(df, [optional])
        if col is not None:
            keep.append(col)

    out = df[keep].copy()
    return out, sent_title, sent_head

def bucket_sentiment(x: pd.Series) -> pd.Series:
    bins = [-np.inf, -0.05, 0.05, np.inf]
    labels = ["negative", "neutral", "positive"]
    return pd.cut(x, bins=bins, labels=labels)

def sentiment_report(df: pd.DataFrame, sent_title: str, sent_head: str) -> dict:
    report = {}

    report["rows"] = len(df)
    report["missing_sent_title"] = int(df[sent_title].isna().sum())
    report["missing_sent_head"] = int(df[sent_head].isna().sum())

    report["describe_title"] = df[sent_title].describe()
    report["describe_head"] = df[sent_head].describe()

    pair = df[[sent_title, sent_head]].dropna()
    report["corr_title_vs_head"] = float(pair[sent_title].corr(pair[sent_head])) if len(pair) else np.nan

    df2 = df.copy()
    df2["BucketTitle"] = bucket_sentiment(df2[sent_title])
    df2["BucketHeadline"] = bucket_sentiment(df2[sent_head])

    report["bucket_counts_title"] = df2["BucketTitle"].value_counts(dropna=False)
    report["bucket_counts_head"] = df2["BucketHeadline"].value_counts(dropna=False)

    both_bucketed = df2.dropna(subset=["BucketTitle", "BucketHeadline"])
    report["bucket_disagreement_rate"] = (
        float((both_bucketed["BucketTitle"] != both_bucketed["BucketHeadline"]).mean())
        if len(both_bucketed) else np.nan
    )

    both = df2.dropna(subset=[sent_title, sent_head]).copy()
    both["AbsDiff"] = (both[sent_title] - both[sent_head]).abs()
    report["top_divergent_rows"] = both.sort_values("AbsDiff", ascending=False).head(15)

    return report

def print_report(report: dict, sent_title: str, sent_head: str) -> None:
    print("\n=== SENTIMENT REPORT ===")
    print(f"Rows: {report['rows']}")
    print(f"Missing {sent_title}: {report['missing_sent_title']}")
    print(f"Missing {sent_head}: {report['missing_sent_head']}")

    print("\n--- Describe (Title Sentiment) ---")
    print(report["describe_title"])

    print("\n--- Describe (Headline Sentiment) ---")
    print(report["describe_head"])

    print("\n--- Correlation (Title vs Headline sentiment) ---")
    print(report["corr_title_vs_head"])

    print("\n--- Bucket Counts (Title) ---")
    print(report["bucket_counts_title"])

    print("\n--- Bucket Counts (Headline) ---")
    print(report["bucket_counts_head"])

    print("\n--- Bucket disagreement rate (Title bucket != Headline bucket) ---")
    print(report["bucket_disagreement_rate"])

    print("\n--- Top 15 rows with largest Title vs Headline sentiment absolute difference ---")
    print(report["top_divergent_rows"].to_string(index=False))

def save_outputs(df: pd.DataFrame, report: dict) -> None:
    df.to_csv("sentiment_only_cleaned.csv", index=False)

    report["top_divergent_rows"].to_csv("top_sentiment_divergences.csv", index=False)

    print("\nSaved files:")
    print("- sentiment_only_cleaned.csv")
    print("- top_sentiment_divergences.csv")


if __name__ == "__main__":
    df, sent_title, sent_head = load_sentiment_only(FILE)
    report = sentiment_report(df, sent_title, sent_head)
    print_report(report, sent_title, sent_head)
    save_outputs(df, report)
