"""
temporal_analysis.py

Purpose
-------
The CORE of this research (Steps 11-13):
  11. Final article-level dataset (join sentiment + topics per article)
  12. Weekly temporal aggregation
  13. Trend features (change, growth rate, moving averages)

Does NOT include anomaly detection (rolling z-score) or crisis-window
labeling — those are Step 16 and Steps 14-15 respectively, built in
crisis_analysis.py next, which consumes this module's output.

Tracked topics/keywords/entity-types
--------------------------------------
Per the brief's "focus on interpretable features, don't create hundreds of
columns" guidance, only a small, deliberately chosen set of topics and
keywords get individual frequency columns — the ones with clear crisis
relevance, decided during topics.py's manual naming step:

    fuel_topic_frequency              <- "Fuel & Oil Prices" + "Fuel QR System & Rationing"
    iran_war_topic_frequency          <- "Iran War & Geopolitics"
    electricity_topic_frequency       <- "Electricity Tariffs & Tax"
    corruption_topic_frequency        <- "Arrests, Bribery & Corruption"
    healthcare_strike_topic_frequency <- "Healthcare Worker Strikes (GMOA)"
    consumer_prices_topic_frequency   <- "Consumer Prices & Food"
    disaster_recovery_topic_frequency <- "General Sri Lanka News (Ditwah Relief...)"

    fuel_keyword_frequency, shortage_keyword_frequency,
    protest_keyword_frequency, crisis_keyword_frequency
        <- substring match against article_keywords.keyword

    {person,organization,location,event,money}_entity_frequency
        <- count of article_entities rows per week per entity_type

Adjust TRACKED_TOPIC_GROUPS / TRACKED_KEYWORDS below if your own topic
names came out differently — these must match your actual applied topic
names from topics.apply_topic_names().
"""

import pandas as pd
import utils

logger = utils.setup_logger(__name__)

# Maps each temporal_features column to the topic_name(s) that count toward it.
# Multiple topic names can feed one column (e.g. two separate fuel-related topics).
TRACKED_TOPIC_GROUPS = {
    "fuel_topic_frequency": ["Fuel & Oil Prices (CPC)", "Fuel Quota & QR System"],
    "iran_war_topic_frequency": ["Iran War & Geopolitics (Trump/Hormuz)"],
    "electricity_topic_frequency": ["Electricity Tariffs & CEB"],
    "corruption_topic_frequency": ["Arrests, Bribery & Corruption (CIABOC)"],
    "healthcare_strike_topic_frequency": ["Healthcare Worker Strikes (GMOA)"],
    "consumer_prices_topic_frequency": ["Consumer Prices & Food (Rice/CAA)"],
    "disaster_recovery_topic_frequency": ["General Sri Lanka News (Cyclone Ditwah relief, Rupee, Tourism)"],
}

# Maps each temporal_features column to a keyword substring to match
# (case-insensitive) against article_keywords.keyword.
TRACKED_KEYWORDS = {
    "fuel_keyword_frequency": "fuel",
    "shortage_keyword_frequency": "shortage",
    "protest_keyword_frequency": "protest",
    "crisis_keyword_frequency": "crisis",
}

ENTITY_TYPE_COLUMNS = {
    "PERSON": "person_entity_frequency",
    "ORGANIZATION": "organization_entity_frequency",
    "LOCATION": "location_entity_frequency",
    "EVENT": "event_entity_frequency",
    "MONEY": "money_entity_frequency",
}


def get_article_level_dataset() -> pd.DataFrame:
    """Step 11: fetches the joined article-level dataset as a DataFrame."""
    import database as db
    rows = db.get_article_level_dataset()
    return pd.DataFrame(rows)


def _sentiment_numeric(row) -> float:
    """Converts sentiment label + score into a signed numeric value for averaging:
    positive -> +score, negative -> -score, neutral -> 0."""
    if row["sentiment"] == "positive":
        return row["sentiment_score"] or 0.0
    elif row["sentiment"] == "negative":
        return -(row["sentiment_score"] or 0.0)
    return 0.0


def build_weekly_panel() -> pd.DataFrame:
    """
    Step 12: builds the weekly aggregation panel.
    Returns a DataFrame with one row per ISO week, all base metrics and
    tracked topic/keyword/entity frequency columns populated. Trend
    features (Step 13) are NOT yet added — call add_trend_features() next.
    """
    import database as db

    logger.info("Fetching article-level dataset...")
    articles_df = get_article_level_dataset()
    if articles_df.empty:
        logger.warning("No article-level data found — has the pipeline been run?")
        return pd.DataFrame()

    articles_df["sentiment_numeric"] = articles_df.apply(_sentiment_numeric, axis=1)

    logger.info("Aggregating base weekly metrics...")
    weekly = articles_df.groupby("publication_week").agg(
        article_count=("article_id", "count"),
        average_sentiment=("sentiment_numeric", "mean"),
        negative_count=("sentiment", lambda s: (s == "negative").sum()),
        positive_count=("sentiment", lambda s: (s == "positive").sum()),
    ).reset_index().rename(columns={"publication_week": "week"})

    weekly["negative_sentiment_ratio"] = weekly["negative_count"] / weekly["article_count"]
    weekly["positive_sentiment_ratio"] = weekly["positive_count"] / weekly["article_count"]
    weekly = weekly.drop(columns=["negative_count", "positive_count"])

    logger.info("Aggregating tracked topic frequencies...")
    for col_name, topic_names in TRACKED_TOPIC_GROUPS.items():
        matches = articles_df[articles_df["topic_name"].isin(topic_names)]
        counts = matches.groupby("publication_week").size().rename(col_name)
        weekly = weekly.merge(counts, left_on="week", right_index=True, how="left")
        weekly[col_name] = weekly[col_name].fillna(0).astype(int)

    logger.info("Aggregating tracked keyword frequencies...")
    keywords_df = pd.DataFrame(db.get_keyword_dataset())
    for col_name, substring in TRACKED_KEYWORDS.items():
        if keywords_df.empty:
            weekly[col_name] = 0
            continue
        matched = keywords_df[keywords_df["keyword"].str.contains(substring, case=False, na=False)]
        # count DISTINCT articles per week matching this keyword, not raw keyword-row count
        distinct_counts = matched.groupby("publication_week")["article_id"].nunique().rename(col_name)
        weekly = weekly.merge(distinct_counts, left_on="week", right_index=True, how="left")
        weekly[col_name] = weekly[col_name].fillna(0).astype(int)

    logger.info("Aggregating entity-type frequencies...")
    entities_df = pd.DataFrame(db.get_entity_dataset())
    for entity_type, col_name in ENTITY_TYPE_COLUMNS.items():
        if entities_df.empty:
            weekly[col_name] = 0
            continue
        matched = entities_df[entities_df["entity_type"] == entity_type]
        counts = matched.groupby("publication_week").size().rename(col_name)
        weekly = weekly.merge(counts, left_on="week", right_index=True, how="left")
        weekly[col_name] = weekly[col_name].fillna(0).astype(int)

    weekly = weekly.sort_values("week").reset_index(drop=True)
    logger.info(f"Weekly panel built: {len(weekly)} weeks.")
    return weekly


def add_trend_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Step 13: adds change/growth-rate/moving-average columns to an existing
    weekly panel. All calculations are STRICTLY backward-looking (use only
    current and prior weeks) — required for the no-leakage rule in later
    ML steps.
    """
    weekly = weekly.sort_values("week").reset_index(drop=True)

    weekly["article_count_change"] = weekly["article_count"].pct_change()
    weekly["sentiment_change"] = weekly["average_sentiment"].diff()

    # Growth rate = combined pct change across all tracked topic/keyword/
    # entity columns, averaged — a single interpretable summary rather than
    # one growth-rate column per tracked item (keeps the schema's fixed
    # single-column design while still reflecting the whole tracked set).
    topic_cols = list(TRACKED_TOPIC_GROUPS.keys())
    keyword_cols = list(TRACKED_KEYWORDS.keys())
    entity_cols = list(ENTITY_TYPE_COLUMNS.values())

    weekly["topic_growth_rate"] = weekly[topic_cols].sum(axis=1).pct_change()
    weekly["keyword_growth_rate"] = weekly[keyword_cols].sum(axis=1).pct_change()
    weekly["entity_growth_rate"] = weekly[entity_cols].sum(axis=1).pct_change()

    # Moving averages — min_periods ensures early weeks (with insufficient
    # history) get NaN instead of a misleadingly short-window average.
    weekly["ma_4week"] = weekly["article_count"].rolling(window=4, min_periods=4).mean()
    weekly["ma_8week"] = weekly["article_count"].rolling(window=8, min_periods=8).mean()

    # Replace inf (from pct_change on a zero baseline) with NaN — an infinite
    # growth rate isn't a meaningful feature value.
    weekly = weekly.replace([float("inf"), float("-inf")], pd.NA)

    return weekly


def process_temporal_analysis():
    """
    Full Step 11-13 pipeline: build the weekly panel, add trend features,
    write to temporal_features.
    """
    import database as db

    weekly = build_weekly_panel()
    if weekly.empty:
        print("No data available — nothing written.")
        return weekly

    weekly = add_trend_features(weekly)

    rows = weekly.to_dict(orient="records")
    # NaN/NA -> None for DB insertion
    for row in rows:
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None

    db.upsert_temporal_features(rows)
    print(f"Done. {len(rows)} weekly rows written to temporal_features.")
    return weekly


if __name__ == "__main__":
    process_temporal_analysis()