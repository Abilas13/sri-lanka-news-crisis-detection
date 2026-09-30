# Early Detection of Emerging Crisis Signals in Sri Lankan News Using NLP and Temporal Analysis

A capstone research project investigating whether measurable patterns in Sri Lankan English-language news — sentiment, topics, keywords, and named entities — can serve as early indicators of emerging national crises.

> **Research question:** Can measurable patterns in Sri Lankan English-language news provide early signals of emerging crises?

This is **not** a system for predicting unknown future crises. It is a retrospective, evaluated study of whether historical Sri Lankan news data contains measurable temporal patterns that preceded two real, documented crisis events — and, if so, how much early warning that signal could plausibly have provided.

---

## Headline Result

At a relaxed decision threshold, the pipeline detected both studied crises with a consistent **~18–19 day lead time** before their official onset, using rolling anomaly detection and leave-one-crisis-out validated machine learning — while also finding that **sentiment analysis alone was a poor predictor** (only 45% agreement with hand-labeled ground truth), and that **topic/keyword frequency carried the real signal** instead.

| Metric | Result |
|---|---|
| Articles processed | 16,794 (99.99% passed validation) |
| Date range | Sept 2023 – Aug 2026 |
| Crisis events studied | Cyclone Ditwah (natural disaster) · 2026 Fuel Crisis (geopolitical/economic) |
| Sentiment model accuracy vs. ground truth | 45.0% (n=120 hand-labeled) |
| Peak anomaly score at crisis onset | 13.92 (z-score, threshold = 2.0) |
| Lead time (threshold = 0.3) | ~18–19 days, both crises |
| ROC-AUC (Random Forest, LOCO validation) | 0.66 – 0.68 |

---

## Pipeline Overview

```
scraped_news (raw, untouched)
      |
      v
Validation -> Date Normalization -> Duplicate Detection -> Text Cleaning
      |
      +--> Sentiment Analysis      (cardiffnlp/twitter-roberta-base-sentiment-latest)
      +--> Topic Modeling          (BERTopic + all-MiniLM-L6-v2)
      +--> Keyword Extraction      (KeyBERT)
      +--> Named Entity Recognition (spaCy en_core_web_sm)
      |
      v
Weekly Temporal Aggregation -> Trend Features -> Rolling Z-Score Anomaly Detection
      |
      v
Pre-Crisis Window Labeling (leakage-free) -> ML Classification (Logistic Regression, Random Forest)
      |
      v
Lead-Time Evaluation
```

Every stage writes to its own MySQL table, keyed by `article_id` back to the original untouched `scraped_news` table. Full schema in [`sql/create_tables.sql`](sql/create_tables.sql).

---

## Project Structure

```
|-- config.py                  # DB connection settings, pipeline constants
|-- requirements.txt
|-- .env.
|
|-- sql/
|   |-- create_tables.sql       # 8 processed-data tables (no FKs -- see Notes)
|   |-- indexes.sql
|   |-- seed_crisis_events.sql
|  
|-- data/
|   |-- scraped_news/
|   |-- processed/   
|
|-- src/
|   |-- database.py              # all DB connectivity/queries
|   |-- utils.py                 # shared helpers (hashing, ISO week conversion, logging)
|   |-- cleaning.py               # validation, text cleaning, basic features
|   |-- date_processing.py        # date normalization
|   |-- deduplication.py           # exact + near-duplicate detection
|   |-- sentiment.py               # sentiment analysis
|   |-- topics.py                  # BERTopic topic modeling
|   |-- keywords.py                # KeyBERT keyword extraction
|   |-- ner.py                     # named entity recognition
|   |-- temporal_analysis.py        # weekly aggregation, trend features
|   |-- crisis_analysis.py          # crisis windows, anomaly detection
|   |-- modeling.py                 # ML classification, lead-time analysis
|   |-- evaluate_sentiment.py       # sentiment model evaluation vs. hand-labeled sample
|   `-- pipeline.py                 # orchestration: process_existing_articles / process_new_article
|
|-- notebooks/
|   |-- 00_data_scrape.ipynb
|   |-- 01_data_check.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_nlp.ipynb
|   |-- 04_topics.ipynb
|   |-- 05_temporal_analysis.ipynb
|   `-- 06_crisis_analysis.ipynb
|   |-- 
|
|-- tests/                       # one test file per src/ module (mocked, no live DB needed)
|-- models/                      # saved BERTopic model, topic_info.csv
`-- data/processed/               # exported CSVs (sentiment eval sample, etc.)
```

---

## Setup

**1. Clone and create a virtual environment**
```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate      # macOS/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**3. Configure your database**
```bash
copy .env.example .env           # Windows
# cp .env.example .env           # macOS/Linux
```
Edit `.env` with your MySQL host, port, user, password, and database name. The database must already contain a `scraped_news` table (`id`, `title`, `content`, `date`, `link`) with your raw articles.

**4. Initialize the schema**
```bash
python -c "import sys; sys.path.insert(0,'src'); import database as db; db.init_schema()"
```
Then run `sql/seed_crisis_events.sql` (or `cleanup_crisis_events.sql`) once to populate your crisis events.

**5. Run the pipeline**
```python
import sys; sys.path.insert(0, 'src')
import pipeline, sentiment, topics, keywords, ner, temporal_analysis, crisis_analysis, modeling

pipeline.process_existing_articles()          # Phase 1: validate, clean, date, dedup
sentiment.process_all_sentiment()               # Step 7
model, topic_info = topics.process_all_topics(nr_topics=25)   # Step 8
topics.apply_topic_names({...})                  # manual naming step, see topics.py
keywords.process_all_keywords()                  # Step 9
ner.process_all_ner()                             # Step 10
temporal_analysis.process_temporal_analysis()      # Steps 11-13
crisis_analysis.process_crisis_analysis()          # Steps 14-16, 18
results, lead_time_df = modeling.run_full_evaluation()  # Steps 17, 19-20
```

Each notebook in `notebooks/` walks through the equivalent analysis with visualizations.

**6. Run tests**
```bash
for f in tests/test_*.py; do python "$f"; done
```
All tests use mocked databases/models -- no live MySQL connection or model downloads required to verify the logic.

---

## Key Findings

- **Sentiment does not equal crisis relevance.** A general-purpose sentiment model measures linguistic tone, not real-world event severity -- evaluated at only 45% agreement with substantive ground-truth labels.
- **Topic and keyword frequency are the real signal.** Tracked crisis-relevant topics rose 12-15x above baseline at the onset of both studied crises.
- **Anomaly detection works.** A simple rolling z-score correctly flagged both crisis onsets, peaking at z = 13.92.
- **~18-19 day lead time**, consistent across two structurally different crisis types (sudden natural disaster vs. gradual geopolitical build-up).
- **Small-sample caveat:** only 2 real crisis events were available for ML validation -- results should be read as exploratory/indicative, not production-grade.

Full methodology, all figures, and complete discussion are in the accompanying project report.

---

## Notes on Design Decisions

- **No foreign keys.** MySQL 8 exhibited a reproducible data-dictionary caching bug when adding FKs immediately after table creation in the same session. Referential integrity is enforced at the application level instead.
- **A year-long data gap** (April 2024 - July 2025) in the underlying dataset eliminated an originally planned third crisis event (the 2024 presidential election); the study proceeded with two events instead.
- **Lightweight models throughout** (`all-MiniLM-L6-v2`, `en_core_web_sm`, base-sized RoBERTa) were chosen deliberately for CPU-only, GPU-less hardware -- documented explicitly in the report rather than left unstated.

---

## Tech Stack

Python - MySQL - SQLAlchemy - pandas - scikit-learn - transformers - BERTopic - sentence-transformers - KeyBERT - spaCy

---

## License / Academic Use

This is a university capstone research project. Code is shared for academic transparency and reproducibility.
