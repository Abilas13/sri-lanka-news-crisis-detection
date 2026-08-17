"""
config.py
Central place for DB connection settings and pipeline constants.
Reads secrets from environment variables (via a .env file) so credentials
never get hardcoded or committed to git.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file in the project root

# --- MySQL connection ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "sri_lanka_news")

# SQLAlchemy connection string using the PyMySQL driver
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Pipeline constants ---
PIPELINE_VERSION = "0.1.0"

MIN_CONTENT_LENGTH_CHARS = 50   # below this, an article is flagged as abnormally short

# Existing raw table — never written to, only read from
RAW_TABLE = "scraped_news"