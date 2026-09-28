
CREATE INDEX idx_published_at ON processed_news (published_at);
CREATE INDEX idx_content_hash ON processed_news (content_hash);
CREATE INDEX idx_processing_status ON processed_news (processing_status);

CREATE INDEX idx_sentiment_article ON article_sentiment (article_id);
CREATE INDEX idx_topics_article ON article_topics (article_id);
CREATE INDEX idx_keywords_article ON article_keywords (article_id);
CREATE INDEX idx_entities_article ON article_entities (article_id);

CREATE INDEX idx_temporal_crisis ON temporal_features (crisis_event);

CREATE INDEX idx_log_article ON processing_log (article_id);
CREATE INDEX idx_log_stage_status ON processing_log (stage, status);