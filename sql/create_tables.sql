
CREATE TABLE IF NOT EXISTS processed_news (
    article_id            BIGINT PRIMARY KEY,
    title_original         TEXT,
    title_clean            TEXT,
    content_original       LONGTEXT,
    content_clean          LONGTEXT,
    date_raw               VARCHAR(255),
    published_at            DATETIME NULL,
    publication_date        DATE NULL,
    publication_year        SMALLINT NULL,
    publication_month       TINYINT NULL,
    publication_week        VARCHAR(8) NULL,
    publication_day         TINYINT NULL,
    article_length_words    INT NULL,
    article_length_chars    INT NULL,
    sentence_count          INT NULL,
    duplicate_status        ENUM('UNIQUE','EXACT_DUPLICATE','NEAR_DUPLICATE') DEFAULT 'UNIQUE',
    duplicate_of            BIGINT NULL,
    content_hash            VARCHAR(64) NULL,
    processing_status       ENUM('PENDING','COMPLETED','FAILED') DEFAULT 'PENDING',
    processing_timestamp    DATETIME NULL,
    pipeline_version        VARCHAR(20) NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS article_sentiment (
    article_id        BIGINT PRIMARY KEY,
    sentiment          ENUM('positive','neutral','negative') NULL,
    sentiment_score    FLOAT NULL,
    sentiment_model    VARCHAR(100) NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS article_topics (
    article_id          BIGINT NOT NULL,
    topic_id             INT NOT NULL,
    topic_name           VARCHAR(100) NULL,
    topic_probability    FLOAT NULL,
    PRIMARY KEY (article_id, topic_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS article_keywords (
    article_id    BIGINT NOT NULL,
    keyword        VARCHAR(255) NOT NULL,
    score          FLOAT NULL,
    `rank`         TINYINT NOT NULL,
    PRIMARY KEY (article_id, `rank`)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS article_entities (
    article_id     BIGINT NOT NULL,
    entity          VARCHAR(255) NOT NULL,
    entity_type     VARCHAR(50) NOT NULL,
    confidence      FLOAT NULL,
    PRIMARY KEY (article_id, entity(191), entity_type)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS crisis_events (
    event_id       INT AUTO_INCREMENT PRIMARY KEY,
    event_name      VARCHAR(255) NOT NULL,
    crisis_type      VARCHAR(100) NULL,
    start_date       DATE NOT NULL,
    end_date         DATE NULL,
    description      TEXT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS temporal_features (
    week                       VARCHAR(8) PRIMARY KEY,
    article_count               INT NULL,
    average_sentiment           FLOAT NULL,
    negative_sentiment_ratio    FLOAT NULL,
    positive_sentiment_ratio    FLOAT NULL,
    article_count_change        FLOAT NULL,
    sentiment_change            FLOAT NULL,
    topic_growth_rate           FLOAT NULL,
    keyword_growth_rate         FLOAT NULL,
    entity_growth_rate          FLOAT NULL,
    ma_4week                    FLOAT NULL,
    ma_8week                    FLOAT NULL,
    anomaly_score                FLOAT NULL,
    crisis_event                 INT NULL,
    days_to_crisis                INT NULL,
    label_pre_crisis              TINYINT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS processing_log (
    log_id       BIGINT AUTO_INCREMENT PRIMARY KEY,
    article_id    BIGINT NOT NULL,
    stage          VARCHAR(50) NOT NULL,
    status         ENUM('SUCCESS','FAILED','SKIPPED') NOT NULL,
    message        TEXT NULL,
    timestamp      DATETIME NOT NULL
) ENGINE=InnoDB;