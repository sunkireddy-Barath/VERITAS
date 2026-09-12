-- VERITAS schema. The bitemporal store maps onto PostgreSQL range types
-- directly, which is the main reason Postgres is the right database here.
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector; drop if unavailable

CREATE TABLE IF NOT EXISTS fact_version (
    version_id     TEXT PRIMARY KEY,
    entity         TEXT        NOT NULL,
    attribute      TEXT        NOT NULL,
    value          TEXT        NOT NULL,
    -- VALID TIME: when the fact was true in the world.
    valid_range    TSTZRANGE   NOT NULL,
    -- TRANSACTION TIME: when we learned it. Half-open; NULL upper = current.
    recorded_at    TIMESTAMPTZ NOT NULL,
    superseded_at  TIMESTAMPTZ,
    source_id      TEXT        NOT NULL,
    confidence     REAL        NOT NULL DEFAULT 0.5,
    change_kind    TEXT        NOT NULL,
    previous_value TEXT,
    reason         TEXT,
    evidence_ids   JSONB       NOT NULL DEFAULT '[]'::jsonb
);

-- The core query is "the version valid at t": a range containment test, which
-- GiST answers with an index scan rather than a table scan.
CREATE INDEX IF NOT EXISTS fact_valid_gist
    ON fact_version USING gist (entity, attribute, valid_range);
CREATE INDEX IF NOT EXISTS fact_entity_attr
    ON fact_version (entity, attribute, recorded_at DESC);
CREATE INDEX IF NOT EXISTS fact_current
    ON fact_version (entity, attribute) WHERE superseded_at IS NULL;

-- One source may not assert two overlapping values for the same fact. This is
-- a genuine invariant of the data model, so it belongs in the database rather
-- than in application code where a second writer could bypass it.
ALTER TABLE fact_version DROP CONSTRAINT IF EXISTS fact_no_overlap;
ALTER TABLE fact_version ADD CONSTRAINT fact_no_overlap
    EXCLUDE USING gist (
        entity WITH =, attribute WITH =, source_id WITH =, valid_range WITH &&
    ) WHERE (superseded_at IS NULL);

CREATE TABLE IF NOT EXISTS document (
    doc_id      TEXT PRIMARY KEY,
    source_id   TEXT        NOT NULL,
    tier        SMALLINT    NOT NULL DEFAULT 3,
    published   TIMESTAMPTZ,
    url         TEXT,
    entity      TEXT,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    digest      TEXT,          -- BLAKE2b, for exact-change detection
    simhash     BIGINT         -- near-duplicate detection
);
CREATE INDEX IF NOT EXISTS document_source ON document (source_id, published DESC);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES document(doc_id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    heading     TEXT,
    start_char  INTEGER,
    end_char    INTEGER,
    valid_range TSTZRANGE,
    embedding   vector(384)    -- must match ModelConfig.d_model
);
CREATE INDEX IF NOT EXISTS chunk_doc ON chunk (doc_id);
-- IVFFlat over cosine distance. Build AFTER bulk load: the index needs data to
-- pick sensible centroids.
-- CREATE INDEX chunk_embedding_ivf ON chunk
--   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS evidence_edge (
    id        BIGSERIAL PRIMARY KEY,
    src       TEXT NOT NULL,
    dst       TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight    REAL DEFAULT 1.0,
    attrs     JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS edge_src ON evidence_edge (src, edge_type);
CREATE INDEX IF NOT EXISTS edge_dst ON evidence_edge (dst, edge_type);

-- Example: the as-of query the whole system is built around.
--   SELECT value FROM fact_version
--    WHERE entity = $1 AND attribute = $2
--      AND valid_range @> $3::timestamptz     -- valid time
--      AND recorded_at <= $4                  -- transaction time
--      AND (superseded_at IS NULL OR superseded_at > $4)
--    ORDER BY recorded_at DESC LIMIT 1;
