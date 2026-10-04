-- Provenance and renumbering: this table was introduced in the fork as
-- migration V68. Upstream 2.5.2 added its own V68–V71 (project_grants and
-- successors), so the fork's migration was renumbered to V72, after them.
-- The DDL is idempotent (IF NOT EXISTS) because stores that applied the old
-- V68 already have the table; the deploy removes the old
-- `68 consolidation_chunk_progress` row from `refinery_schema_history`
-- before the first boot of the new binary.
--
-- Durable per-block progress for opt-in map-reduce consolidation so a
-- crashed or re-run consolidation reuses LLM extractions that already
-- succeeded instead of re-paying for every call.
--
-- Keyed by the FULL typed scope plus a content-derived fingerprint
-- (prompt/schema version, model, and the block's observation ids +
-- sanitized text), never by a bare session id: a foreign
-- workspace/project/session row must be invisible to every other scope.
-- `created_at` is bookkeeping only — reuse decisions compare fingerprints,
-- never timestamps, so a rolled-back clock cannot change the outcome.
-- No foreign key: the fingerprint is content-derived, not a row. A
-- successful publish prunes the session's rows; a run that crashes after
-- publishing but before pruning is reconciled on the next run by the
-- anchor page's map-reduce publication marker (`consolidation_marker`:
-- prompt versions, model, mode, the RESOLVED consolidation instructions,
-- and a digest of the sanitized observations — every field length-prefixed
-- so distinct inputs can never alias), NOT by the bare session-origin
-- stamp — the heuristic SessionEnd synthesizer writes the anchor with only
-- the origin stamp (no marker), so that page is not a map-reduce publication
-- and the pipeline runs. A changed instructions page is a different
-- operation, so it re-runs instead of reconciling; the batch drops any
-- update to the reserved `_prompts/consolidation.md` page (input, not
-- output). The marker is the only proof of publication.
CREATE TABLE IF NOT EXISTS consolidation_chunk_progress (
    workspace_id      TEXT    NOT NULL,
    project_id        TEXT    NOT NULL,
    session_id        TEXT    NOT NULL,
    chunk_fingerprint TEXT    NOT NULL,
    extraction_json   TEXT    NOT NULL,
    created_at        INTEGER NOT NULL,
    PRIMARY KEY (workspace_id, project_id, session_id, chunk_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_consolidation_chunk_progress_at
    ON consolidation_chunk_progress(created_at);
