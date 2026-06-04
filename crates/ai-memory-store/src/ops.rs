//! Mutating SQL operations executed on the writer thread.
//!
//! Each operation is one transaction. Calling them from anywhere other than
//! the writer thread would violate the single-writer invariant (see
//! [`crate::writer`]).

use std::collections::BTreeSet;

use ai_memory_core::{
    AgentKind, HandoffId, LinkTarget, NewHandoff, NewObservation, NewPage, NewSession,
    ObservationId, ObservationKind, PageId, PagePath, ProjectId, SessionId, WorkspaceId,
};

/// Summary returned by [`reorg_sessions`] and exposed via
/// [`crate::writer::WriterHandle::reorg_sessions`].
#[derive(Debug, Default, Clone)]
pub struct ReorgSummary {
    /// Sessions whose `project_id` was changed.
    pub sessions_moved: usize,
    /// Observations updated to match their session's new project.
    pub observations_updated: usize,
    /// `is_latest=1` pages marked `is_latest=0` (mash-up graveyard).
    pub pages_graveyarded: usize,
}

/// Summary returned by [`purge_project`] and exposed via
/// [`crate::writer::WriterHandle::purge_project`].
#[derive(Debug, Default, Clone)]
pub struct PurgeSummary {
    /// Human-readable `workspace/project` label. Set by the caller (writer
    /// only knows IDs); filled in by [`purge_project`] from its parameters.
    pub label: String,
    /// Distinct page paths that were present before the delete (all versions,
    /// not just `is_latest=1`). The admin handler uses this list to remove
    /// the corresponding files from the wiki directory.
    pub page_paths: Vec<String>,
    /// Number of `pages` rows deleted (all versions, not just latest).
    pub pages_deleted: u64,
    /// Number of `sessions` rows deleted.
    pub sessions_deleted: u64,
    /// Number of `observations` rows deleted.
    pub observations_deleted: u64,
    /// Number of `handoffs` rows deleted.
    pub handoffs_deleted: u64,
    /// Number of `page_embeddings` rows deleted (cascades through pages).
    pub embeddings_deleted: u64,
}
use jiff::Timestamp;
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::error::{StoreError, StoreResult};

/// One embedding upsert requested by a backfill or embed command.
#[derive(Debug)]
pub struct EmbeddingWrite {
    /// Page receiving the embedding.
    pub page_id: PageId,
    /// Packed little-endian `f32` vector bytes.
    pub vector_bytes: Vec<u8>,
    /// Embedding provider name.
    pub provider: String,
    /// Embedding model name.
    pub model: String,
    /// Vector dimension.
    pub dim: u32,
}

/// Upsert a page by path, superseding any existing latest version when the
/// content (sha256 of body) has changed.
///
/// Returns the id of the page row that should now be considered current.
pub fn upsert_page(conn: &mut Connection, page: &NewPage) -> StoreResult<PageId> {
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    let result_id = upsert_page_in_tx(&tx, page, now)?;
    tx.commit()?;
    Ok(result_id)
}

/// Resolve a workspace by name, creating it if missing. Atomic.
pub fn get_or_create_workspace(
    conn: &mut Connection,
    name: &str,
) -> StoreResult<ai_memory_core::WorkspaceId> {
    let tx = conn.transaction()?;
    let existing: Option<Vec<u8>> = tx
        .query_row(
            "SELECT id FROM workspaces WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )
        .optional()?;
    let id = if let Some(bytes) = existing {
        ai_memory_core::WorkspaceId::from_slice(&bytes)?
    } else {
        let id = ai_memory_core::WorkspaceId::new();
        tx.execute(
            "INSERT INTO workspaces (id, name, created_at) VALUES (?1, ?2, ?3)",
            params![id.as_bytes(), name, Timestamp::now().as_microsecond()],
        )?;
        id
    };
    tx.commit()?;
    Ok(id)
}

/// Resolve a project by `(workspace_id, name)`, creating it if missing.
/// Atomic.
pub fn get_or_create_project(
    conn: &mut Connection,
    workspace_id: &ai_memory_core::WorkspaceId,
    name: &str,
    repo_path: Option<&str>,
) -> StoreResult<ai_memory_core::ProjectId> {
    let tx = conn.transaction()?;
    let existing: Option<Vec<u8>> = tx
        .query_row(
            "SELECT id FROM projects WHERE workspace_id = ?1 AND name = ?2",
            params![workspace_id.as_bytes(), name],
            |row| row.get(0),
        )
        .optional()?;
    let id = if let Some(bytes) = existing {
        ai_memory_core::ProjectId::from_slice(&bytes)?
    } else {
        let id = ai_memory_core::ProjectId::new();
        tx.execute(
            "INSERT INTO projects (id, workspace_id, name, repo_path, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id.as_bytes(),
                workspace_id.as_bytes(),
                name,
                repo_path,
                Timestamp::now().as_microsecond()
            ],
        )?;
        id
    };
    tx.commit()?;
    Ok(id)
}

/// Assert that `project_id` currently belongs to `workspace_id`.
///
/// Wiki writes call this before touching the filesystem so a stale hook/cache
/// carrying the old workspace for a moved project fails before it can create an
/// orphan file. The pairing INSERT triggers are still the final SQL backstop.
pub fn ensure_project_workspace(
    conn: &Connection,
    workspace_id: &WorkspaceId,
    project_id: &ProjectId,
) -> StoreResult<()> {
    let found = conn
        .query_row(
            "SELECT 1 FROM projects WHERE id = ?1 AND workspace_id = ?2",
            params![project_id.as_bytes(), workspace_id.as_bytes()],
            |_| Ok(()),
        )
        .optional()?;
    if found.is_some() {
        Ok(())
    } else {
        Err(StoreError::NotFound(format!(
            "project {project_id} does not belong to workspace {workspace_id}"
        )))
    }
}

/// Upsert a batch of pages inside one transaction. Either *all* pages
/// land (each becoming the new `is_latest=true` version) or none do.
///
/// This is the M7b atomic-fan-out path: the consolidator can hand a
/// list of {sessions, concepts, decisions} pages and trust that
/// either the whole batch supersedes or the wiki is unchanged.
pub fn upsert_pages_batch(conn: &mut Connection, pages: &[NewPage]) -> StoreResult<Vec<PageId>> {
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    let mut out = Vec::with_capacity(pages.len());
    for page in pages {
        let id = upsert_page_in_tx(&tx, page, now)?;
        out.push(id);
    }
    tx.commit()?;
    Ok(out)
}

struct ExistingPageVersion {
    id: Vec<u8>,
    body_sha256: Vec<u8>,
    frontmatter_json: String,
    title: String,
    tier: String,
    pinned: i64,
}

/// Normalise a page path into FTS-friendly search text, indexing BOTH forms
/// so either a whole-slug or a single-word query hits:
/// - segments: `/` and `.` → space, KEEPING `-`/`_` (FTS token chars) so the
///   full hyphenated slug stays one token (`foo-bar` matches a `"foo-bar"`
///   query);
/// - words: also split `-`/`_` so each word is its own token (`bar` matches).
///
/// `notes/foo-bar.md` → `notes foo-bar md notes foo bar md`.
///
/// MUST stay byte-identical to the backfill expression in migration V17 so
/// the `rebuild` and live-write paths index the same text (matching bm25
/// term frequencies, not just the same match set).
pub(crate) fn path_search_text(path: &str) -> String {
    let segments = path.replace(['/', '.'], " ");
    let words = segments.replace(['-', '_'], " ");
    format!("{segments} {words}")
}

fn upsert_page_in_tx(
    tx: &rusqlite::Transaction<'_>,
    page: &NewPage,
    now: i64,
) -> StoreResult<PageId> {
    let path_search = path_search_text(page.path.as_str());
    let body_sha256: [u8; 32] = {
        let mut hasher = Sha256::new();
        hasher.update(page.body.as_bytes());
        hasher.finalize().into()
    };
    let frontmatter_str = serde_json::to_string(&page.frontmatter_json)?;
    let tier_str = page.tier.as_str();

    let existing: Option<ExistingPageVersion> = tx
        .query_row(
            "SELECT id, body_sha256, frontmatter_json, title, tier, pinned FROM pages \
             WHERE workspace_id = ?1 AND project_id = ?2 AND path = ?3 AND is_latest = 1",
            params![
                page.workspace_id.as_bytes(),
                page.project_id.as_bytes(),
                page.path.as_str(),
            ],
            |row| {
                Ok(ExistingPageVersion {
                    id: row.get(0)?,
                    body_sha256: row.get(1)?,
                    frontmatter_json: row.get(2)?,
                    title: row.get(3)?,
                    tier: row.get(4)?,
                    pinned: row.get(5)?,
                })
            },
        )
        .optional()?;

    if let Some(existing) = existing {
        if existing.body_sha256 == body_sha256
            && existing.frontmatter_json == frontmatter_str
            && existing.title == page.title
            && existing.tier == tier_str
            && existing.pinned == i64::from(page.pinned)
        {
            return PageId::from_slice(&existing.id).map_err(StoreError::from);
        }
        let new_id = PageId::new();
        tx.execute(
            "UPDATE pages SET is_latest = 0 WHERE id = ?1",
            params![&existing.id],
        )?;
        tx.execute(
            "INSERT INTO pages \
             (id, workspace_id, project_id, path, path_search, title, tier, body, body_sha256, \
              frontmatter_json, is_latest, supersedes, pinned, author_id, \
              created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 1, ?11, ?12, ?13, ?14, ?14)",
            params![
                new_id.as_bytes(),
                page.workspace_id.as_bytes(),
                page.project_id.as_bytes(),
                page.path.as_str(),
                path_search,
                page.title,
                tier_str,
                page.body,
                body_sha256.as_slice(),
                frontmatter_str,
                &existing.id,
                i64::from(page.pinned),
                page.author_id.map(|id| id.as_bytes().to_vec()),
                now,
            ],
        )?;
        replace_links_in_tx(tx, &new_id, page)?;
        refresh_incoming_links_for_path(tx, page, &new_id)?;
        audit(
            tx,
            "supersede_page",
            Some(page.workspace_id.as_bytes()),
            Some(page.project_id.as_bytes()),
            Some(new_id.as_bytes()),
            page.author_id
                .as_ref()
                .map(ai_memory_core::UserId::as_bytes),
            now,
        )?;
        return Ok(new_id);
    }
    let new_id = PageId::new();
    tx.execute(
        "INSERT INTO pages \
         (id, workspace_id, project_id, path, path_search, title, tier, body, body_sha256, \
          frontmatter_json, is_latest, pinned, author_id, created_at, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 1, ?11, ?12, ?13, ?13)",
        params![
            new_id.as_bytes(),
            page.workspace_id.as_bytes(),
            page.project_id.as_bytes(),
            page.path.as_str(),
            path_search,
            page.title,
            tier_str,
            page.body,
            body_sha256.as_slice(),
            frontmatter_str,
            i64::from(page.pinned),
            page.author_id.map(|id| id.as_bytes().to_vec()),
            now,
        ],
    )?;
    replace_links_in_tx(tx, &new_id, page)?;
    refresh_incoming_links_for_path(tx, page, &new_id)?;
    audit(
        tx,
        "create_page",
        Some(page.workspace_id.as_bytes()),
        Some(page.project_id.as_bytes()),
        Some(new_id.as_bytes()),
        page.author_id
            .as_ref()
            .map(ai_memory_core::UserId::as_bytes),
        now,
    )?;
    Ok(new_id)
}

fn replace_links_in_tx(
    tx: &rusqlite::Transaction<'_>,
    from_page_id: &PageId,
    page: &NewPage,
) -> StoreResult<()> {
    tx.execute(
        "DELETE FROM links WHERE from_page_id = ?1",
        params![from_page_id.as_bytes()],
    )?;

    let mut seen = BTreeSet::new();
    for link in &page.links {
        let key = (
            link.workspace.clone(),
            link.project.clone(),
            link.path.as_str().to_string(),
        );
        if !seen.insert(key) {
            continue;
        }
        let to_page_id = latest_page_id_for_link(tx, page, link)?;
        let to_page_blob = to_page_id.as_ref().map(|id| &id.as_bytes()[..]);
        tx.execute(
            "INSERT INTO links \
                 (from_page_id, to_page_id, to_workspace, to_project, to_path, link_type) \
             VALUES (?1, ?2, ?3, ?4, ?5, 'references')",
            params![
                from_page_id.as_bytes(),
                to_page_blob,
                link.workspace,
                link.project,
                link.path.as_str(),
            ],
        )?;
    }
    Ok(())
}

/// Resolve a link target to the latest page id it points at, or `None` if the
/// target workspace / project / page does not exist yet (an unresolved forward
/// link). A bare link resolves within the source page's own project; a
/// `[[project:path]]` / `[[workspace/project:path]]` link resolves against the
/// named project (same workspace when only the project is given).
fn latest_page_id_for_link(
    tx: &rusqlite::Transaction<'_>,
    page: &NewPage,
    link: &LinkTarget,
) -> StoreResult<Option<PageId>> {
    let (workspace_blob, project_blob): (Vec<u8>, Vec<u8>) = match &link.project {
        None => (
            page.workspace_id.as_bytes().to_vec(),
            page.project_id.as_bytes().to_vec(),
        ),
        Some(project_name) => {
            let workspace_blob: Vec<u8> = match &link.workspace {
                None => page.workspace_id.as_bytes().to_vec(),
                Some(workspace_name) => {
                    let found: Option<Vec<u8>> = tx
                        .query_row(
                            "SELECT id FROM workspaces WHERE name = ?1",
                            params![workspace_name],
                            |row| row.get(0),
                        )
                        .optional()?;
                    match found {
                        Some(id) => id,
                        None => return Ok(None),
                    }
                }
            };
            let project_blob: Option<Vec<u8>> = tx
                .query_row(
                    "SELECT id FROM projects WHERE workspace_id = ?1 AND name = ?2",
                    params![workspace_blob, project_name],
                    |row| row.get(0),
                )
                .optional()?;
            match project_blob {
                Some(id) => (workspace_blob, id),
                None => return Ok(None),
            }
        }
    };

    let bytes: Option<Vec<u8>> = tx
        .query_row(
            "SELECT id FROM pages \
             WHERE workspace_id = ?1 AND project_id = ?2 AND path = ?3 AND is_latest = 1",
            params![workspace_blob, project_blob, link.path.as_str()],
            |row| row.get(0),
        )
        .optional()?;
    bytes
        .map(|bytes| PageId::from_slice(&bytes).map_err(StoreError::from))
        .transpose()
}

fn refresh_incoming_links_for_path(
    tx: &rusqlite::Transaction<'_>,
    page: &NewPage,
    latest_page_id: &PageId,
) -> StoreResult<()> {
    // (1) Bare (same-project) links: from_page lives in this page's project and
    // the target carries no scope. Repoints all matches (not only unresolved):
    // a new page version changes the latest id, so resolved links must follow.
    tx.execute(
        "UPDATE links \
         SET to_page_id = ?1 \
         WHERE to_project IS NULL AND to_path = ?2 \
           AND EXISTS ( \
               SELECT 1 FROM pages from_page \
               WHERE from_page.id = links.from_page_id \
                 AND from_page.workspace_id = ?3 \
                 AND from_page.project_id = ?4 \
           )",
        params![
            latest_page_id.as_bytes(),
            page.path.as_str(),
            page.workspace_id.as_bytes(),
            page.project_id.as_bytes(),
        ],
    )?;

    // (2) Cross-project links naming this page's project by name. `to_workspace`
    // may be explicit (cross-workspace) or NULL (same workspace as the source).
    let project_name: Option<String> = tx
        .query_row(
            "SELECT name FROM projects WHERE id = ?1",
            params![page.project_id.as_bytes()],
            |row| row.get(0),
        )
        .optional()?;
    let workspace_name: Option<String> = tx
        .query_row(
            "SELECT name FROM workspaces WHERE id = ?1",
            params![page.workspace_id.as_bytes()],
            |row| row.get(0),
        )
        .optional()?;
    if let (Some(project_name), Some(workspace_name)) = (project_name, workspace_name) {
        tx.execute(
            "UPDATE links \
             SET to_page_id = ?1 \
             WHERE to_project = ?2 AND to_path = ?3 \
               AND ( \
                   to_workspace = ?4 \
                   OR ( \
                       to_workspace IS NULL \
                       AND EXISTS ( \
                           SELECT 1 FROM pages from_page \
                           WHERE from_page.id = links.from_page_id \
                             AND from_page.workspace_id = ?5 \
                       ) \
                   ) \
               )",
            params![
                latest_page_id.as_bytes(),
                project_name,
                page.path.as_str(),
                workspace_name,
                page.workspace_id.as_bytes(),
            ],
        )?;
    }
    Ok(())
}

/// Begin (or re-affirm) a session row keyed on the caller-supplied id.
/// Idempotent: a second call with the same id leaves the row untouched.
pub fn begin_session(conn: &mut Connection, session: &NewSession) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let agent = session.agent_kind.as_str();
    let cwd: Option<String> = session
        .cwd
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    conn.execute(
        "INSERT INTO sessions (id, workspace_id, project_id, agent_kind, cwd, started_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(id) DO NOTHING",
        params![
            session.id.as_bytes(),
            session.workspace_id.as_bytes(),
            session.project_id.as_bytes(),
            agent,
            cwd,
            now,
        ],
    )?;
    Ok(())
}

/// Stamp a session as ended, optionally linking the synthesised summary
/// page.
pub fn end_session(
    conn: &mut Connection,
    session_id: &SessionId,
    summary_page_id: Option<&PageId>,
) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let page_blob: Option<&[u8]> = summary_page_id.map(|p| &p.as_bytes()[..]);
    conn.execute(
        "UPDATE sessions SET ended_at = ?1, summary_page_id = ?2 WHERE id = ?3",
        params![now, page_blob, session_id.as_bytes()],
    )?;
    Ok(())
}

/// Append a single observation. Caller is expected to have already
/// inserted the parent session via [`begin_session`].
pub fn insert_observation(
    conn: &mut Connection,
    obs: &NewObservation,
) -> StoreResult<ObservationId> {
    let id = ObservationId::new();
    let now = Timestamp::now().as_microsecond();
    let kind = observation_kind_as_str(obs.kind);
    let importance: i64 = i64::from(obs.importance.clamp(1, 10));
    let (extension, source_event) = match (&obs.extension, &obs.source_event) {
        (Some(extension), Some(source_event)) => {
            (Some(extension.as_str()), Some(source_event.as_str()))
        }
        _ => (None, None),
    };
    conn.execute(
        "INSERT INTO observations \
         (id, session_id, workspace_id, project_id, kind, extension, source_event, title, body, \
          importance, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            id.as_bytes(),
            obs.session_id.as_bytes(),
            obs.workspace_id.as_bytes(),
            obs.project_id.as_bytes(),
            kind,
            extension,
            source_event,
            obs.title,
            obs.body,
            importance,
            now,
        ],
    )?;
    Ok(id)
}

/// Store / replace one page's embedding. Bytes are the host-endian
/// `f32` packing of the unit-normalised vector. Provider/model/dim
/// are denormalised onto the row so a single SELECT can detect
/// heterogeneity (refuse-on-mismatch path).
pub fn store_embedding(
    conn: &mut Connection,
    page_id: &PageId,
    vector_bytes: &[u8],
    provider: &str,
    model: &str,
    dim: u32,
) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    conn.execute(
        "INSERT INTO page_embeddings (page_id, vector, provider, model, dim, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(page_id) DO UPDATE SET \
             vector = excluded.vector, \
             provider = excluded.provider, \
             model = excluded.model, \
             dim = excluded.dim, \
             created_at = excluded.created_at",
        params![page_id.as_bytes(), vector_bytes, provider, model, dim, now,],
    )?;
    Ok(())
}

/// Store / replace a batch of page embeddings in one transaction.
pub fn store_embeddings(conn: &mut Connection, embeddings: &[EmbeddingWrite]) -> StoreResult<()> {
    if embeddings.is_empty() {
        return Ok(());
    }
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO page_embeddings (page_id, vector, provider, model, dim, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(page_id) DO UPDATE SET \
                 vector = excluded.vector, \
                 provider = excluded.provider, \
                 model = excluded.model, \
                 dim = excluded.dim, \
                 created_at = excluded.created_at",
        )?;
        for embedding in embeddings {
            stmt.execute(params![
                embedding.page_id.as_bytes(),
                embedding.vector_bytes.as_slice(),
                embedding.provider.as_str(),
                embedding.model.as_str(),
                embedding.dim,
                now,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Bump `access_count` + `last_accessed_at` for the pages whose ids
/// appear in `page_ids`. Idempotent for unknown ids (no-op).
/// Used by the read path to feed the M8 reinforcement term.
pub fn bump_access_for_pages(conn: &mut Connection, page_ids: &[PageId]) -> StoreResult<()> {
    if page_ids.is_empty() {
        return Ok(());
    }
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "UPDATE pages \
             SET access_count = access_count + 1, last_accessed_at = ?1 \
             WHERE id = ?2 AND is_latest = 1",
        )?;
        for id in page_ids {
            stmt.execute(params![now, id.as_bytes()])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Mark a set of `is_latest=1` pages as soft-deleted by the forget
/// sweep. Distinguished from M7 supersession by `supersedes IS NULL`.
pub fn soft_delete_for_decay(conn: &mut Connection, page_ids: &[PageId]) -> StoreResult<usize> {
    if page_ids.is_empty() {
        return Ok(0);
    }
    let now = Timestamp::now().as_microsecond();
    let mut affected = 0usize;
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "UPDATE pages \
             SET is_latest = 0, superseded_at = ?1 \
             WHERE id = ?2 AND is_latest = 1",
        )?;
        for id in page_ids {
            affected += stmt.execute(params![now, id.as_bytes()])?;
        }
    }
    audit(
        &tx,
        "soft_delete_for_decay",
        None,
        None,
        None,
        // Decay sweep is a system op (scheduled / admin-triggered) — no
        // user-attributable actor at the row level.
        None,
        Timestamp::now().as_microsecond(),
    )?;
    tx.commit()?;
    Ok(affected)
}

/// Delete every version of a page (by path) from the index. Used when the
/// wiki file is removed (`Wiki::delete_page`): the watcher does not handle
/// file deletions, so the derived rows must be dropped explicitly or the
/// page keeps surfacing in search/recent with stale content. FK cascades
/// drop outgoing links + embeddings; the `pages_fts_ad` trigger keeps FTS in
/// sync; incoming links are set to NULL (unresolved). Idempotent.
pub fn delete_page(
    conn: &Connection,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    path: &PagePath,
) -> StoreResult<()> {
    conn.execute(
        "DELETE FROM pages WHERE workspace_id = ?1 AND project_id = ?2 AND path = ?3",
        params![
            workspace_id.as_bytes(),
            project_id.as_bytes(),
            path.as_str()
        ],
    )?;
    Ok(())
}

/// Hard-delete rows that were soft-deleted by an earlier sweep at
/// least `hard_delete_after_days` ago AND received zero subsequent
/// accesses. Safe: M7 supersedes-chain pages have a non-null
/// `supersedes` so they never match.
pub fn hard_delete_decayed_pages(
    conn: &mut Connection,
    hard_delete_after_days: i64,
) -> StoreResult<usize> {
    let cutoff = Timestamp::now().as_microsecond() - hard_delete_after_days * 86_400_000_000;
    let n = conn.execute(
        "DELETE FROM pages \
         WHERE is_latest = 0 \
           AND supersedes IS NULL \
           AND superseded_at IS NOT NULL \
           AND superseded_at < ?1 \
           AND access_count = 0",
        params![cutoff],
    )?;
    Ok(n)
}

/// Insert a new handoff in state=open.
pub fn insert_handoff(conn: &mut Connection, h: &NewHandoff) -> StoreResult<HandoffId> {
    let id = HandoffId::new();
    let now = Timestamp::now().as_microsecond();
    let open_q = serde_json::to_string(&h.open_questions)?;
    let next_s = serde_json::to_string(&h.next_steps)?;
    let files = serde_json::to_string(&h.files_touched)?;
    let from_session: Option<&[u8]> = h.from_session_id.as_ref().map(|s| &s.as_bytes()[..]);
    let cwd: Option<String> = h.cwd.as_ref().map(|p| p.to_string_lossy().into_owned());
    let from_agent = h.from_agent.as_str();
    let to_agent = h.to_agent.map(AgentKind::as_str);
    conn.execute(
        "INSERT INTO handoffs \
         (id, workspace_id, project_id, from_session_id, from_agent, to_agent, cwd, summary, \
          open_questions, next_steps, files_touched, state, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'open', ?12)",
        params![
            id.as_bytes(),
            h.workspace_id.as_bytes(),
            h.project_id.as_bytes(),
            from_session,
            from_agent,
            to_agent,
            cwd,
            h.summary,
            open_q,
            next_s,
            files,
            now,
        ],
    )?;
    Ok(id)
}

/// Mark a handoff accepted by `accepting_agent` / `accepting_session`.
pub fn accept_handoff(
    conn: &mut Connection,
    handoff_id: &HandoffId,
    accepting_agent: AgentKind,
    accepting_session: Option<&SessionId>,
) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let agent = accepting_agent.as_str();
    let session: Option<&[u8]> = accepting_session.map(|s| &s.as_bytes()[..]);
    conn.execute(
        "UPDATE handoffs SET state = 'accepted', accepted_by = ?1, accepted_at = ?2, \
         accepted_by_session = ?3 \
         WHERE id = ?4 AND state = 'open'",
        params![agent, now, session, handoff_id.as_bytes()],
    )?;
    Ok(())
}

/// Mark an open handoff expired so it will no longer be consumed.
pub fn cancel_handoff(conn: &mut Connection, handoff_id: &HandoffId) -> StoreResult<bool> {
    let changed = conn.execute(
        "UPDATE handoffs SET state = 'expired' WHERE id = ?1 AND state = 'open'",
        params![handoff_id.as_bytes()],
    )?;
    Ok(changed > 0)
}

fn observation_kind_as_str(kind: ObservationKind) -> &'static str {
    kind.as_str()
}

fn audit(
    tx: &rusqlite::Transaction<'_>,
    op: &str,
    workspace_id: Option<&[u8; 16]>,
    project_id: Option<&[u8; 16]>,
    page_id: Option<&[u8; 16]>,
    author_id: Option<&[u8; 16]>,
    at: i64,
) -> StoreResult<()> {
    tx.execute(
        "INSERT INTO audit_log (at, op, workspace_id, project_id, page_id, author_id, detail) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}')",
        params![
            at,
            op,
            workspace_id.map(|b| &b[..]),
            project_id.map(|b| &b[..]),
            page_id.map(|b| &b[..]),
            author_id.map(|b| &b[..]),
        ],
    )?;
    Ok(())
}

/// Retro-fit sessions + observations to per-cwd projects and graveyard
/// any `is_latest=1` pages (which are mash-ups across the old single-project
/// bucket). Executes atomically in one transaction.
///
/// `plan` contains `(session_id, new_project_id)` pairs. Sessions not in
/// the plan are left untouched. Pages are graveyarded unconditionally so a
/// fresh consolidation can regenerate clean per-project pages.
pub fn reorg_sessions(
    conn: &mut Connection,
    plan: &[(SessionId, ProjectId)],
) -> StoreResult<ReorgSummary> {
    if plan.is_empty() {
        return Ok(ReorgSummary::default());
    }
    let tx = conn.transaction()?;
    let mut sessions_moved = 0usize;
    let mut observations_updated = 0usize;
    for (session_id, new_project_id) in plan {
        let rows = tx.execute(
            "UPDATE sessions SET project_id = ?1 WHERE id = ?2 AND project_id != ?1",
            params![new_project_id.as_bytes(), session_id.as_bytes()],
        )?;
        sessions_moved += rows;
        // Update observations whose session_id matches, keeping project_id
        // in sync with the session row we just moved.
        let obs_rows = tx.execute(
            "UPDATE observations SET project_id = ?1 WHERE session_id = ?2",
            params![new_project_id.as_bytes(), session_id.as_bytes()],
        )?;
        observations_updated += obs_rows;
    }
    // Graveyard all current latest pages — they mixed observations from
    // multiple projects and must be regenerated per-project by the next
    // consolidation pass.
    let pages_graveyarded: usize =
        tx.execute("UPDATE pages SET is_latest = 0 WHERE is_latest = 1", [])?;
    tx.commit()?;
    Ok(ReorgSummary {
        sessions_moved,
        observations_updated,
        pages_graveyarded,
    })
}

/// Rename a project within its workspace.
///
/// Only the `name` column is updated — all pages, sessions, observations,
/// and handoffs remain associated with the same `project_id`. No files
/// move on disk (the wiki is flat: every page from every project lives
/// under `wiki/`; only the `project_id` foreign key distinguishes them).
///
/// # Errors
/// - [`StoreError::InvalidProjectName`] when `new_name` is empty,
///   contains a `/` character, or is all whitespace.
/// - [`StoreError::ProjectNameTaken`] when a project with `new_name`
///   already exists in the same workspace.
/// - [`StoreError::Sqlite`] on any other SQL failure.
pub fn rename_project(
    conn: &mut Connection,
    workspace_id: &WorkspaceId,
    project_id: &ProjectId,
    new_name: &str,
) -> StoreResult<()> {
    let trimmed = new_name.trim();
    if trimmed.is_empty() {
        return Err(StoreError::InvalidProjectName(
            "project name must not be empty or all whitespace".into(),
        ));
    }
    if trimmed.contains('/') {
        return Err(StoreError::InvalidProjectName(
            "project name must not contain '/' (it appears in URL paths)".into(),
        ));
    }

    let rows = conn.execute(
        "UPDATE projects SET name = ?1 WHERE id = ?2 AND workspace_id = ?3",
        params![trimmed, project_id.as_bytes(), workspace_id.as_bytes()],
    );

    match rows {
        // Zero rows affected means the project row vanished between the
        // admin handler's `lookup_ws_proj_no_create` and this UPDATE —
        // the classic shape is a concurrent `purge-project` racing the
        // rename. Without this check, the rename would happily return
        // `Ok(())` and the admin handler would respond `200 OK` for an
        // operation that touched nothing, contradicting the purge's
        // (also `200 OK`) destruction of the same row.
        Ok(0) => Err(StoreError::NotFound(format!(
            "project id {project_id} no longer exists in workspace {workspace_id} \
             (race with concurrent purge or delete)",
        ))),
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(err, _))
            if err.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
                || err.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Err(StoreError::ProjectNameTaken(trimmed.to_string()))
        }
        Err(e) => Err(StoreError::Sqlite(e)),
    }
}

/// Record a successfully-applied wiki-structure migration.
///
/// Uses `INSERT OR IGNORE` so re-running the same name is a no-op
/// (idempotent by design — the runner already skips known names, but
/// this guards against any concurrent writes).
pub fn insert_wiki_migration(
    conn: &mut Connection,
    name: &str,
    applied_at: i64,
) -> StoreResult<()> {
    conn.execute(
        "INSERT OR IGNORE INTO wiki_migrations (name, applied_at) VALUES (?1, ?2)",
        params![name, applied_at],
    )?;
    Ok(())
}

/// Delete a project and all its data inside one transaction.
///
/// Execution order:
/// 1. Count rows in each dependent table (pages/all versions, sessions,
///    observations, handoffs, embeddings) before the delete so we can
///    report how many rows were removed.
/// 2. Collect all distinct page paths stored under the project — these are
///    the on-disk files the caller must clean up after this function returns.
/// 3. DELETE FROM projects WHERE id = ? — the ON DELETE CASCADE clauses in
///    V01 + V02 propagate the delete to pages, sessions, observations,
///    handoffs, and page_embeddings automatically.
/// 4. Commit and return the [`PurgeSummary`].
///
/// The `workspace_project_label` string is passed in by the caller (the
/// admin handler has the human-readable names; the writer only has IDs) and
/// forwarded verbatim into [`PurgeSummary::label`] for logging.
///
/// # Errors
/// Returns [`StoreError`] if any SQL statement fails. The transaction is
/// rolled back automatically on error.
pub fn purge_project(
    conn: &mut Connection,
    workspace_id: &WorkspaceId,
    project_id: &ProjectId,
    workspace_project_label: &str,
) -> StoreResult<PurgeSummary> {
    let tx = conn.transaction()?;

    let count = |sql: &str, id: &[u8]| -> StoreResult<u64> {
        let n: Option<i64> = tx
            .query_row(sql, rusqlite::params![id], |row| row.get(0))
            .optional()?;
        Ok(u64::try_from(n.unwrap_or(0)).unwrap_or(0))
    };

    let pid = project_id.as_bytes();
    let pages_deleted = count("SELECT COUNT(*) FROM pages WHERE project_id = ?1", &pid[..])?;
    let sessions_deleted = count(
        "SELECT COUNT(*) FROM sessions WHERE project_id = ?1",
        &pid[..],
    )?;
    let observations_deleted = count(
        "SELECT COUNT(*) FROM observations WHERE project_id = ?1",
        &pid[..],
    )?;
    let handoffs_deleted = count(
        "SELECT COUNT(*) FROM handoffs WHERE project_id = ?1",
        &pid[..],
    )?;
    // page_embeddings cascade through pages; count pages that have them.
    let embeddings_deleted = count(
        "SELECT COUNT(*) FROM page_embeddings \
         WHERE page_id IN (SELECT id FROM pages WHERE project_id = ?1)",
        &pid[..],
    )?;

    // Collect all distinct on-disk paths for the caller to clean up.
    // We use DISTINCT because multiple versions of the same logical page
    // share a path; the file only exists once. The statement must be
    // dropped before we call tx.commit() to release the borrow on `tx`.
    let page_paths: Vec<String> = {
        let mut path_stmt = tx.prepare("SELECT DISTINCT path FROM pages WHERE project_id = ?1")?;
        path_stmt
            .query_map(rusqlite::params![&pid[..]], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?
    };

    // Cascade handles pages / sessions / observations / handoffs /
    // page_embeddings. The workspace row is intentionally left intact —
    // other projects may still live there.
    tx.execute(
        "DELETE FROM projects WHERE id = ?1 AND workspace_id = ?2",
        rusqlite::params![&pid[..], workspace_id.as_bytes()],
    )?;

    tx.commit()?;
    Ok(PurgeSummary {
        label: workspace_project_label.to_string(),
        page_paths,
        pages_deleted,
        sessions_deleted,
        observations_deleted,
        handoffs_deleted,
        embeddings_deleted,
    })
}

/// Summary returned by [`move_project_workspace`] and exposed via
/// [`crate::writer::WriterHandle::move_project_workspace`].
#[derive(Debug, Default, Clone)]
pub struct MoveSummary {
    /// `pages` rows re-stamped (all versions, not just latest).
    pub pages_moved: u64,
    /// `sessions` rows re-stamped.
    pub sessions_moved: u64,
    /// `observations` rows re-stamped.
    pub observations_moved: u64,
    /// `handoffs` rows re-stamped.
    pub handoffs_moved: u64,
    /// `audit_log` rows re-stamped.
    pub audit_log_moved: u64,
}

/// Re-stamp a project's `workspace_id` across every domain table in ONE
/// transaction, keeping the same `project_id`. This is a lossless "true move":
/// pages, sessions, observations, handoffs and supersession history all stay
/// attached to the project — unlike a copy+purge, which drops everything but
/// the latest pages.
///
/// `page_embeddings` and `links` are keyed by `page_id` (not `workspace_id`),
/// so they follow automatically with no re-stamp.
///
/// The destination workspace row MUST already exist (FK on
/// `projects.workspace_id`); the caller get-or-creates it first. A same-named
/// project already living in the destination workspace makes the `projects`
/// UPDATE violate `UNIQUE (workspace_id, name)` and the whole transaction
/// rolls back — the caller must detect that merge case and route it through
/// copy+purge instead.
pub fn move_project_workspace(
    conn: &mut Connection,
    project_id: &ProjectId,
    from_workspace: &WorkspaceId,
    to_workspace: &WorkspaceId,
) -> StoreResult<MoveSummary> {
    let tx = conn.transaction()?;

    let pid = project_id.as_bytes();
    let from = from_workspace.as_bytes();
    let to = to_workspace.as_bytes();

    // Re-stamp child tables first (they carry the denormalized workspace_id),
    // then the project row last. Order is irrelevant inside the transaction,
    // but doing projects last keeps the UNIQUE(workspace_id, name) failure —
    // the merge-collision signal — as the final, cheapest check.
    let pages_moved = tx.execute(
        "UPDATE pages SET workspace_id = ?1 WHERE project_id = ?2",
        params![&to[..], &pid[..]],
    )? as u64;
    let sessions_moved = tx.execute(
        "UPDATE sessions SET workspace_id = ?1 WHERE project_id = ?2",
        params![&to[..], &pid[..]],
    )? as u64;
    let observations_moved = tx.execute(
        "UPDATE observations SET workspace_id = ?1 WHERE project_id = ?2",
        params![&to[..], &pid[..]],
    )? as u64;
    let handoffs_moved = tx.execute(
        "UPDATE handoffs SET workspace_id = ?1 WHERE project_id = ?2",
        params![&to[..], &pid[..]],
    )? as u64;
    let audit_log_moved = tx.execute(
        "UPDATE audit_log SET workspace_id = ?1 WHERE project_id = ?2 AND workspace_id = ?3",
        params![&to[..], &pid[..], &from[..]],
    )? as u64;

    let projects_updated = tx.execute(
        "UPDATE projects SET workspace_id = ?1 WHERE id = ?2 AND workspace_id = ?3",
        params![&to[..], &pid[..], &from[..]],
    )?;
    if projects_updated != 1 {
        return Err(StoreError::NotFound(format!(
            "project {project_id} not found in source workspace {from_workspace}"
        )));
    }

    tx.commit()?;
    Ok(MoveSummary {
        pages_moved,
        sessions_moved,
        observations_moved,
        handoffs_moved,
        audit_log_moved,
    })
}

/// Remove embedding rows in a workspace/project scope whose `(provider, model, dim)`
/// does not match the configured triple, plus rows tied to superseded pages.
pub fn delete_stale_page_embeddings(
    conn: &mut Connection,
    workspace_id: &WorkspaceId,
    project_id: Option<&ProjectId>,
    provider: &str,
    model: &str,
    dim: u32,
) -> StoreResult<u64> {
    let tx = conn.transaction()?;
    let (n, orphans) = if let Some(project_id) = project_id {
        let n = tx.execute(
            "DELETE FROM page_embeddings \
             WHERE page_id IN (\
                SELECT id FROM pages \
                WHERE workspace_id = ?1 AND project_id = ?2 AND is_latest = 1\
             ) \
               AND NOT (provider = ?3 AND model = ?4 AND dim = CAST(?5 AS INTEGER))",
            params![
                workspace_id.as_bytes(),
                project_id.as_bytes(),
                provider,
                model,
                dim
            ],
        )?;
        let orphans = tx.execute(
            "DELETE FROM page_embeddings \
             WHERE page_id IN (\
                SELECT id FROM pages \
                WHERE workspace_id = ?1 AND project_id = ?2 AND is_latest = 0\
             )",
            params![workspace_id.as_bytes(), project_id.as_bytes()],
        )?;
        (n, orphans)
    } else {
        let n = tx.execute(
            "DELETE FROM page_embeddings \
             WHERE page_id IN (\
                SELECT id FROM pages \
                WHERE workspace_id = ?1 AND is_latest = 1\
             ) \
               AND NOT (provider = ?2 AND model = ?3 AND dim = CAST(?4 AS INTEGER))",
            params![workspace_id.as_bytes(), provider, model, dim],
        )?;
        let orphans = tx.execute(
            "DELETE FROM page_embeddings \
             WHERE page_id IN (\
                SELECT id FROM pages \
                WHERE workspace_id = ?1 AND is_latest = 0\
             )",
            params![workspace_id.as_bytes()],
        )?;
        (n, orphans)
    };
    tx.commit()?;
    Ok(u64::try_from(n.saturating_add(orphans)).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    //! Focused unit tests for the load-bearing mutating SQL paths.
    //!
    //! `Store::open` exercises these incidentally through
    //! integration tests, but specific edges — supersession on body
    //! change, no-op on identical body, handoff state transitions,
    //! end_session summary linkage, embedding PK-replacement —
    //! deserve direct coverage so a regression surfaces with a
    //! one-line diff instead of a cascading e2e failure.
    use super::*;
    use ai_memory_core::{
        LinkTarget, NewHandoff, NewPage, NewSession, PagePath, Tier, WorkspaceId,
    };
    use rusqlite::Connection;
    use tempfile::TempDir;

    /// Open a fresh DB with migrations applied + a default workspace
    /// and "scratch" project pre-created. Tuple-return keeps the
    /// tempdir alive for the duration of the test.
    fn fresh_db() -> (
        TempDir,
        Connection,
        ai_memory_core::WorkspaceId,
        ai_memory_core::ProjectId,
    ) {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");
        let mut conn = Connection::open(&db_path).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        crate::migrations::run(&mut conn).unwrap();
        let ws = get_or_create_workspace(&mut conn, "default").unwrap();
        let proj = get_or_create_project(&mut conn, &ws, "scratch", None).unwrap();
        (tmp, conn, ws, proj)
    }

    fn page(
        ws: ai_memory_core::WorkspaceId,
        proj: ai_memory_core::ProjectId,
        path: &str,
        body: &str,
    ) -> NewPage {
        NewPage {
            workspace_id: ws,
            project_id: proj,
            path: PagePath::new(path).unwrap(),
            title: "test".into(),
            body: body.into(),
            tier: Tier::Semantic,
            frontmatter_json: serde_json::json!({}),
            pinned: false,
            links: Vec::new(),
            author_id: None,
        }
    }

    /// Trickier path: upserting a page with a CHANGED body must
    /// produce a NEW row and mark the previous row `is_latest = 0`.
    /// This is the M7 supersession chain — the entire wiki versioning
    /// guarantee rides on it.
    /// V16: every page write lands an `audit_log` row whose
    /// `author_id` mirrors the NewPage's. Anonymous writes leave it
    /// NULL (the entire audit-log-by-author query pattern relies on
    /// the partial index covering only the non-NULL minority).
    #[test]
    fn audit_log_records_author_for_attributed_create_page() {
        use ai_memory_core::UserId;

        let (_tmp, mut conn, ws, proj) = fresh_db();

        // Seed a synthetic user row so the FK on author_id resolves.
        let user_id = UserId::new();
        let now = jiff::Timestamp::now().as_microsecond();
        conn.execute(
            "INSERT INTO users \
             (id, username, name, email, token_hash, created_at) \
             VALUES (?1, 'alice', NULL, NULL, X'00', ?2)",
            params![user_id.as_bytes(), now],
        )
        .unwrap();

        let mut np = page(ws, proj, "notes/by-alice.md", "alice body");
        np.author_id = Some(user_id);
        let page_id = upsert_page(&mut conn, &np).unwrap();

        let author_bytes: Vec<u8> = conn
            .query_row(
                "SELECT author_id FROM audit_log \
                 WHERE op = 'create_page' AND page_id = ?1",
                params![page_id.as_bytes()],
                |r| r.get(0),
            )
            .unwrap();
        let recorded = UserId::from_slice(&author_bytes).unwrap();
        assert_eq!(
            recorded, user_id,
            "create_page audit row must carry the writer's user_id"
        );
    }

    /// Backward-compat gate (and the headline of the "no behaviour
    /// change for legacy installs" promise): anonymous writes leave
    /// audit_log.author_id NULL — the partial index stays empty for
    /// pre-multi-user history.
    #[test]
    fn audit_log_records_null_author_for_anonymous_create_page() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let np = page(ws, proj, "notes/anon.md", "anon body");
        assert!(np.author_id.is_none());
        let page_id = upsert_page(&mut conn, &np).unwrap();

        let author: Option<Vec<u8>> = conn
            .query_row(
                "SELECT author_id FROM audit_log \
                 WHERE op = 'create_page' AND page_id = ?1",
                params![page_id.as_bytes()],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            author.is_none(),
            "anonymous writes must record audit_log.author_id = NULL"
        );
    }

    /// Supersession rows carry the SUPERSEDING author, not the
    /// original. Two consecutive attributed writes (alice then bob)
    /// yield a create_page row tagged alice and a supersede_page row
    /// tagged bob — point-in-time truth, not "latest author".
    #[test]
    fn audit_log_supersede_records_new_authors_identity() {
        use ai_memory_core::UserId;

        let (_tmp, mut conn, ws, proj) = fresh_db();
        let now = jiff::Timestamp::now().as_microsecond();
        let alice = UserId::new();
        let bob = UserId::new();
        conn.execute(
            "INSERT INTO users (id, username, name, email, token_hash, created_at) \
             VALUES (?1, 'alice', NULL, NULL, X'01', ?2), \
                    (?3, 'bob',   NULL, NULL, X'02', ?2)",
            params![alice.as_bytes(), now, bob.as_bytes()],
        )
        .unwrap();

        let mut np1 = page(ws, proj, "notes/shared.md", "v1");
        np1.author_id = Some(alice);
        upsert_page(&mut conn, &np1).unwrap();

        let mut np2 = page(ws, proj, "notes/shared.md", "v2 — different body");
        np2.author_id = Some(bob);
        let v2_id = upsert_page(&mut conn, &np2).unwrap();

        let bob_bytes: Vec<u8> = conn
            .query_row(
                "SELECT author_id FROM audit_log \
                 WHERE op = 'supersede_page' AND page_id = ?1",
                params![v2_id.as_bytes()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            UserId::from_slice(&bob_bytes).unwrap(),
            bob,
            "supersede_page audit row must carry the SUPERSEDING author"
        );
    }

    #[test]
    fn upsert_page_supersedes_on_body_change() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let id1 = upsert_page(&mut conn, &page(ws, proj, "notes/foo.md", "v1 body")).unwrap();
        let id2 = upsert_page(&mut conn, &page(ws, proj, "notes/foo.md", "v2 body")).unwrap();

        assert_ne!(id1, id2, "supersession must produce a new row id");

        let latest_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages WHERE path = ?1 AND is_latest = 1",
                params!["notes/foo.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(latest_count, 1, "exactly one latest version expected");

        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages WHERE path = ?1",
                params!["notes/foo.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(total, 2, "old version must remain on disk for history");

        // The newest row should point at the older as its predecessor
        // (supersedes column), so chains are reconstructible.
        let supersedes: Option<Vec<u8>> = conn
            .query_row(
                "SELECT supersedes FROM pages WHERE id = ?1",
                params![&id2.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert!(supersedes.is_some(), "new row must link to its predecessor");
    }

    /// Idempotency: re-upserting the same body should NOT create a
    /// second row. The watcher's reconciliation calls upsert_page on
    /// every file on every tick — without this, a quiet repo would
    /// accumulate spurious history every 30 seconds.
    #[test]
    fn upsert_page_is_noop_when_body_unchanged() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let p = page(ws, proj, "notes/foo.md", "same body");
        let id1 = upsert_page(&mut conn, &p).unwrap();
        let id2 = upsert_page(&mut conn, &p).unwrap();

        assert_eq!(id1, id2, "identical body should not supersede");
        conn.execute(
            "UPDATE pages SET updated_at = 123 WHERE id = ?1",
            params![id1.as_bytes()],
        )
        .unwrap();
        let id3 = upsert_page(&mut conn, &p).unwrap();
        assert_eq!(id1, id3, "identical body should keep the same page id");
        let updated_at: i64 = conn
            .query_row(
                "SELECT updated_at FROM pages WHERE id = ?1",
                params![id1.as_bytes()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            updated_at, 123,
            "unchanged content should not dirty the row"
        );
        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages WHERE path = ?1",
                params!["notes/foo.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(total, 1, "no duplicate row for unchanged content");
    }

    #[test]
    fn upsert_page_supersedes_on_frontmatter_change() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let mut p1 = page(ws, proj, "_slots/project_context.md", "same body");
        p1.frontmatter_json = serde_json::json!({
            "title": "Project context",
            "slot_kind": "state",
        });
        let id1 = upsert_page(&mut conn, &p1).unwrap();

        let mut p2 = p1.clone();
        p2.frontmatter_json = serde_json::json!({
            "title": "Project context",
            "slot_kind": "invariant",
        });
        let id2 = upsert_page(&mut conn, &p2).unwrap();

        assert_ne!(id1, id2, "frontmatter-only changes must supersede");
        let latest_frontmatter: String = conn
            .query_row(
                "SELECT frontmatter_json FROM pages WHERE id = ?1 AND is_latest = 1",
                params![id2.as_bytes()],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            latest_frontmatter.contains("invariant"),
            "latest row should store the updated slot_kind"
        );
    }

    #[test]
    fn upsert_page_persists_and_resolves_links() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let mut source = page(ws, proj, "concepts/source.md", "see target");
        source.links = vec![PagePath::new("decisions/target.md").unwrap().into()];
        let source_id = upsert_page(&mut conn, &source).unwrap();

        let unresolved: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM links \
                 WHERE from_page_id = ?1 AND to_path = ?2 AND to_page_id IS NULL",
                params![source_id.as_bytes(), "decisions/target.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(unresolved, 1, "forward link should persist unresolved");

        let target_id = upsert_page(
            &mut conn,
            &page(ws, proj, "decisions/target.md", "target body"),
        )
        .unwrap();

        let resolved: Option<Vec<u8>> = conn
            .query_row(
                "SELECT to_page_id FROM links WHERE from_page_id = ?1 AND to_path = ?2",
                params![source_id.as_bytes(), "decisions/target.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(resolved.as_deref(), Some(&target_id.as_bytes()[..]));
    }

    /// A `[[infra:runbooks/02.md]]` link from one project resolves to a page
    /// in a sibling project once that page exists — the cross-project edge.
    #[test]
    fn upsert_page_resolves_cross_project_link() {
        let (_tmp, mut conn, ws, scratch) = fresh_db();
        let infra = get_or_create_project(&mut conn, &ws, "infra", None).unwrap();

        let mut source = page(ws, scratch, "concepts/dep.md", "depends on infra runbook");
        source.links = vec![LinkTarget {
            workspace: None,
            project: Some("infra".into()),
            path: PagePath::new("runbooks/02.md").unwrap(),
        }];
        let source_id = upsert_page(&mut conn, &source).unwrap();

        // Persisted with the scope, unresolved until the target project's page exists.
        let (to_project, resolved): (Option<String>, Option<Vec<u8>>) = conn
            .query_row(
                "SELECT to_project, to_page_id FROM links \
                 WHERE from_page_id = ?1 AND to_path = ?2",
                params![source_id.as_bytes(), "runbooks/02.md"],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(to_project.as_deref(), Some("infra"));
        assert!(
            resolved.is_none(),
            "cross-project link is unresolved before the target exists"
        );

        // Create the target in `infra` → the forward link repoints across projects.
        let target_id =
            upsert_page(&mut conn, &page(ws, infra, "runbooks/02.md", "the runbook")).unwrap();
        let resolved: Option<Vec<u8>> = conn
            .query_row(
                "SELECT to_page_id FROM links WHERE from_page_id = ?1 AND to_path = ?2",
                params![source_id.as_bytes(), "runbooks/02.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            resolved.as_deref(),
            Some(&target_id.as_bytes()[..]),
            "link must resolve across projects once the target lands"
        );
    }

    /// Handoff state machine: insert → Open; accept_handoff → Accepted
    /// with accepted_by stamped. Calling accept again must be safe
    /// (idempotent at the DB level) because hooks fire-and-forget.
    #[test]
    fn accept_handoff_transitions_open_to_accepted() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let new = NewHandoff {
            workspace_id: ws,
            project_id: proj,
            from_session_id: None,
            from_agent: AgentKind::ClaudeCode,
            to_agent: None,
            cwd: None,
            summary: "test summary".into(),
            open_questions: vec![],
            next_steps: vec![],
            files_touched: vec![],
        };
        let id = insert_handoff(&mut conn, &new).unwrap();

        // Pre-state: Open, accepted_by NULL.
        let (state, accepted_by): (String, Option<String>) = conn
            .query_row(
                "SELECT state, accepted_by FROM handoffs WHERE id = ?1",
                params![&id.as_bytes()[..]],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(state, "open");
        assert!(accepted_by.is_none());

        accept_handoff(&mut conn, &id, AgentKind::Codex, None).unwrap();
        let (state, accepted_by): (String, Option<String>) = conn
            .query_row(
                "SELECT state, accepted_by FROM handoffs WHERE id = ?1",
                params![&id.as_bytes()[..]],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(state, "accepted");
        assert_eq!(accepted_by.as_deref(), Some("codex"));

        // Idempotency: accepting an already-accepted handoff must
        // either succeed silently or fail clearly, never corrupt
        // the row. (Current impl is a no-op UPDATE with a state
        // guard.)
        let second = accept_handoff(&mut conn, &id, AgentKind::Codex, None);
        assert!(second.is_ok(), "double-accept must not error");
    }

    #[test]
    fn cancel_handoff_transitions_open_to_expired() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let new = NewHandoff {
            workspace_id: ws,
            project_id: proj,
            from_session_id: None,
            from_agent: AgentKind::ClaudeCode,
            to_agent: None,
            cwd: None,
            summary: "accidental handoff".into(),
            open_questions: vec![],
            next_steps: vec![],
            files_touched: vec![],
        };
        let id = insert_handoff(&mut conn, &new).unwrap();

        assert!(cancel_handoff(&mut conn, &id).unwrap());
        let state: String = conn
            .query_row(
                "SELECT state FROM handoffs WHERE id = ?1",
                params![&id.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(state, "expired");

        assert!(
            !cancel_handoff(&mut conn, &id).unwrap(),
            "double-cancel should be a no-op"
        );
    }

    /// Supported hook agents persist concrete agent_kind values. V01's CHECK
    /// omitted agents added after launch; regression for hook-router WARNs.
    #[test]
    fn begin_session_accepts_all_supported_agent_kinds() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        for agent_kind in [
            AgentKind::ClaudeCode,
            AgentKind::Codex,
            AgentKind::OpenCode,
            AgentKind::Cursor,
            AgentKind::GeminiCli,
            AgentKind::ClaudeDesktop,
            AgentKind::OpenClaw,
            AgentKind::AntigravityCli,
            AgentKind::Omp,
            AgentKind::Other,
        ] {
            let sid = SessionId::new();
            begin_session(
                &mut conn,
                &NewSession {
                    id: sid,
                    workspace_id: ws,
                    project_id: proj,
                    agent_kind,
                    cwd: Some(std::path::PathBuf::from(r"C:\GIT\ai-memory")),
                },
            )
            .unwrap();

            let stored: String = conn
                .query_row(
                    "SELECT agent_kind FROM sessions WHERE id = ?1",
                    params![&sid.as_bytes()[..]],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(stored, agent_kind.as_str());
        }
    }

    /// end_session links the synthesised summary page so callers can
    /// jump straight from session row to summary.
    #[test]
    fn end_session_links_summary_page_when_provided() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let sid = SessionId::new();
        begin_session(
            &mut conn,
            &NewSession {
                id: sid,
                workspace_id: ws,
                project_id: proj,
                agent_kind: AgentKind::ClaudeCode,
                cwd: None,
            },
        )
        .unwrap();
        let page_id = upsert_page(
            &mut conn,
            &page(ws, proj, "sessions/abc.md", "summary body"),
        )
        .unwrap();
        end_session(&mut conn, &sid, Some(&page_id)).unwrap();

        let summary: Option<Vec<u8>> = conn
            .query_row(
                "SELECT summary_page_id FROM sessions WHERE id = ?1",
                params![&sid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            summary.is_some(),
            "summary_page_id must persist when supplied"
        );
        let bytes = summary.unwrap();
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[..], &page_id.as_bytes()[..]);
    }

    /// end_session without a summary leaves the column NULL — the
    /// session ended but no page was synthesised (e.g. zero
    /// observations recorded). This must not be confused with the
    /// summary-linked case.
    #[test]
    fn end_session_without_summary_page_id_leaves_null() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let sid = SessionId::new();
        begin_session(
            &mut conn,
            &NewSession {
                id: sid,
                workspace_id: ws,
                project_id: proj,
                agent_kind: AgentKind::ClaudeCode,
                cwd: None,
            },
        )
        .unwrap();
        end_session(&mut conn, &sid, None).unwrap();
        let summary: Option<Vec<u8>> = conn
            .query_row(
                "SELECT summary_page_id FROM sessions WHERE id = ?1",
                params![&sid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert!(summary.is_none());
    }

    /// Embeddings are keyed by page_id (PK). Re-storing for the same
    /// page must REPLACE, not duplicate — otherwise `ai-memory embed
    /// --reembed` would multiply rows on each run.
    #[test]
    fn store_embedding_replaces_existing_row() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let pid = upsert_page(&mut conn, &page(ws, proj, "notes/x.md", "body")).unwrap();
        store_embedding(
            &mut conn,
            &pid,
            &vec![0u8; 1536 * 4],
            "test",
            "model-a",
            1536,
        )
        .unwrap();
        store_embedding(
            &mut conn,
            &pid,
            &vec![1u8; 1536 * 4],
            "test",
            "model-b",
            1536,
        )
        .unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM page_embeddings WHERE page_id = ?1",
                params![&pid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "embedding row must be replaced, not duplicated");

        let model: String = conn
            .query_row(
                "SELECT model FROM page_embeddings WHERE page_id = ?1",
                params![&pid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(model, "model-b", "latest model metadata wins");
    }

    #[test]
    fn store_embeddings_batches_rows_in_one_call() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let p1 = upsert_page(&mut conn, &page(ws, proj, "notes/a.md", "body a")).unwrap();
        let p2 = upsert_page(&mut conn, &page(ws, proj, "notes/b.md", "body b")).unwrap();

        store_embeddings(
            &mut conn,
            &[
                EmbeddingWrite {
                    page_id: p1,
                    vector_bytes: vec![0u8; 4],
                    provider: "test".into(),
                    model: "model".into(),
                    dim: 1,
                },
                EmbeddingWrite {
                    page_id: p2,
                    vector_bytes: vec![1u8; 4],
                    provider: "test".into(),
                    model: "model".into(),
                    dim: 1,
                },
            ],
        )
        .unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM page_embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn delete_stale_page_embeddings_removes_mismatched_rows() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let other = get_or_create_project(&mut conn, &ws, "other", None).unwrap();
        let p1 = upsert_page(&mut conn, &page(ws, proj, "a.md", "body a")).unwrap();
        let p2 = upsert_page(&mut conn, &page(ws, proj, "b.md", "body b")).unwrap();
        let p3 = upsert_page(&mut conn, &page(ws, other, "c.md", "body c")).unwrap();
        let old = upsert_page(&mut conn, &page(ws, proj, "old.md", "old body")).unwrap();
        let _new = upsert_page(&mut conn, &page(ws, proj, "old.md", "new body")).unwrap();
        store_embedding(
            &mut conn,
            &p1,
            &[0u8; 4],
            "google",
            "models/gemini-embedding-001",
            768,
        )
        .unwrap();
        store_embedding(
            &mut conn,
            &p3,
            &[2u8; 4],
            "google",
            "models/gemini-embedding-001",
            768,
        )
        .unwrap();
        store_embedding(
            &mut conn,
            &p2,
            &[1u8; 4],
            "openai",
            "openai/text-embedding-3-small",
            1536,
        )
        .unwrap();
        store_embedding(
            &mut conn,
            &old,
            &[3u8; 4],
            "openai",
            "openai/text-embedding-3-small",
            1536,
        )
        .unwrap();
        let n = super::delete_stale_page_embeddings(
            &mut conn,
            &ws,
            Some(&proj),
            "openai",
            "openai/text-embedding-3-small",
            1536,
        )
        .unwrap();
        assert_eq!(n, 2);
        let remaining: i64 = conn
            .query_row("SELECT COUNT(*) FROM page_embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, 2);
        let model: String = conn
            .query_row(
                "SELECT model FROM page_embeddings WHERE page_id = ?1",
                params![&p2.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(model, "openai/text-embedding-3-small");
        let other_rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM page_embeddings WHERE page_id = ?1",
                params![&p3.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            other_rows, 1,
            "explicit project purge must not touch siblings"
        );
    }

    #[test]
    fn path_search_text_indexes_slug_and_words() {
        // Both forms: hyphenated slug kept whole, plus split into words.
        assert_eq!(
            path_search_text("notes/foo-bar.md"),
            "notes foo-bar md notes foo bar md"
        );
        assert_eq!(path_search_text("a/b_c.md"), "a b_c md a b c md");
    }

    /// A page is findable by its PATH slug even when the slug appears in
    /// neither the title nor the body — the V17 `path_search` FTS column.
    #[test]
    fn fts_matches_page_by_path_slug_not_in_body() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        // Title + body deliberately avoid the slug words.
        let mut p = page(
            ws,
            proj,
            "notes/followup-bulk-rename-runbook-titles.md",
            "totally unrelated prose about elephants",
        );
        p.title = "Elephants".into();
        upsert_page(&mut conn, &p).unwrap();

        // The slug, as a quoted single token (how prepare_fts5_query renders a
        // hyphenated term), matches via the path_search column.
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages_fts \
                 WHERE pages_fts MATCH ?1",
                params!["\"followup-bulk-rename-runbook-titles\""],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "slug in path must be searchable");

        // A distinct path segment is independently searchable too.
        let seg: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages_fts WHERE pages_fts MATCH 'runbook'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(seg, 1, "path segment token must match");

        // Body words still match (regression: body stays indexed at col 1).
        let body_hit: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages_fts WHERE pages_fts MATCH 'elephants'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(body_hit, 1, "body must remain searchable");
    }

    #[test]
    fn pages_fts_path_migration_preserves_accent_folding() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let mut p = page(ws, proj, "notes/descricao.md", "descrição do projeto");
        p.title = "Descrição".into();
        upsert_page(&mut conn, &p).unwrap();

        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages_fts WHERE pages_fts MATCH 'descricao'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "page FTS should remain accent-insensitive");
    }

    #[test]
    fn pages_fts_update_trigger_ignores_access_counter_updates() {
        let (_tmp, conn, _ws, _proj) = fresh_db();
        let sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'pages_fts_au'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            sql.contains("AFTER UPDATE OF title, body, path_search ON pages"),
            "pages_fts_au must not fire on access_count/last_accessed_at updates: {sql}"
        );
    }

    /// True move: re-stamping a project's workspace_id keeps the same
    /// project_id and carries pages, sessions and observations along —
    /// the whole point of the lossless move. The summary counts must
    /// match what actually moved.
    #[test]
    fn move_project_workspace_restamps_all_domain_rows() {
        use ai_memory_core::ObservationKind;

        let (_tmp, mut conn, src_ws, proj) = fresh_db();
        let dst_ws = get_or_create_workspace(&mut conn, "djalmajr").unwrap();

        // Seed a page, a session and an observation under the source ws.
        let page_id = upsert_page(&mut conn, &page(src_ws, proj, "notes/a.md", "body")).unwrap();
        let sid = SessionId::new();
        begin_session(
            &mut conn,
            &NewSession {
                id: sid,
                workspace_id: src_ws,
                project_id: proj,
                agent_kind: AgentKind::ClaudeCode,
                cwd: None,
            },
        )
        .unwrap();
        insert_observation(
            &mut conn,
            &NewObservation {
                session_id: sid,
                workspace_id: src_ws,
                project_id: proj,
                kind: ObservationKind::UserPrompt,
                extension: None,
                source_event: None,
                title: "t".into(),
                body: "b".into(),
                importance: 5,
            },
        )
        .unwrap();

        let summary = move_project_workspace(&mut conn, &proj, &src_ws, &dst_ws).unwrap();
        assert_eq!(summary.pages_moved, 1);
        assert_eq!(summary.sessions_moved, 1);
        assert_eq!(summary.observations_moved, 1);

        // The project_id is unchanged; every row now points at dst_ws.
        // `projects` keys the project by `id`; child tables by `project_id`.
        let count_in = |table: &str, ws: &ai_memory_core::WorkspaceId| -> i64 {
            let id_col = if table == "projects" {
                "id"
            } else {
                "project_id"
            };
            conn.query_row(
                &format!("SELECT COUNT(*) FROM {table} WHERE {id_col} = ?1 AND workspace_id = ?2"),
                params![&proj.as_bytes()[..], ws.as_bytes()],
                |r| r.get(0),
            )
            .unwrap()
        };
        for table in ["projects", "pages", "sessions", "observations"] {
            assert_eq!(count_in(table, &dst_ws), 1, "{table} must move to dst ws");
            assert_eq!(count_in(table, &src_ws), 0, "{table} must leave src ws");
        }
        // The page keeps its id (embeddings/links follow via page_id).
        let still_there: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages WHERE id = ?1 AND workspace_id = ?2",
                params![&page_id.as_bytes()[..], dst_ws.as_bytes()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(still_there, 1);
    }

    /// A same-named project already in the destination workspace makes the
    /// projects UPDATE collide with UNIQUE(workspace_id, name); the whole
    /// transaction must roll back, leaving the source intact. The admin
    /// layer detects this merge case up front and routes it to copy+purge.
    #[test]
    fn move_project_workspace_rolls_back_on_name_collision() {
        let (_tmp, mut conn, src_ws, proj) = fresh_db();
        let dst_ws = get_or_create_workspace(&mut conn, "djalmajr").unwrap();
        // Destination already holds a project named "scratch".
        get_or_create_project(&mut conn, &dst_ws, "scratch", None).unwrap();
        upsert_page(&mut conn, &page(src_ws, proj, "notes/a.md", "body")).unwrap();

        let err = move_project_workspace(&mut conn, &proj, &src_ws, &dst_ws);
        assert!(err.is_err(), "name collision must fail the move");

        // Source rows untouched after rollback.
        let src_pages: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages WHERE project_id = ?1 AND workspace_id = ?2",
                params![&proj.as_bytes()[..], src_ws.as_bytes()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(src_pages, 1, "rollback must preserve source pages");
    }

    #[test]
    fn ensure_project_workspace_rejects_stale_pair_before_disk_write() {
        let (_tmp, conn, ws, proj) = fresh_db();
        let other_ws = WorkspaceId::new();

        ensure_project_workspace(&conn, &ws, &proj).unwrap();
        assert!(
            matches!(
                ensure_project_workspace(&conn, &other_ws, &proj),
                Err(StoreError::NotFound(_))
            ),
            "a stale workspace/project pair must fail before wiki writes touch disk"
        );
    }

    #[test]
    fn v18_migration_refuses_existing_split_brain_rows() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");
        let mut conn = Connection::open(&db_path).unwrap();
        crate::migrations::run_to(&mut conn, 17).unwrap();

        let src_ws = get_or_create_workspace(&mut conn, "src").unwrap();
        let stale_ws = get_or_create_workspace(&mut conn, "stale").unwrap();
        let proj = get_or_create_project(&mut conn, &src_ws, "scratch", None).unwrap();
        let mut bad_page = page(src_ws, proj, "notes/split.md", "body");
        bad_page.workspace_id = stale_ws;
        upsert_page(&mut conn, &bad_page).unwrap();

        let err = crate::migrations::run_to(&mut conn, 18).unwrap_err();
        assert!(
            err.to_string().contains("CHECK constraint failed"),
            "V18 must abort instead of preserving split-brain rows: {err}"
        );
    }

    /// V18 integrity triggers: an INSERT whose `workspace_id` disagrees with
    /// the project's actual workspace ABORTs (the split-brain a stale hook
    /// cache would otherwise create), while the consistent pair inserts fine.
    #[test]
    fn insert_with_mismatched_workspace_is_rejected() {
        use ai_memory_core::ObservationKind;

        let (_tmp, mut conn, ws, proj) = fresh_db();
        let other_ws = get_or_create_workspace(&mut conn, "other").unwrap();

        // A page under the WRONG workspace (project lives in `ws`) is refused.
        let mut bad_page = page(ws, proj, "notes/a.md", "body");
        bad_page.workspace_id = other_ws;
        assert!(
            upsert_page(&mut conn, &bad_page).is_err(),
            "page insert with mismatched workspace must abort"
        );

        // The consistent pair inserts fine.
        upsert_page(&mut conn, &page(ws, proj, "notes/a.md", "body")).unwrap();

        // The session insert is guarded too: a mismatched pair aborts.
        let bad_sid = SessionId::new();
        assert!(
            begin_session(
                &mut conn,
                &NewSession {
                    id: bad_sid,
                    workspace_id: other_ws,
                    project_id: proj,
                    agent_kind: AgentKind::ClaudeCode,
                    cwd: None,
                },
            )
            .is_err(),
            "session insert with mismatched workspace must abort"
        );

        let sid = SessionId::new();
        begin_session(
            &mut conn,
            &NewSession {
                id: sid,
                workspace_id: ws,
                project_id: proj,
                agent_kind: AgentKind::ClaudeCode,
                cwd: None,
            },
        )
        .unwrap();

        // The split-brain case the maintainer flagged: a hook writes an
        // observation with a stale workspace id for a moved project.
        let mismatched_obs = NewObservation {
            session_id: sid,
            workspace_id: other_ws,
            project_id: proj,
            kind: ObservationKind::UserPrompt,
            extension: None,
            source_event: None,
            title: "t".into(),
            body: "b".into(),
            importance: 5,
        };
        assert!(
            insert_observation(&mut conn, &mismatched_obs).is_err(),
            "observation insert with mismatched workspace must abort"
        );

        // Same observation under the correct workspace is accepted.
        let good_obs = NewObservation {
            workspace_id: ws,
            ..mismatched_obs
        };
        insert_observation(&mut conn, &good_obs).unwrap();

        // The handoff insert is the fourth INSERT trigger; audit flagged
        // that the original V18 test omitted it, so the only coverage
        // for the handoffs trigger was the temp-table CHECK on migration.
        // Assert the BEFORE INSERT trigger fires on a stale pair, and
        // a corrected pair lands cleanly.
        let mismatched_handoff = NewHandoff {
            workspace_id: other_ws,
            project_id: proj,
            from_session_id: None,
            from_agent: AgentKind::ClaudeCode,
            to_agent: None,
            cwd: None,
            summary: "stale".into(),
            open_questions: vec![],
            next_steps: vec![],
            files_touched: vec![],
        };
        assert!(
            insert_handoff(&mut conn, &mismatched_handoff).is_err(),
            "handoff insert with mismatched workspace must abort"
        );
        let good_handoff = NewHandoff {
            workspace_id: ws,
            ..mismatched_handoff
        };
        insert_handoff(&mut conn, &good_handoff).unwrap();
    }

    /// Regression for the rename-vs-purge race that live exploration
    /// caught: a `rename_project` for a row that was deleted between
    /// the admin handler's `lookup_ws_proj_no_create` and the
    /// `UPDATE projects` used to silently return `Ok(())` — the admin
    /// endpoint then responded `200 OK` for an operation that touched
    /// zero rows, contradicting the concurrent purge's (also `200 OK`)
    /// destruction of the same project. After the fix, the writer
    /// returns `StoreError::NotFound`, which the admin handler maps to
    /// `404 Not Found`. Pins both the writer-side semantic and a
    /// concrete recipe for the failure shape so a future refactor
    /// can't quietly downgrade the error back to a silent Ok.
    #[test]
    fn rename_project_after_purge_returns_not_found() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        // Simulate the post-purge state: project row gone.
        // `purge_project` drives the cascading deletes we want here.
        let _ = purge_project(&mut conn, &ws, &proj, "default/scratch")
            .expect("purge of fresh project should succeed");
        // Now try to rename the project that no longer exists. The
        // pre-fix code returned `Ok(())` because `UPDATE` affected
        // zero rows. The fix returns `NotFound` so admin handlers
        // can respond 404 honestly.
        let err = rename_project(&mut conn, &ws, &proj, "renamed")
            .expect_err("rename of purged project must error");
        match err {
            StoreError::NotFound(_) => {}
            other => panic!("expected StoreError::NotFound, got {other:?}"),
        }
    }

    /// Belt-and-suspenders for the common path: rename of an existing
    /// project still succeeds. Without this, a future "always return
    /// NotFound" regression would also pass the test above by accident.
    #[test]
    fn rename_project_of_live_project_succeeds() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        rename_project(&mut conn, &ws, &proj, "renamed-live")
            .expect("rename of live project must succeed");
    }
}
