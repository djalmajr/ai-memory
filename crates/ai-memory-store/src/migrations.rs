//! refinery-driven schema migrations.

use std::path::Path;

use rusqlite::{Connection, OpenFlags};

use crate::error::{StoreError, StoreResult};

refinery::embed_migrations!("migrations");

/// Checksum recorded by refinery for the V62 SQL shipped in v2.2.1.
///
/// The historical SQL is pinned in
/// `testdata/V62__page_ingestion_windows.pre776.sql`; a regression test
/// derives this value through refinery's own checksum implementation.
const ORIGINAL_V62_CHECKSUM: u64 = 16_518_128_245_137_964_566;
const V62_VERSION: u32 = 62;
const V62_NAME: &str = "page_ingestion_windows";

/// History-row name the fork gave its chunk-progress table before upstream
/// took the version numbers it had used.
const FORK_CHUNK_PROGRESS_NAME: &str = "consolidation_chunk_progress";
const FORK_CHUNK_PROGRESS_VERSIONS: [u32; 2] = [72, 81];
/// Version the fork's last numbered chunk-progress migration used; upstream
/// 3.2.0 ships its own V84.
const FORK_CHUNK_PROGRESS_V84: u32 = 84;
/// Fork-only table created outside the refinery numbering, after the runner.
const FORK_CHUNK_PROGRESS_DDL: &str = include_str!("consolidation_chunk_progress.sql");

/// refinery 0.10 reports `Migration::version()` as signed
/// (`SchemaVersion` = `i32`) where 0.8 returned `u32`; the values are the
/// same. A version either comes from refinery parsing a `V{digits}__{name}`
/// name or from the store's own history rows, so a negative value is
/// corruption, not a version: reject it with the number in the error
/// instead of saturating it into the "no version" baseline (0). Use this
/// only at the boundaries that must keep `u32`; prefer exact i32
/// comparisons for the internals.
fn version_u32(version: i32) -> StoreResult<u32> {
    u32::try_from(version).map_err(|_| {
        StoreError::InvalidState(format!(
            "migration version {version} is negative; expected a non-negative V{{digits}} version"
        ))
    })
}

/// Run all pending migrations against an open connection.
///
/// # Errors
/// Propagates the underlying refinery error if a migration fails. When the
/// store is *ahead* of this binary, the error is remapped to
/// [`StoreError::DataSchemaAhead`]. An earlier embedded migration missing from
/// the store's history instead reports [`StoreError::MigrationHistoryGap`].
/// refinery reports both conditions as `MissingVersion`.
pub fn run(conn: &mut rusqlite::Connection) -> StoreResult<()> {
    repair_known_migration_checksums(conn)?;
    repair_fork_chunk_progress_rows(conn)?;
    forget_fork_v84_history_row(conn)?;
    migrations::runner()
        .set_abort_divergent(true)
        .set_abort_missing(true)
        .run(conn)
        .map_err(classify_run_error)?;
    conn.execute_batch(FORK_CHUNK_PROGRESS_DDL)?;
    Ok(())
}

/// Rewrite only checksums whose SQL bytes are known to be equivalent.
///
/// Windows release builds before v3.1.2 embedded the checked-out migration
/// files with CRLF endings. The SQL and resulting schema are identical to the
/// LF files embedded now, but refinery hashes the raw bytes. V62 also has one
/// older released layout whose equivalent backfill moved to the bounded boot
/// path (#776). Match version, name and the exact historical checksum before
/// changing anything; every other mismatch remains a hard refinery error.
fn repair_known_migration_checksums(conn: &mut rusqlite::Connection) -> StoreResult<()> {
    let has_history: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type = 'table' AND name = 'refinery_schema_history')",
        [],
        |row| row.get(0),
    )?;
    if !has_history {
        return Ok(());
    }
    let runner = migrations::runner();
    let mut known = Vec::new();
    for migration in runner.get_migrations() {
        let Some(crlf_checksum) = crlf_checksum(migration)? else {
            continue;
        };
        known.push((
            version_u32(migration.version())?,
            migration.name().to_owned(),
            crlf_checksum,
            migration.checksum(),
        ));
    }
    let Some(v62) = runner
        .get_migrations()
        .iter()
        .find(|migration| migration.version() == V62_VERSION as i32)
    else {
        return Ok(());
    };
    known.push((
        V62_VERSION,
        V62_NAME.to_owned(),
        ORIGINAL_V62_CHECKSUM,
        v62.checksum(),
    ));
    let original_v62 = include_str!("testdata/V62__page_ingestion_windows.pre776.sql");
    known.push((
        V62_VERSION,
        V62_NAME.to_owned(),
        checksum_for_sql(V62_VERSION, V62_NAME, &original_v62.replace('\n', "\r\n")),
        v62.checksum(),
    ));

    let tx = conn.transaction()?;
    let mut repaired = 0_usize;
    for (version, name, historical, current) in known {
        repaired += tx.execute(
            "UPDATE refinery_schema_history SET checksum = ?1 \
             WHERE version = ?2 AND name = ?3 AND checksum = ?4",
            rusqlite::params![current.to_string(), version, name, historical.to_string()],
        )?;
    }
    tx.commit()?;
    if repaired > 0 {
        tracing::info!(
            migrations = repaired,
            "repaired known byte-equivalent migration checksums"
        );
    }
    Ok(())
}

fn crlf_checksum(migration: &refinery::Migration) -> StoreResult<Option<u64>> {
    let Some(sql) = migration.sql() else {
        return Ok(None);
    };
    Ok(Some(checksum_for_sql(
        version_u32(migration.version())?,
        migration.name(),
        &sql.replace('\n', "\r\n"),
    )))
}

fn checksum_for_sql(version: u32, name: &str, sql: &str) -> u64 {
    refinery::Migration::unapplied(&format!("V{version}__{name}"), sql)
        .expect("embedded migration identity must remain valid")
        .checksum()
}

/// Rewrite the fork's `consolidation_chunk_progress` history rows at V72
/// (recorded before upstream 3.0) and V81 (before 3.1.1) to the embedded
/// migration of that version. Upstream's V72 and V81 only rebuild `sessions`
/// with an `agent_kind` list that V73 and V83 rebuild again with the complete
/// list, so marking them applied describes the real schema; the fork table
/// itself is created by `run()` after the runner, outside the refinery
/// numbering, with `IF NOT EXISTS`. Only that exact name is
/// matched, with any checksum: no other build wrote it at these versions.
/// Every other divergence remains a hard refinery error.
fn repair_fork_chunk_progress_rows(conn: &rusqlite::Connection) -> StoreResult<()> {
    let has_history: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type = 'table' AND name = 'refinery_schema_history')",
        [],
        |row| row.get(0),
    )?;
    if !has_history {
        return Ok(());
    }
    let runner = migrations::runner();
    for version in FORK_CHUNK_PROGRESS_VERSIONS {
        let Some(migration) = runner
            .get_migrations()
            .iter()
            .find(|migration| migration.version() == version as i32)
        else {
            continue;
        };
        let repaired = conn.execute(
            "UPDATE refinery_schema_history SET name = ?1, checksum = ?2 \
             WHERE version = ?3 AND name = ?4",
            rusqlite::params![
                migration.name(),
                migration.checksum().to_string(),
                version,
                FORK_CHUNK_PROGRESS_NAME
            ],
        )?;
        if repaired == 1 {
            tracing::info!(
                version,
                "repaired fork consolidation_chunk_progress migration history row"
            );
        }
    }
    Ok(())
}

/// Delete the history row `84 consolidation_chunk_progress` written by fork
/// builds that numbered the chunk-progress table V84. No embedded migration
/// carries that row any more, so `set_abort_missing(true)` would refuse the
/// store; the table it created stays and `run()` keeps it current. Only that
/// exact name is deleted: any other V84 row is left for the runner to judge.
fn forget_fork_v84_history_row(conn: &rusqlite::Connection) -> StoreResult<()> {
    let has_history: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type = 'table' AND name = 'refinery_schema_history')",
        [],
        |row| row.get(0),
    )?;
    if !has_history {
        return Ok(());
    }
    let forgotten = conn.execute(
        "DELETE FROM refinery_schema_history WHERE version = ?1 AND name = ?2",
        rusqlite::params![FORK_CHUNK_PROGRESS_V84, FORK_CHUNK_PROGRESS_NAME],
    )?;
    if forgotten == 1 {
        tracing::info!(
            version = FORK_CHUNK_PROGRESS_V84,
            "forgot fork consolidation_chunk_progress migration history row"
        );
    }
    Ok(())
}

/// Highest schema version baked into this binary (the max embedded migration).
fn max_supported_version() -> StoreResult<u32> {
    // The ceiling is computed in i32 and converted once, at the u32
    // boundary; embedded versions are non-negative, so the conversion
    // errors only if corruption ever reached the embedded set. The
    // `unwrap_or(0)` is the empty-set default (no embedded migrations),
    // not a refused conversion.
    let max_version = migrations::runner()
        .get_migrations()
        .iter()
        .map(|migration| migration.version())
        .max()
        .unwrap_or(0);
    version_u32(max_version)
}

/// Where a store's schema sits relative to this binary's embedded
/// migrations, as read from the file without touching it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaStatus {
    /// Highest version recorded in the store's applied-migration history.
    pub current: u32,
    /// Highest version embedded in this binary.
    pub target: u32,
}

impl SchemaStatus {
    /// True when the store still needs at least one embedded migration.
    #[must_use]
    pub fn has_pending_migrations(&self) -> bool {
        self.current < self.target
    }
}

/// Read the store's schema position without migrating the database.
///
/// Opens `<data_dir>/db/memory.sqlite` read-only (never with CREATE, so an
/// absent file stays absent), reads the highest applied version from
/// refinery's history table, and compares it with the highest embedded
/// version. The database file itself is not written. On a WAL store SQLite
/// may still create or touch its `-wal`/`-shm` companions and take its usual
/// read locks, as any reader does.
///
/// `Ok(None)` when the database file is absent or carries no applied
/// migration history yet (a fresh install has nothing to report); `Err` on
/// I/O or SQLite failure. A boot caller treats `Err` as "no report": log it
/// and keep starting, never block startup on this read.
pub fn schema_status(data_dir: &Path) -> StoreResult<Option<SchemaStatus>> {
    let db_path = data_dir.join("db").join(crate::DB_FILENAME);
    let metadata = match std::fs::metadata(&db_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() {
        return Ok(None);
    }
    // Read-only and without CREATE: this report must not manufacture or
    // write the database it reads.
    let conn = Connection::open_with_flags(&db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let has_history: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type = 'table' AND name = 'refinery_schema_history')",
        [],
        |row| row.get(0),
    )?;
    if !has_history {
        return Ok(None);
    }
    let current: u32 = conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM refinery_schema_history",
        [],
        |row| row.get(0),
    )?;
    // No applied migration: a blank or first-boot database is a fresh
    // install, not a store that needs a pre-migration backup warning.
    if current == 0 {
        return Ok(None);
    }
    Ok(Some(SchemaStatus {
        current,
        target: max_supported_version()?,
    }))
}

/// Translate refinery's two `MissingVersion` cases without confusing a
/// skipped migration with a downgrade. Every other failure passes through.
fn classify_run_error(err: refinery::Error) -> StoreError {
    if let refinery::error::Kind::MissingVersion(migration) = err.kind() {
        let identity = format!("V{} ({})", migration.version(), migration.name());
        // refinery returns the embedded (unapplied) migration when an older
        // version was skipped, and the history row when a build lacks it.
        if migration.applied_on().is_none() {
            return StoreError::MigrationHistoryGap { missing: identity };
        }
        let supported = match max_supported_version() {
            Ok(supported) => supported,
            Err(error) => return error,
        };
        // Exact i32 comparison: `supported` is the embedded ceiling (small,
        // non-negative, so `as i32` is lossless); `migration.version()`
        // comes from the store history and may be negative or corrupt — a
        // negative stays a rejection, never a version read as ahead of the
        // ceiling.
        if migration.version() > supported as i32 {
            return StoreError::DataSchemaAhead {
                applied: identity,
                supported,
            };
        }
    }
    StoreError::Migration(err)
}

#[cfg(test)]
pub(crate) fn run_to(conn: &mut rusqlite::Connection, target: u32) -> StoreResult<()> {
    // refinery 0.10 takes a signed `SchemaVersion`. A target that does not
    // fit in `i32` is refused before the runner is invoked (`refinery::Error`
    // has no public constructor): saturating it to `i32::MAX` would silently
    // mean "run every embedded migration".
    let version = i32::try_from(target).map_err(|_| {
        StoreError::InvalidState(format!(
            "migration target {target} does not fit in refinery's signed version"
        ))
    })?;
    migrations::runner()
        .set_target(refinery::Target::Version(version))
        .run(conn)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_core::{
        AgentKind, HandoffId, NewObservation, NewSession, ObservationKind, SessionId,
    };
    use rusqlite::{Connection, params};

    #[test]
    fn v94_preserves_old_response_rows_and_cascades_context() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run_to(&mut conn, 93).unwrap();
        conn.execute_batch("INSERT INTO workspaces(id,name,created_at) VALUES(zeroblob(16),'usage',1);
            INSERT INTO projects(id,workspace_id,name,created_at) VALUES(zeroblob(16),zeroblob(16),'usage',1);
            INSERT INTO sessions(id,workspace_id,project_id,agent_kind,started_at) VALUES(zeroblob(16),zeroblob(16),zeroblob(16),'codex',1);
            INSERT INTO token_calls(session_id,call_id,input_tokens,output_tokens,cache_write_tokens,cache_read_tokens,observed_at) VALUES(zeroblob(16),'old-response',70,7,0,0,123);").unwrap();
        run(&mut conn).unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let preserved:(u64,u64,i64,Option<String>,Option<String>,Option<i64>)=conn.query_row("SELECT input_tokens,output_tokens,observed_at,turn_id,native_response_id,response_at FROM token_calls",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).unwrap();
        assert_eq!(preserved, (70, 7, 123, None, None, None));
        conn.execute_batch("INSERT INTO token_call_tools(session_id,call_id,tool_id,tool_name,relation) VALUES(zeroblob(16),'old-response','tool-1','Read','requested');
            DELETE FROM token_calls WHERE call_id='old-response';").unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM token_call_tools", [], |r| r
                .get::<_, u64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn v92_normalizes_codex_cache_once_and_preserves_claude() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run_to(&mut conn, 91).unwrap();
        conn.execute_batch("INSERT INTO workspaces(id,name,created_at) VALUES(zeroblob(16),'usage',1);
            INSERT INTO projects(id,workspace_id,name,created_at) VALUES(zeroblob(16),zeroblob(16),'usage',1);
            INSERT INTO sessions(id,workspace_id,project_id,agent_kind,started_at) VALUES(zeroblob(16),zeroblob(16),zeroblob(16),'codex',1);
            INSERT INTO sessions(id,workspace_id,project_id,agent_kind,started_at) VALUES(x'00000000000000000000000000000001',zeroblob(16),zeroblob(16),'claude-code',1);
            INSERT INTO session_usage(session_id,input_tokens,cache_read_tokens,updated_at) SELECT id,1000,800,1 FROM sessions;").unwrap();
        run(&mut conn).unwrap();
        run(&mut conn).unwrap();
        let codex: u64 = conn
            .query_row(
                "SELECT input_tokens FROM session_usage WHERE session_id=zeroblob(16)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let claude:u64=conn.query_row("SELECT input_tokens FROM session_usage WHERE session_id=x'00000000000000000000000000000001'",[],|r|r.get(0)).unwrap();
        assert_eq!(codex, 200);
        assert_eq!(claude, 1000);
    }

    /// V95 rebuilds `agent_messages` so a message can be held in `editing`.
    /// Existing mail keeps its text, state, order and addressee, the three
    /// indexes come back, and `editing` is the only value the constraint gained.
    #[test]
    fn v95_accepts_the_editing_state_and_keeps_existing_messages() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run_to(&mut conn, 94).unwrap();
        conn.execute_batch(
            "INSERT INTO workspaces(id,name,created_at) VALUES(zeroblob(16),'mail',1);
             INSERT INTO projects(id,workspace_id,name,created_at) VALUES(zeroblob(16),zeroblob(16),'mail',1);
             INSERT INTO agent_messages(id,from_workspace_id,from_project_id,from_agent,to_workspace_id,to_project_id,subject,body,state,created_at,to_agent,from_owner_user)
                 VALUES(x'01000000000000000000000000000000',zeroblob(16),zeroblob(16),'other',zeroblob(16),zeroblob(16),'s','queued','pending',5,'codex','user:sender');
             INSERT INTO agent_messages(id,from_workspace_id,from_project_id,from_agent,to_workspace_id,to_project_id,body,state,created_at,claimed_at,claimed_by_agent)
                 VALUES(x'02000000000000000000000000000000',zeroblob(16),zeroblob(16),'other',zeroblob(16),zeroblob(16),'done','claimed',6,7,'codex');",
        )
        .unwrap();
        run(&mut conn).unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let rows: Vec<(String, String, i64, Option<String>)> = conn
            .prepare(
                "SELECT state, body, created_at, to_agent FROM agent_messages ORDER BY created_at",
            )
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            rows,
            vec![
                (
                    "pending".to_owned(),
                    "queued".to_owned(),
                    5,
                    Some("codex".to_owned())
                ),
                ("claimed".to_owned(), "done".to_owned(), 6, None),
            ]
        );
        conn.execute(
            "UPDATE agent_messages SET state='editing' WHERE body='queued'",
            [],
        )
        .unwrap();
        assert!(
            conn.execute(
                "UPDATE agent_messages SET state='held' WHERE body='queued'",
                [],
            )
            .is_err(),
            "the state constraint must still reject unknown values"
        );
        let indexes: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='agent_messages' \
                 AND name IN ('idx_agent_messages_inbox','idx_agent_messages_outbox','idx_messages_agent_inbox')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(indexes, 3);
        let edited_at: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('agent_messages') WHERE name='edited_at'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(edited_at, 1);
        let provenance: (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT from_owner_user, subject FROM agent_messages WHERE body='queued'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            provenance,
            (Some("user:sender".to_owned()), Some("s".to_owned())),
            "the sender identity and subject survive the rebuild"
        );
        assert!(
            conn.execute(
                "UPDATE agent_messages SET from_session_id=zeroblob(16) WHERE body='queued'",
                [],
            )
            .is_err(),
            "the sender-session foreign key survives the rebuild"
        );
        conn.execute_batch("INSERT INTO sessions(id,workspace_id,project_id,agent_kind,started_at) VALUES(zeroblob(16),zeroblob(16),zeroblob(16),'codex',1);
            UPDATE agent_messages SET from_session_id=zeroblob(16) WHERE body='queued';").unwrap();
        let dangling: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_foreign_key_check('agent_messages')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(dangling, 0);
    }

    #[test]
    fn project_id_pinning_upgrade_preserves_runs_and_failed_prepare_rolls_back() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 95).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let ws = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project = crate::ops::get_or_create_project(&mut conn, &ws, "project", None).unwrap();
        let workstream = ai_memory_core::WorkstreamId::new();
        let run_id = ai_memory_core::ManagedRunId::new();
        conn.execute("INSERT INTO workstreams \
            (id, workspace_id, project_id, repo_fingerprint, worktree_fingerprint, name, cwd, created_at, selected_at, updated_at) \
            VALUES (?1, ?2, ?3, 'repo', 'tree', 'default', '/repo', 1, 1, 1)",
            params![workstream.as_bytes(), ws.as_bytes(), project.as_bytes()]).unwrap();
        conn.execute("INSERT INTO managed_runs \
            (id, workstream_id, agent_kind, lease_owner, state, sync_after, sync_through, context_delivered, lease_expires_at, started_at) \
            VALUES (?1, ?2, 'codex', 'host:1', 'active', 3, 7, 1, ?3, 1)",
            params![run_id.as_bytes(), workstream.as_bytes(), jiff::Timestamp::now().as_microsecond() + 60_000_000]).unwrap();
        run(&mut conn).unwrap();
        let snapshot = |conn: &Connection| {
            conn.query_row(
            "SELECT state, sync_after, sync_through, context_delivered, scope_by_id FROM managed_runs WHERE id = ?1",
            params![run_id.as_bytes()], |row| Ok((row.get::<_,String>(0)?, row.get::<_,i64>(1)?, row.get::<_,i64>(2)?, row.get::<_,bool>(3)?, row.get::<_,bool>(4)?))).unwrap()
        };
        assert_eq!(snapshot(&conn), ("active".into(), 3, 7, true, false));
        conn.execute_batch("CREATE TRIGGER reject_new_run BEFORE INSERT ON managed_runs BEGIN SELECT RAISE(ABORT, 'injected failure'); END;").unwrap();
        let input = crate::PrepareWorkstreamRun {
            workspace_id: ws,
            project_id: project,
            scope_by_id: true,
            repo_fingerprint: "repo".into(),
            worktree_fingerprint: "tree".into(),
            cwd: "/repo".into(),
            agent: AgentKind::Codex,
            automatic_harness: false,
            available_agents: Vec::new(),
            selection: crate::WorkstreamSelection::Named("default".into()),
            lease_owner: "host:2".into(),
        };
        assert!(crate::workstream::prepare_run(&mut conn, &input, None, true).is_err());
        assert_eq!(
            snapshot(&conn),
            ("active".into(), 3, 7, true, false),
            "failed prepare must roll back forced expiry and preserve the legacy flag"
        );
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM managed_runs", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
        conn.execute_batch("DROP TRIGGER reject_new_run;").unwrap();
        let prepared = crate::workstream::prepare_run(&mut conn, &input, None, true).unwrap();
        let pinned: bool = conn
            .query_row(
                "SELECT scope_by_id FROM managed_runs WHERE id = ?1",
                params![prepared.run_id.as_bytes()],
                |row| row.get(0),
            )
            .unwrap();
        assert!(pinned);
        assert_eq!(snapshot(&conn).0, "expired");
    }

    /// A store migrated by a newer build (an applied version above anything
    /// this binary embeds) must fail to open with the actionable
    /// `DataSchemaAhead` error, not refinery's raw "missing from the
    /// filesystem" wording.
    #[test]
    fn data_ahead_of_binary_reports_schema_ahead_not_raw_refinery() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("memory.sqlite");
        let mut conn = Connection::open(&db_path).unwrap();

        // Bring the store up to this binary's current schema.
        run(&mut conn).unwrap();

        // Simulate data written by a *newer* build: forge an applied migration
        // whose version sits above the embedded ceiling. refinery stores
        // `applied_on` as RFC3339 and `checksum` as a u64 string, and parses
        // both eagerly, so the row must be well-formed.
        let future = max_supported_version().unwrap() + 100;
        conn.execute(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum) \
             VALUES (?1, ?2, ?3, ?4)",
            params![future, "future_feature", "2026-07-14T00:00:00Z", "0"],
        )
        .unwrap();

        let err = run(&mut conn).unwrap_err();
        match err {
            StoreError::DataSchemaAhead { applied, supported } => {
                assert!(applied.contains(&format!("V{future}")), "applied={applied}");
                assert!(applied.contains("future_feature"), "applied={applied}");
                assert_eq!(supported, max_supported_version().unwrap());
            }
            other => panic!("expected DataSchemaAhead, got: {other:?}"),
        }
    }

    /// Version-boundary contract (refinery 0.10 signed versions): a `u32`
    /// target that does not fit the signed `SchemaVersion` and a negative
    /// stored version are refused explicitly, with the number in the error
    /// — never saturated into `i32::MAX` ("run everything") or into the
    /// "no version" baseline (0). A valid target still migrates.
    #[test]
    fn version_boundaries_are_refused_without_saturating_or_migrating() {
        let mut conn = Connection::open_in_memory().unwrap();

        // The smallest u32 outside i32 (i32::MAX + 1) and u32::MAX are both
        // refused before the refinery runner is invoked: no migration runs,
        // no history table is created, and the number is in the error.
        for target in [i32::MAX as u32 + 1, u32::MAX] {
            let err = run_to(&mut conn, target).unwrap_err();
            let StoreError::InvalidState(message) = &err else {
                panic!("expected InvalidState for target {target}, got: {err:?}");
            };
            assert!(
                message.contains(&target.to_string()),
                "target={target} message={message}"
            );
            assert_eq!(
                schema_object_count(&conn, "table", "refinery_schema_history"),
                0,
                "a refused target must not create migration history"
            );
        }

        // A negative version is corruption: an explicit error carrying the
        // number, not the baseline 0.
        let err = version_u32(-1).unwrap_err();
        let StoreError::InvalidState(message) = &err else {
            panic!("expected InvalidState for version -1, got: {err:?}");
        };
        assert!(message.contains("-1"), "message={message}");

        // Control: a valid target still migrates to exactly that version.
        run_to(&mut conn, 64).unwrap();
        assert_eq!(applied_versions(&conn).last(), Some(&64));
    }

    #[test]
    fn missing_earlier_migration_reports_a_history_gap_without_applying_anything() {
        let mut conn = Connection::open_in_memory().unwrap();
        // Between v2.3.2 and v2.4, main briefly contained V66 before it
        // received V65 from release/2.4 (#1612).
        let historical: Vec<_> = migrations::runner()
            .get_migrations()
            .iter()
            .filter(|migration| migration.version() <= 66 && migration.version() != 65)
            .cloned()
            .collect();
        refinery::Runner::new(&historical).run(&mut conn).unwrap();
        let before = applied_versions(&conn);
        assert_eq!(before.last(), Some(&66));
        assert!(!before.contains(&65));

        let err = run(&mut conn).unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains("migration history has a gap"),
            "{rendered}"
        );
        assert!(
            rendered.contains("V65 (page_compacted_marker)"),
            "{rendered}"
        );
        assert!(rendered.contains("not recorded"), "{rendered}");
        assert!(!rendered.contains("newer than"), "{rendered}");
        assert_eq!(applied_versions(&conn), before);
        let compacted_column: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('pages') WHERE name = 'compacted_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(compacted_column, 0, "a refused open must not apply V65");
    }

    #[test]
    fn an_ordinary_older_store_migrates_without_a_history_gap() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 64).unwrap();
        run(&mut conn).unwrap();
        let applied = applied_versions(&conn);
        assert!(applied.contains(&65) && applied.contains(&66));
        assert_eq!(
            applied.last(),
            Some(&i64::from(max_supported_version().unwrap()))
        );
    }

    /// The rendered message must drop refinery's misleading phrasing and carry
    /// the operator-facing explanation and remedy.
    #[test]
    fn schema_ahead_message_is_actionable() {
        let rendered = StoreError::DataSchemaAhead {
            applied: "V99 (future_feature)".to_string(),
            supported: 30,
        }
        .to_string();

        assert!(
            !rendered.contains("missing from the filesystem"),
            "must not leak refinery's raw wording: {rendered}"
        );
        assert!(
            rendered.contains("newer than this ai-memory build"),
            "{rendered}"
        );
        assert!(rendered.contains("V99 (future_feature)"), "{rendered}");
        assert!(rendered.contains("through V30"), "{rendered}");
    }

    /// A migration that fails partway must surface as a typed
    /// `StoreError::Migration`, must not be recorded in refinery's history
    /// (per-migration transaction rollback), and re-running after the
    /// precondition is fixed must converge to the full embedded schema.
    #[test]
    fn failed_migration_rolls_back_and_recovers_after_fix() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("memory.sqlite");
        let mut conn = Connection::open(&db_path).unwrap();

        // Migrate up to just before V31 (managed workstreams), then poison the
        // run by pre-creating a table V31 builds mid-script — after it has
        // already created `workstreams` and its pairing trigger.
        run_to(&mut conn, 30).unwrap();
        conn.execute("CREATE TABLE workstream_native_sessions (id INTEGER)", [])
            .unwrap();

        // (a) The failure surfaces as the typed migration error, not a panic
        // or a misclassified schema-ahead error.
        let err = run(&mut conn).unwrap_err();
        match &err {
            StoreError::Migration(_) => {}
            other => panic!("expected StoreError::Migration, got: {other:?}"),
        }

        // (c) No half-migrated state: refinery's history must stop at V30 —
        // the failed V31 is not recorded — and its earlier statements were
        // rolled back, leaving the poisoned table untouched.
        let applied = applied_versions(&conn);
        assert_eq!(
            applied.last(),
            Some(&30),
            "failed migration must not be recorded: {applied:?}"
        );
        assert!(
            !applied.contains(&31),
            "V31 must be absent from history: {applied:?}"
        );
        assert_eq!(schema_object_count(&conn, "table", "workstreams"), 0);
        assert_eq!(
            schema_object_count(&conn, "trigger", "workstreams_ws_proj_pairing_ai"),
            0,
            "statements before the failure must roll back with the migration"
        );
        let poisoned_cols: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('workstream_native_sessions')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            poisoned_cols, 1,
            "the pre-existing table must survive untouched"
        );

        // (b) Fix the precondition and re-run: the store converges to the
        // embedded ceiling with the real V31 schema in place.
        conn.execute("DROP TABLE workstream_native_sessions", [])
            .unwrap();
        run(&mut conn).unwrap();
        let applied = applied_versions(&conn);
        assert_eq!(
            applied.last(),
            Some(&i64::from(max_supported_version().unwrap())),
            "re-run must reach the embedded ceiling: {applied:?}"
        );
        assert!(applied.contains(&31) && applied.contains(&32));
        assert_eq!(schema_object_count(&conn, "table", "workstreams"), 1);
        assert_eq!(
            schema_object_count(&conn, "trigger", "workstreams_ws_proj_pairing_ai"),
            1
        );
    }

    fn applied_versions(conn: &Connection) -> Vec<i64> {
        let mut stmt = conn
            .prepare("SELECT version FROM refinery_schema_history ORDER BY version")
            .unwrap();
        stmt.query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<Vec<i64>, _>>()
            .unwrap()
    }

    fn schema_object_count(conn: &Connection, kind: &str, name: &str) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = ?1 AND name = ?2",
            params![kind, name],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn v28_to_v29_preserves_existing_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 28).unwrap();
        let workspace_id = [7_u8; 16];
        conn.execute(
            "INSERT INTO workspaces (id, name, created_at) VALUES (?1, 'existing', 1)",
            params![workspace_id.as_slice()],
        )
        .unwrap();

        run(&mut conn).unwrap();
        let name: String = conn
            .query_row(
                "SELECT name FROM workspaces WHERE id = ?1",
                params![workspace_id.as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(name, "existing");
        let state_table: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'maintenance_scheduler_state'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(state_table, 1);
    }

    #[test]
    fn v38_adds_scoped_entity_index_without_disturbing_existing_pages() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 37).unwrap();

        let ws1 = [1_u8; 16];
        let ws2 = [2_u8; 16];
        let proj1 = [3_u8; 16];
        let proj2 = [4_u8; 16];
        let page1 = [5_u8; 16];
        let page2 = [6_u8; 16];
        let hash = [0_u8; 32];
        for (workspace, name) in [(ws1, "one"), (ws2, "two")] {
            conn.execute(
                "INSERT INTO workspaces (id, name, created_at) VALUES (?1, ?2, 1)",
                params![workspace.as_slice(), name],
            )
            .unwrap();
        }
        for (project, workspace, name) in [(proj1, ws1, "project-one"), (proj2, ws2, "project-two")]
        {
            conn.execute(
                "INSERT INTO projects (id, workspace_id, name, created_at) \
                 VALUES (?1, ?2, ?3, 1)",
                params![project.as_slice(), workspace.as_slice(), name],
            )
            .unwrap();
        }
        for (page, workspace, project, path) in
            [(page1, ws1, proj1, "one.md"), (page2, ws2, proj2, "two.md")]
        {
            conn.execute(
                "INSERT INTO pages \
                 (id, workspace_id, project_id, path, title, tier, body, body_sha256, \
                  frontmatter_json, is_latest, pinned, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, 'title', 'semantic', 'body', ?5, '{}', 1, 0, 1, 1)",
                params![
                    page.as_slice(),
                    workspace.as_slice(),
                    project.as_slice(),
                    path,
                    hash.as_slice()
                ],
            )
            .unwrap();
        }

        run(&mut conn).unwrap();
        assert_eq!(schema_object_count(&conn, "table", "entities"), 1);
        assert_eq!(schema_object_count(&conn, "table", "entity_page_links"), 1);
        assert_eq!(
            schema_object_count(&conn, "trigger", "entities_ws_proj_pairing_ai"),
            1
        );
        assert_eq!(
            schema_object_count(&conn, "trigger", "entity_page_links_scope_pairing_ai"),
            1
        );
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM pages", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            2,
            "V38 must preserve existing pages"
        );

        let entity1 = [7_u8; 16];
        conn.execute(
            "INSERT INTO entities (id, workspace_id, project_id, name, created_at) \
             VALUES (?1, ?2, ?3, 'sqlite', 1)",
            params![entity1.as_slice(), ws1.as_slice(), proj1.as_slice()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO entity_page_links (entity_id, page_id) VALUES (?1, ?2)",
            params![entity1.as_slice(), page1.as_slice()],
        )
        .unwrap();

        let wrong_scope = conn
            .execute(
                "INSERT INTO entities (id, workspace_id, project_id, name, created_at) \
                 VALUES (?1, ?2, ?3, 'mismatch', 1)",
                params![[8_u8; 16].as_slice(), ws2.as_slice(), proj1.as_slice()],
            )
            .unwrap_err();
        assert!(
            wrong_scope
                .to_string()
                .contains("workspace/project mismatch")
        );
        let wrong_page = conn
            .execute(
                "INSERT INTO entity_page_links (entity_id, page_id) VALUES (?1, ?2)",
                params![entity1.as_slice(), page2.as_slice()],
            )
            .unwrap_err();
        assert!(
            wrong_page
                .to_string()
                .contains("entity/page scope mismatch")
        );
        assert!(
            conn.execute(
                "INSERT INTO entities (id, workspace_id, project_id, name, created_at) \
                 VALUES (?1, ?2, ?3, ?4, 1)",
                params![
                    [9_u8; 16].as_slice(),
                    ws1.as_slice(),
                    proj1.as_slice(),
                    "x".repeat(65)
                ],
            )
            .is_err(),
            "the schema must enforce the public entity length bound"
        );
    }

    #[test]
    fn v33_to_v35_preserves_queue_and_backfills_end_generation() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 33).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let session_id = SessionId::new();
        // Era-appropriate raw insert. This fixture deliberately stops at V33
        // and then migrates forward, so it must not go through `begin_session`:
        // that writes whatever columns the CURRENT schema has, and every later
        // migration that adds one would break a test about an older era.
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at) \
             VALUES (?1, ?2, ?3, ?4, NULL, ?5)",
            params![
                session_id.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
                AgentKind::Codex.as_str(),
                jiff::Timestamp::now().as_microsecond(),
            ],
        )
        .unwrap();
        crate::ops::insert_observation(
            &mut conn,
            &NewObservation {
                occurred_at: None,
                session_id,
                workspace_id,
                project_id,
                kind: ObservationKind::UserPrompt,
                extension: None,
                source_event: None,
                title: "prompt".into(),
                body: "continue".into(),
                importance: 8,
                file_path: None,
            },
        )
        .unwrap();
        conn.execute(
            "UPDATE sessions SET ended_at = 1 WHERE id = ?1",
            params![session_id.as_bytes()],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        assert_eq!(schema_object_count(&conn, "table", "sessions"), 1);
        assert_eq!(
            schema_object_count(&conn, "table", "session_consolidation_jobs"),
            1
        );
        assert_eq!(
            schema_object_count(
                &conn,
                "trigger",
                "session_consolidation_jobs_session_pairing_ai"
            ),
            1
        );
        let ended_observation_count: u64 = conn
            .query_row(
                "SELECT ended_observation_count FROM sessions WHERE id = ?1",
                params![session_id.as_bytes()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            ended_observation_count, 1,
            "V35 must baseline existing ended sessions without catch-up work"
        );
        assert!(
            crate::session_consolidation::enqueue(&mut conn, workspace_id, project_id, session_id,)
                .unwrap()
        );
    }

    #[test]
    fn v39_to_v41_preserve_existing_rows_as_shared_and_add_listing_indexes() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 38).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let session_id = SessionId::new();
        let handoff_id = HandoffId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at) \
             VALUES (?1, ?2, ?3, ?4, NULL, 1)",
            params![
                session_id.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
                AgentKind::Codex.as_str(),
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO handoffs \
             (id, workspace_id, project_id, from_agent, summary, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, 1)",
            params![
                handoff_id.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
                AgentKind::Codex.as_str(),
                "legacy baton",
            ],
        )
        .unwrap();

        run(&mut conn).unwrap();

        let actor_user: Option<String> = conn
            .query_row(
                "SELECT actor_user FROM sessions WHERE id = ?1",
                params![session_id.as_bytes()],
                |row| row.get(0),
            )
            .unwrap();
        let (owner_user, accepted_by_user): (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT owner_user, accepted_by_user FROM handoffs WHERE id = ?1",
                params![handoff_id.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(actor_user, None, "legacy sessions remain shared");
        assert_eq!(owner_user, None, "legacy handoffs remain shared");
        assert_eq!(accepted_by_user, None);
        for index in [
            "idx_handoffs_open_owner",
            "idx_sessions_open_owner",
            "idx_handoffs_project_recent",
            "idx_handoffs_project_owner_recent",
        ] {
            assert_eq!(
                schema_object_count(&conn, "index", index),
                1,
                "missing ownership/listing index {index}",
            );
        }
    }

    #[test]
    fn v43_to_v44_preserves_session_state_and_all_scope_guards() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 43).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'codex', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            ("codex".into(), "/repo".into(), 7, Some("user:alice".into()))
        );

        for index in [
            "idx_sessions_recent",
            "idx_sessions_project",
            "idx_sessions_started_at",
            "idx_sessions_scope_ended",
            "idx_sessions_open_owner",
        ] {
            assert_eq!(
                schema_object_count(&conn, "index", index),
                1,
                "V44 dropped session index {index}"
            );
        }
        for trigger in [
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            assert_eq!(
                schema_object_count(&conn, "trigger", trigger),
                1,
                "V44 dropped scope guard {trigger}"
            );
        }
        let foreign_key_errors: i64 = conn
            .query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(foreign_key_errors, 0);

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::Hermes,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();
    }

    #[test]
    fn v44_to_v45_preserves_session_state_and_accepts_kiro_cli() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 44).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'hermes', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            (
                "hermes".into(),
                "/repo".into(),
                7,
                Some("user:alice".into())
            )
        );

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::KiroCli,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    #[test]
    fn v46_to_v47_preserves_session_state_and_accepts_command_code() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 46).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'kiro-cli', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            (
                "kiro-cli".into(),
                "/repo".into(),
                7,
                Some("user:alice".into())
            )
        );

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::CommandCode,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    #[test]
    fn v49_to_v50_preserves_session_state_and_accepts_pool() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 49).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'command-code', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            (
                "command-code".into(),
                "/repo".into(),
                7,
                Some("user:alice".into())
            )
        );

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::Pool,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    #[test]
    fn v50_to_v51_preserves_session_state_and_accepts_zcode() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 50).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'pool', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            ("pool".into(), "/repo".into(), 7, Some("user:alice".into()))
        );

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::Zcode,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    #[test]
    fn v71_to_v72_preserves_session_state_and_accepts_grizzybot() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 71).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'zcode', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();
        // V64 mail points at sessions with ON DELETE SET NULL; the rebuild
        // must neither null the sender nor leave a dangling reference.
        let message_id = uuid::Uuid::now_v7();
        conn.execute(
            "INSERT INTO agent_messages \
             (id, from_workspace_id, from_project_id, from_agent, from_session_id, \
              to_workspace_id, to_project_id, body, created_at) \
             VALUES (?1, ?2, ?3, 'zcode', ?4, ?2, ?3, 'hello', 1)",
            params![
                message_id.as_bytes().as_slice(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
                existing.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let sender: Option<Vec<u8>> = conn
            .query_row(
                "SELECT from_session_id FROM agent_messages WHERE id = ?1",
                params![message_id.as_bytes().as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(sender.as_deref(), Some(existing.as_bytes().as_slice()));
        let dangling: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_foreign_key_check('agent_messages')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(dangling, 0, "agent_messages must still reference sessions");

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            ("zcode".into(), "/repo".into(), 7, Some("user:alice".into()))
        );

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::Grizzybot,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    #[test]
    fn v72_to_v73_preserves_grizzybot_sessions_and_accepts_copilot_cli() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 72).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'grizzybot', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            (
                "grizzybot".into(),
                "/repo".into(),
                7,
                Some("user:alice".into())
            )
        );

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::CopilotCli,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    #[test]
    fn v77_to_v78_preserves_sessions_and_their_ingest_keys_and_accepts_droid() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 77).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'copilot-cli', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();
        // V77 points ingest_keys at sessions; the table rebuild must leave
        // that reference intact.
        conn.execute(
            "INSERT INTO ingest_keys (project_id, key, seen_at, session_id, agent_kind) \
             VALUES (?1, 'k1', 1, ?2, 'copilot-cli')",
            params![project_id.as_bytes(), existing.as_bytes()],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            (
                "copilot-cli".into(),
                "/repo".into(),
                7,
                Some("user:alice".into())
            )
        );
        let dangling: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_foreign_key_check('ingest_keys')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(dangling, 0, "ingest_keys must still reference sessions");

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::Droid,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    /// V80 rebuilds `sessions` again, for Meta Muse Code: rows, the ingest
    /// keys and observations that reference them, and the scope-pairing
    /// triggers all survive, and a `muse` session is accepted afterwards.
    #[test]
    fn v79_to_v80_preserves_sessions_ingest_keys_and_observations_and_accepts_muse() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 79).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'droid', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();
        // V79 cannot store a Muse session yet.
        assert!(
            conn.execute(
                "INSERT INTO sessions \
                 (id, workspace_id, project_id, agent_kind, started_at) \
                 VALUES (?1, ?2, ?3, 'muse', 1)",
                params![
                    SessionId::new().as_bytes(),
                    workspace_id.as_bytes(),
                    project_id.as_bytes(),
                ],
            )
            .is_err(),
            "the V79 CHECK must reject muse"
        );
        conn.execute(
            "INSERT INTO ingest_keys (project_id, key, seen_at, session_id, agent_kind) \
             VALUES (?1, 'k1', 1, ?2, 'droid')",
            params![project_id.as_bytes(), existing.as_bytes()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO observations \
             (id, session_id, workspace_id, project_id, kind, title, body, created_at) \
             VALUES (?1, ?2, ?3, ?4, 'prompt', 'kept', 'kept body', 1)",
            params![
                uuid::Uuid::new_v4().as_bytes(),
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            ("droid".into(), "/repo".into(), 7, Some("user:alice".into()))
        );
        let observations: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM observations WHERE session_id = ?1",
                params![existing.as_bytes()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(observations, 1, "the session's observations survive");
        for table in ["ingest_keys", "observations"] {
            let dangling: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM pragma_foreign_key_check('{table}')"),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(dangling, 0, "{table} must still reference sessions");
        }

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::Muse,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    /// V81 rebuilds `sessions` again, for CodeBuddy Code: rows, the ingest
    /// keys and observations that reference them, and the scope-pairing
    /// triggers all survive, and a `codebuddy` session is accepted afterwards.
    #[test]
    fn v80_to_v81_preserves_sessions_ingest_keys_and_observations_and_accepts_codebuddy() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 80).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, ended_at, \
              ended_observation_count, actor_user) \
             VALUES (?1, ?2, ?3, 'muse', '/repo', 1, 2, 7, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();
        // V80 cannot store a CodeBuddy session yet.
        assert!(
            conn.execute(
                "INSERT INTO sessions \
                 (id, workspace_id, project_id, agent_kind, started_at) \
                 VALUES (?1, ?2, ?3, 'codebuddy', 1)",
                params![
                    SessionId::new().as_bytes(),
                    workspace_id.as_bytes(),
                    project_id.as_bytes(),
                ],
            )
            .is_err(),
            "the V80 CHECK must reject codebuddy"
        );
        conn.execute(
            "INSERT INTO ingest_keys (project_id, key, seen_at, session_id, agent_kind) \
             VALUES (?1, 'k1', 1, ?2, 'muse')",
            params![project_id.as_bytes(), existing.as_bytes()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO observations \
             (id, session_id, workspace_id, project_id, kind, title, body, created_at) \
             VALUES (?1, ?2, ?3, ?4, 'prompt', 'kept', 'kept body', 1)",
            params![
                uuid::Uuid::new_v4().as_bytes(),
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, i64, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, ended_observation_count, actor_user \
                 FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            ("muse".into(), "/repo".into(), 7, Some("user:alice".into()))
        );
        let observations: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM observations WHERE session_id = ?1",
                params![existing.as_bytes()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(observations, 1, "the session's observations survive");
        for table in ["ingest_keys", "observations"] {
            let dangling: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM pragma_foreign_key_check('{table}')"),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(dangling, 0, "{table} must still reference sessions");
        }

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::CodeBuddy,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    /// V83 expands the sessions enum for IBM Bob. Existing sessions and
    /// dependent observations survive the table rebuild, while the old V82
    /// constraint rejects the new value and the migrated schema accepts it.
    #[test]
    fn v82_to_v83_preserves_sessions_and_observations_and_accepts_ibm_bob() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 82).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, actor_user) \
             VALUES (?1, ?2, ?3, 'codebuddy', '/repo', 1, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();
        assert!(
            conn.execute(
                "INSERT INTO sessions \
                 (id, workspace_id, project_id, agent_kind, started_at) \
                 VALUES (?1, ?2, ?3, 'ibm-bob', 1)",
                params![
                    SessionId::new().as_bytes(),
                    workspace_id.as_bytes(),
                    project_id.as_bytes(),
                ],
            )
            .is_err(),
            "the V82 CHECK must reject ibm-bob"
        );
        conn.execute(
            "INSERT INTO observations \
             (id, session_id, workspace_id, project_id, kind, title, body, created_at) \
             VALUES (?1, ?2, ?3, ?4, 'prompt', 'kept', 'kept body', 1)",
            params![
                uuid::Uuid::new_v4().as_bytes(),
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, actor_user FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            (
                "codebuddy".into(),
                "/repo".into(),
                Some("user:alice".into())
            )
        );
        let observations: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM observations WHERE session_id = ?1",
                params![existing.as_bytes()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(observations, 1, "the session's observations survive");

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::IbmBob,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            0
        );
        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    /// V88 expands the sessions enum for Qwen Code. Existing sessions and
    /// dependent observations survive the table rebuild, while the old V87
    /// constraint rejects the new value and the migrated schema accepts it.
    #[test]
    fn v87_to_v88_preserves_sessions_and_observations_and_accepts_qwen_code() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 87).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, actor_user) \
             VALUES (?1, ?2, ?3, 'ibm-bob', '/repo', 1, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();
        assert!(
            conn.execute(
                "INSERT INTO sessions \
                 (id, workspace_id, project_id, agent_kind, started_at) \
                 VALUES (?1, ?2, ?3, 'qwen-code', 1)",
                params![
                    SessionId::new().as_bytes(),
                    workspace_id.as_bytes(),
                    project_id.as_bytes(),
                ],
            )
            .is_err(),
            "the V87 CHECK must reject qwen-code"
        );
        conn.execute(
            "INSERT INTO observations \
             (id, session_id, workspace_id, project_id, kind, title, body, created_at) \
             VALUES (?1, ?2, ?3, ?4, 'prompt', 'kept', 'kept body', 1)",
            params![
                uuid::Uuid::new_v4().as_bytes(),
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, actor_user FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            ("ibm-bob".into(), "/repo".into(), Some("user:alice".into()))
        );
        let observations: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM observations WHERE session_id = ?1",
                params![existing.as_bytes()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(observations, 1, "the session's observations survive");

        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::QwenCode,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            0
        );
        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    /// V89 expands the persisted session kind without losing the session
    /// graph rebuilt by V88.
    #[test]
    fn v88_to_v89_preserves_sessions_and_accepts_qoder() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 88).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        let existing = SessionId::new();
        conn.execute(
            "INSERT INTO sessions \
             (id, workspace_id, project_id, agent_kind, cwd, started_at, actor_user) \
             VALUES (?1, ?2, ?3, 'qwen-code', '/repo', 1, 'user:alice')",
            params![
                existing.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();
        assert!(
            conn.execute(
                "INSERT INTO sessions \
                 (id, workspace_id, project_id, agent_kind, started_at) \
                 VALUES (?1, ?2, ?3, 'qoder', 1)",
                params![
                    SessionId::new().as_bytes(),
                    workspace_id.as_bytes(),
                    project_id.as_bytes(),
                ],
            )
            .is_err(),
            "the V88 CHECK must reject qoder"
        );

        conn.pragma_update(None, "foreign_keys", "OFF").unwrap();
        run(&mut conn).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();

        let preserved: (String, String, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, cwd, actor_user FROM sessions WHERE id = ?1",
                params![existing.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            (
                "qwen-code".into(),
                "/repo".into(),
                Some("user:alice".into())
            )
        );
        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: SessionId::new(),
                workspace_id,
                project_id,
                agent_kind: AgentKind::Qoder,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            0
        );
        for name in [
            "idx_sessions_open_owner",
            "sessions_ws_proj_pairing_ai",
            "auto_improve_scheduler_claims_session_pairing_ai",
            "session_consolidation_jobs_session_pairing_ai",
        ] {
            let kind = if name.starts_with("idx_") {
                "index"
            } else {
                "trigger"
            };
            assert_eq!(schema_object_count(&conn, kind, name), 1, "missing {name}");
        }
    }

    #[test]
    fn v88_to_v89_preserves_existing_sessions_and_keeps_missing_usage_unknown() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 88).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let before = applied_versions(&conn);
        let workspace = crate::ops::get_or_create_workspace(&mut conn, "upgrade").unwrap();
        let project =
            crate::ops::get_or_create_project(&mut conn, &workspace, "app", None).unwrap();
        let session = SessionId::new();
        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: session,
                workspace_id: workspace,
                project_id: project,
                agent_kind: AgentKind::Codex,
                cwd: Some("/repo".into()),
                actor_user: Some("user:alice".into()),
            },
        )
        .unwrap();

        run_to(&mut conn, 89).unwrap();
        let mut expected = before;
        expected.push(89);
        assert_eq!(applied_versions(&conn), expected);
        run(&mut conn).unwrap();
        let preserved: (String, String, String) = conn
            .query_row(
                "SELECT agent_kind, cwd, actor_user FROM sessions WHERE id = ?1",
                params![session.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            preserved,
            ("codex".into(), "/repo".into(), "user:alice".into())
        );
        let missing: i64 = conn
            .query_row("SELECT COUNT(*) FROM session_usage", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            missing, 0,
            "upgrading must not invent usage for existing sessions"
        );

        let mut report = ai_memory_core::NewSessionUsage {
            session_id: SessionId::new(),
            input_tokens: 12,
            output_tokens: 4,
            cache_write_tokens: 0,
            cache_read_tokens: 2,
            model: Some("fixture-model".into()),
        };
        assert!(crate::ops::upsert_session_usage(&conn, &report).is_err());
        report.session_id = session;
        crate::ops::upsert_session_usage(&conn, &report).unwrap();
        conn.execute(
            "DELETE FROM sessions WHERE id = ?1",
            params![session.as_bytes()],
        )
        .unwrap();
        let remaining: i64 = conn
            .query_row("SELECT COUNT(*) FROM session_usage", [], |row| row.get(0))
            .unwrap();
        assert_eq!(remaining, 0, "usage must not outlive its session");
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            0
        );
    }

    #[test]
    fn v81_to_v82_preserves_managed_runs_and_adds_previous_session_identity() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 81).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project =
            crate::ops::get_or_create_project(&mut conn, &workspace, "project", None).unwrap();
        let workstream = ai_memory_core::WorkstreamId::new();
        let run_id = ai_memory_core::ManagedRunId::new();
        conn.execute(
            "INSERT INTO workstreams \
             (id, workspace_id, project_id, repo_fingerprint, worktree_fingerprint, name, cwd, \
              created_at, selected_at, updated_at) \
             VALUES (?1, ?2, ?3, 'repo', 'tree', 'default', '/repo', 1, 1, 1)",
            params![
                workstream.as_bytes(),
                workspace.as_bytes(),
                project.as_bytes(),
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO managed_runs \
             (id, workstream_id, agent_kind, lease_owner, state, sync_after, sync_through, \
              context_delivered, lease_expires_at, started_at) \
             VALUES (?1, ?2, 'codex', 'host:1', 'finished', 3, 7, 1, 10, 1)",
            params![run_id.as_bytes(), workstream.as_bytes()],
        )
        .unwrap();

        run(&mut conn).unwrap();

        let preserved: (String, i64, i64, Option<String>, Option<String>) = conn
            .query_row(
                "SELECT agent_kind, sync_after, sync_through, previous_agent_kind, \
                        previous_native_session_id \
                 FROM managed_runs WHERE id = ?1",
                params![run_id.as_bytes()],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(preserved, ("codex".into(), 3, 7, None, None));
        conn.execute(
            "UPDATE managed_runs SET previous_agent_kind = 'claude-code', \
             previous_native_session_id = 'native-1' WHERE id = ?1",
            params![run_id.as_bytes()],
        )
        .unwrap();
        let previous: (String, String) = conn
            .query_row(
                "SELECT previous_agent_kind, previous_native_session_id \
                 FROM managed_runs WHERE id = ?1",
                params![run_id.as_bytes()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(previous, ("claude-code".into(), "native-1".into()));
    }

    /// The repair constant comes from the released SQL, not a remembered
    /// value that could silently drift from its provenance.
    #[test]
    fn original_v62_checksum_matches_the_released_sql() {
        let released_sql = include_str!("testdata/V62__page_ingestion_windows.pre776.sql");
        let checksum = checksum_for_sql(V62_VERSION, V62_NAME, released_sql);

        assert_eq!(checksum, ORIGINAL_V62_CHECKSUM);
        let embedded = migrations::runner()
            .get_migrations()
            .iter()
            .find(|migration| migration.version() == V62_VERSION as i32)
            .expect("V62 must remain embedded")
            .checksum();
        assert_ne!(
            embedded, checksum,
            "the V62 repair is unnecessary if the embedded SQL matches v2.2.1"
        );
    }

    /// The reporter's three samples pin the exact Windows-native failure: the
    /// checksums are the current embedded SQL with only LF changed to CRLF.
    #[test]
    fn windows_release_checksums_match_crlf_embedded_sql() {
        let runner = migrations::runner();
        for (version, reported) in [
            (1, 17_285_868_314_470_184_417),
            (62, 3_297_905_123_477_343_401),
            (83, 11_621_067_245_702_300_616),
        ] {
            let migration = runner
                .get_migrations()
                .iter()
                .find(|migration| migration.version() == version)
                .unwrap();
            assert_eq!(
                crlf_checksum(migration).unwrap(),
                Some(reported),
                "V{version}"
            );
        }
    }

    /// Every known byte-equivalent checksum is repaired before the strict
    /// runner sees it. An arbitrary mismatch still fails closed.
    #[test]
    fn only_known_equivalent_migration_checksums_are_repaired() {
        let mut conn = Connection::open_in_memory().unwrap();
        run(&mut conn).unwrap();
        let runner = migrations::runner();
        let v62 = runner
            .get_migrations()
            .iter()
            .find(|migration| migration.version() == V62_VERSION as i32)
            .unwrap();
        let current_checksum = v62.checksum();
        assert_ne!(current_checksum, ORIGINAL_V62_CHECKSUM);

        for version in [1, 62, 83] {
            let migration = runner
                .get_migrations()
                .iter()
                .find(|migration| migration.version() == version)
                .unwrap();
            let rewritten = conn
                .execute(
                    "UPDATE refinery_schema_history SET checksum = ?1 WHERE version = ?2",
                    params![
                        crlf_checksum(migration).unwrap().unwrap().to_string(),
                        version
                    ],
                )
                .unwrap();
            assert_eq!(rewritten, 1, "V{version} history row must exist");
        }

        // A strict runner (refinery's default) rejects the divergence.
        let strict = migrations::runner().run(&mut conn);
        match strict {
            Err(err) => assert!(
                matches!(err.kind(), refinery::error::Kind::DivergentVersion(_, _)),
                "expected DivergentVersion, got: {err:?}"
            ),
            Ok(_) => panic!("strict runner should reject a divergent checksum"),
        }

        run(&mut conn).expect("the Windows-native checksums must be repaired");
        for version in [1, 62, 83] {
            let expected = runner
                .get_migrations()
                .iter()
                .find(|migration| migration.version() == version)
                .unwrap()
                .checksum()
                .to_string();
            let repaired: String = conn
                .query_row(
                    "SELECT checksum FROM refinery_schema_history WHERE version = ?1",
                    [version],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(repaired, expected, "V{version}");
        }

        conn.execute(
            "UPDATE refinery_schema_history SET checksum = ?1 WHERE version = ?2",
            params![ORIGINAL_V62_CHECKSUM.to_string(), V62_VERSION],
        )
        .unwrap();
        run(&mut conn).expect("the released V62 checksum must be repaired");
        let repaired: String = conn
            .query_row(
                "SELECT checksum FROM refinery_schema_history WHERE version = ?1",
                [V62_VERSION],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(repaired, current_checksum.to_string());

        let original_crlf = checksum_for_sql(
            V62_VERSION,
            V62_NAME,
            &include_str!("testdata/V62__page_ingestion_windows.pre776.sql").replace('\n', "\r\n"),
        );
        conn.execute(
            "UPDATE refinery_schema_history SET checksum = ?1 WHERE version = ?2",
            params![original_crlf.to_string(), V62_VERSION],
        )
        .unwrap();
        run(&mut conn).expect("the Windows-native original V62 checksum must be repaired");

        conn.execute(
            "UPDATE refinery_schema_history SET checksum = '0' WHERE version = ?1",
            [V62_VERSION],
        )
        .unwrap();
        match run(&mut conn).unwrap_err() {
            StoreError::Migration(err) => assert!(matches!(
                err.kind(),
                refinery::error::Kind::DivergentVersion(_, _)
            )),
            other => panic!("unknown divergence must remain fatal: {other:?}"),
        }

        conn.execute(
            "UPDATE refinery_schema_history SET checksum = ?1 WHERE version = ?2",
            params![current_checksum.to_string(), V62_VERSION],
        )
        .unwrap();
        let summary = crate::ops::backfill_page_windows(&mut conn).unwrap();
        assert_eq!(summary, crate::ops::PageWindowBackfillSummary::default());
    }

    #[test]
    fn v48_to_v49_replaces_the_decay_tombstone_index_predicate() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 48).unwrap();

        let before: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_pages_evicted'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(before.contains("supersedes IS NULL"), "{before}");

        run(&mut conn).unwrap();
        let after: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_pages_evicted'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            after.contains("workspace_id, project_id, superseded_at"),
            "{after}"
        );
        assert!(after.contains("superseded_at IS NOT NULL"), "{after}");
        assert!(!after.contains("supersedes IS NULL"), "{after}");
    }

    #[test]
    fn project_coordinate_keys_upgrade_and_backfill_without_changing_project_ids() {
        let coordinate_version = migrations::runner()
            .get_migrations()
            .iter()
            .find(|migration| migration.name() == "project_coordinate_keys")
            .map(|migration| version_u32(migration.version()))
            .unwrap()
            .unwrap();
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, coordinate_version - 1).unwrap();
        let workspace_id = [8_u8; 16];
        let project_id = [9_u8; 16];
        conn.execute(
            "INSERT INTO workspaces (id, name, created_at) VALUES (?1, 'acme', 1)",
            params![workspace_id.as_slice()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO projects (id, workspace_id, name, created_at, identity, identity_source) \
             VALUES (?1, ?2, 'api', 1, 'github.com/acme/group/api', 'git_remote')",
            params![project_id.as_slice(), workspace_id.as_slice()],
        )
        .unwrap();

        run(&mut conn).unwrap();
        assert_eq!(crate::project_coordinates::backfill(&mut conn).unwrap(), 1);
        let row: (Vec<u8>, String, String, String) = conn
            .query_row(
                "SELECT id, name, canonical_name, legacy_name FROM projects",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(row.0, project_id);
        assert_eq!(row.1, "api");
        assert_eq!(row.2, "acme-group-api");
        assert_eq!(row.3, "api");
        assert_eq!(crate::project_coordinates::backfill(&mut conn).unwrap(), 0);
    }

    /// Upstream's `UNIQUE (workspace_id, name)` is case-sensitive, so one
    /// workspace may hold both `API` and `api`. The identity index is on
    /// case-folded values, and backfilling `lower(name)` into it failed the
    /// whole upgrade on such an install. Looked up by name, not number: this
    /// migration is renumbered on every upstream sync that adds one of its own.
    #[test]
    fn project_identity_upgrades_a_workspace_with_names_differing_only_in_case() {
        let identity_version = migrations::runner()
            .get_migrations()
            .iter()
            .find(|m| m.name() == "project_identity")
            .map(|migration| version_u32(migration.version()))
            .expect("the project_identity migration is embedded")
            .unwrap();

        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, identity_version - 1).unwrap();
        let workspace_id = [7_u8; 16];
        conn.execute(
            "INSERT INTO workspaces (id, name, created_at) VALUES (?1, 'acme', 1)",
            params![workspace_id.as_slice()],
        )
        .unwrap();
        for (id, name) in [([1_u8; 16], "API"), ([2_u8; 16], "api")] {
            conn.execute(
                "INSERT INTO projects (id, workspace_id, name, created_at) VALUES (?1, ?2, ?3, 1)",
                params![id.as_slice(), workspace_id.as_slice(), name],
            )
            .unwrap();
        }

        run(&mut conn).expect("names differing only in case must not fail the upgrade");

        // Both projects survive, and neither is claimed: which one a future
        // `api/` checkout resolves to is not the migration's decision.
        let unclaimed: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM projects WHERE identity = '' AND identity_source = ''",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(unclaimed, 2);

        // The index still does its job once the resolver claims an identity.
        conn.execute(
            "UPDATE projects SET identity = 'api', identity_source = 'folder_name' WHERE name = 'api'",
            [],
        )
        .unwrap();
        let duplicate = conn.execute(
            "UPDATE projects SET identity = 'api', identity_source = 'folder_name' WHERE name = 'API'",
            [],
        );
        assert!(
            duplicate.is_err(),
            "a claimed identity must stay unique per workspace"
        );
    }

    /// The fork table (introduced as V68, then V72, V81 and V84, now outside
    /// the numbering) is created idempotently: a store that ran an earlier
    /// numbering already has the `consolidation_chunk_progress` table and
    /// index, so `run()` must execute the DDL over the existing objects
    /// without failing on `table already exists`, and keep their rows.
    #[test]
    fn fork_table_ddl_is_idempotent_over_a_store_that_already_has_the_table() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 83).unwrap();
        conn.execute_batch(FORK_CHUNK_PROGRESS_DDL).unwrap();
        insert_chunk_checkpoint(&conn);

        run(&mut conn).expect("the DDL must apply idempotently over an existing table");

        assert_fork_table_and_index(&conn);
        assert_eq!(chunk_checkpoint_count(&conn), 1, "the row must survive");
        assert_eq!(
            applied_versions(&conn).last(),
            Some(&i64::from(max_supported_version().unwrap()))
        );
    }

    /// A fresh store gets the fork table from `run()` even though no
    /// embedded migration creates it.
    #[test]
    fn a_fresh_store_gets_the_fork_table_after_the_runner() {
        let mut conn = Connection::open_in_memory().unwrap();
        run(&mut conn).unwrap();
        assert_fork_table_and_index(&conn);
    }

    /// Regression guard: the fork table must never come back as a numbered
    /// migration, where it competes with upstream for a version.
    #[test]
    fn no_embedded_migration_is_the_fork_chunk_progress_table() {
        let runner = migrations::runner();
        let numbered: Vec<u32> = runner
            .get_migrations()
            .iter()
            .filter(|m| m.name() == FORK_CHUNK_PROGRESS_NAME)
            .map(|migration| version_u32(migration.version()).unwrap())
            .collect();
        assert!(numbered.is_empty(), "numbered as {numbered:?}");
    }

    /// A store that ran the fork's V84 build carries the history row
    /// `84 consolidation_chunk_progress`, a number upstream 3.2.0 gave to
    /// `api_credential_pepper_verifier`. `run()` forgets that row, so the
    /// upstream V84 runs and creates `api_credential_pepper_state`, and the
    /// fork table and its checkpoint stay.
    #[test]
    fn a_store_with_the_fork_v84_history_row_forgets_it_and_keeps_the_table() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 83).unwrap();
        conn.execute_batch(FORK_CHUNK_PROGRESS_DDL).unwrap();
        insert_history_row(&conn, 84, FORK_CHUNK_PROGRESS_NAME);
        insert_chunk_checkpoint(&conn);

        run(&mut conn).expect("a store carrying the fork's V84 row must open");

        assert_eq!(
            history_row(&conn, 84),
            embedded_name_and_checksum(84),
            "the fork's V84 row must give way to the upstream V84"
        );
        assert_eq!(
            applied_versions(&conn),
            (1..=i64::from(max_supported_version().unwrap())).collect::<Vec<i64>>()
        );
        assert_eq!(
            schema_object_count(&conn, "table", "api_credential_pepper_state"),
            1
        );
        assert_fork_table_and_index(&conn);
        assert_eq!(chunk_checkpoint_count(&conn), 1, "the checkpoint must stay");
    }

    /// Only `84 consolidation_chunk_progress` is forgotten. A V84 row with any
    /// other name (case variants included), or the fork's name at another
    /// version, stays in history and the runner refuses it as a divergence
    /// from the embedded migration of that version; the legitimate control is
    /// the test above.
    #[test]
    fn only_the_fork_v84_name_is_forgotten() {
        for (version, name) in [
            (84, "something_else"),
            (84, "Consolidation_Chunk_Progress"),
            (84, "consolidation_chunk_progress "),
            (85, FORK_CHUNK_PROGRESS_NAME),
        ] {
            let mut conn = Connection::open_in_memory().unwrap();
            run_to(&mut conn, 83).unwrap();
            insert_history_row(&conn, version, name);

            assert_divergent(run(&mut conn), &format!("V{version} {name:?}"));
            assert_eq!(
                history_row(&conn, version),
                (name.to_string(), "0".to_string()),
                "V{version} {name:?} must not be deleted"
            );
        }
    }

    fn insert_history_row(conn: &Connection, version: u32, name: &str) {
        conn.execute(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum) \
             VALUES (?1, ?2, '2026-10-20T00:00:00Z', '0')",
            params![version, name],
        )
        .unwrap();
    }

    fn insert_chunk_checkpoint(conn: &Connection) {
        conn.execute(
            "INSERT INTO consolidation_chunk_progress \
             (workspace_id, project_id, session_id, chunk_fingerprint, extraction_json, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![[1u8; 16].as_slice(), [2u8; 16].as_slice(), [3u8; 16].as_slice(), "fp", "{}", 1],
        )
        .unwrap();
    }

    fn chunk_checkpoint_count(conn: &Connection) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM consolidation_chunk_progress",
            [],
            |r| r.get(0),
        )
        .unwrap()
    }

    fn assert_fork_table_and_index(conn: &Connection) {
        assert_eq!(
            schema_object_count(conn, "table", "consolidation_chunk_progress"),
            1
        );
        assert_eq!(
            schema_object_count(conn, "index", "idx_consolidation_chunk_progress_at"),
            1
        );
    }

    /// The production store of the fork recorded this table as history row
    /// `72 consolidation_chunk_progress`, the number upstream 3.0 gave to
    /// `sessions_grizzybot_agent_kind`. The fork history repair keeps
    /// that store opening: refinery treats V72 as applied and skips upstream's
    /// V72 body, and V73 rebuilds `sessions` with the complete CHECK list
    /// (grizzybot included), so nothing is lost. The fork table survives and
    /// every later embedded migration is applied.
    #[test]
    fn a_store_with_the_fork_v72_history_row_migrates_to_the_current_version() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 71).unwrap();

        conn.execute_batch(FORK_CHUNK_PROGRESS_DDL).unwrap();
        conn.execute(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum)              VALUES (72, 'consolidation_chunk_progress', '2026-09-28T00:00:00Z', '0')",
            [],
        )
        .unwrap();

        run(&mut conn).expect("a store carrying the fork's V72 row must still open");

        let versions = applied_versions(&conn);
        for version in 72..=i64::from(max_supported_version().unwrap()) {
            assert!(
                versions.contains(&version),
                "V{version} must be recorded: {versions:?}"
            );
        }
        assert_eq!(
            schema_object_count(&conn, "table", "consolidation_chunk_progress"),
            1
        );
        let sessions_sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'sessions'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            sessions_sql.contains("'grizzybot'"),
            "the skipped V72 must not leave sessions without grizzybot: {sessions_sql}"
        );
    }

    /// The fork's V81-era numbering recorded this table as history row
    /// `81 consolidation_chunk_progress`, the number upstream 3.1.1 gave to
    /// `sessions_codebuddy_agent_kind`. The fork history repair keeps
    /// that store opening: refinery treats V81 as applied and skips
    /// upstream's V81 body, and V83 rebuilds `sessions` with the complete
    /// CHECK list (codebuddy and ibm-bob included), so nothing is lost. The
    /// fork table survives and every later embedded migration is applied.
    #[test]
    fn a_store_with_the_fork_v81_history_row_migrates_to_the_current_version() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 80).unwrap();

        conn.execute_batch(FORK_CHUNK_PROGRESS_DDL).unwrap();
        conn.execute(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum)              VALUES (81, 'consolidation_chunk_progress', '2026-10-15T00:00:00Z', '0')",
            [],
        )
        .unwrap();

        run(&mut conn).expect("a store carrying the fork's V81 row must still open");

        let versions = applied_versions(&conn);
        for version in 81..=i64::from(max_supported_version().unwrap()) {
            assert!(
                versions.contains(&version),
                "V{version} must be recorded: {versions:?}"
            );
        }
        assert_eq!(
            schema_object_count(&conn, "table", "consolidation_chunk_progress"),
            1
        );
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "project", None).unwrap();
        for agent_kind in [AgentKind::CodeBuddy, AgentKind::IbmBob] {
            crate::ops::begin_session(
                &mut conn,
                &NewSession {
                    occurred_at: None,
                    id: SessionId::new(),
                    workspace_id,
                    project_id,
                    agent_kind,
                    cwd: Some("/repo".into()),
                    actor_user: Some("user:alice".into()),
                },
            )
            .unwrap_or_else(|e| panic!("{agent_kind:?} must be accepted: {e}"));
        }
    }

    fn history_row(conn: &Connection, version: u32) -> (String, String) {
        conn.query_row(
            "SELECT name, checksum FROM refinery_schema_history WHERE version = ?1",
            [version],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap()
    }

    fn embedded_name_and_checksum(version: u32) -> (String, String) {
        let runner = migrations::runner();
        let migration = runner
            .get_migrations()
            .iter()
            .find(|migration| migration.version() == version as i32)
            .expect("migration must be embedded");
        (
            migration.name().to_string(),
            migration.checksum().to_string(),
        )
    }

    fn assert_divergent(result: StoreResult<()>, case: &str) {
        match result {
            Err(StoreError::Migration(err)) => assert!(
                matches!(err.kind(), refinery::error::Kind::DivergentVersion(_, _)),
                "{case}: expected DivergentVersion, got: {err:?}"
            ),
            other => panic!("{case}: divergence must remain fatal: {other:?}"),
        }
    }

    /// The fork's production history: the chunk-progress table recorded as
    /// both `72` and `81` with the checksums the fork binaries wrote, nothing
    /// after V81, and the table already holding a row. The strict runner must
    /// open it up to the newest embedded migration, both rows end up
    /// describing the embedded migrations, and a second `run()` changes
    /// nothing.
    #[test]
    fn the_fork_production_history_with_v72_and_v81_rows_opens_under_the_strict_runner() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 80).unwrap();
        let rewritten = conn
            .execute(
                "UPDATE refinery_schema_history \
                 SET name = 'consolidation_chunk_progress', checksum = '2728718870490857001' \
                 WHERE version = 72",
                [],
            )
            .unwrap();
        assert_eq!(rewritten, 1);
        conn.execute_batch(FORK_CHUNK_PROGRESS_DDL).unwrap();
        conn.execute(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum) \
             VALUES (81, 'consolidation_chunk_progress', '2026-10-15T00:00:00Z', \
             '1780142640744897068')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO consolidation_chunk_progress \
             (workspace_id, project_id, session_id, chunk_fingerprint, extraction_json, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![[1u8; 16].as_slice(), [2u8; 16].as_slice(), [3u8; 16].as_slice(), "fp", "{}", 1],
        )
        .unwrap();

        run(&mut conn).expect("the fork production history must open");

        assert_eq!(
            applied_versions(&conn),
            (1..=i64::from(max_supported_version().unwrap())).collect::<Vec<i64>>()
        );
        assert_fork_table_and_index(&conn);
        for version in [72, 81] {
            assert_eq!(
                history_row(&conn, version),
                embedded_name_and_checksum(version),
                "V{version} must describe the embedded migration"
            );
        }
        let rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM consolidation_chunk_progress",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rows, 1, "the pre-existing row must survive");

        let history_before = history_snapshot(&conn);
        run(&mut conn).expect("a second run must be a no-op");
        assert_eq!(history_before, history_snapshot(&conn));
        assert_fork_table_and_index(&conn);
        assert_eq!(chunk_checkpoint_count(&conn), 1);
    }

    /// The fork's production history today: 1..=83 with the V72/V81 rows
    /// already repaired to upstream's name and checksum, and the fork table
    /// holding a checkpoint. It opens, applies every embedded migration after
    /// V83, ends with upstream's `api_credential_pepper_state` next to the
    /// fork table, index and checkpoint, and a second `run()` changes nothing.
    #[test]
    fn the_fork_production_history_repaired_through_v83_opens_on_the_current_binary() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 83).unwrap();
        conn.execute_batch(FORK_CHUNK_PROGRESS_DDL).unwrap();
        insert_chunk_checkpoint(&conn);
        for version in FORK_CHUNK_PROGRESS_VERSIONS {
            assert_eq!(
                history_row(&conn, version),
                embedded_name_and_checksum(version)
            );
        }

        run(&mut conn).expect("the repaired fork production history must open");

        assert_eq!(
            applied_versions(&conn),
            (1..=i64::from(max_supported_version().unwrap())).collect::<Vec<i64>>()
        );
        assert_eq!(
            schema_object_count(&conn, "table", "api_credential_pepper_state"),
            1
        );
        assert_fork_table_and_index(&conn);
        assert_eq!(chunk_checkpoint_count(&conn), 1, "the checkpoint must stay");

        let history_before = history_snapshot(&conn);
        run(&mut conn).expect("a second run must be a no-op");
        assert_eq!(history_before, history_snapshot(&conn));
        assert_fork_table_and_index(&conn);
        assert_eq!(chunk_checkpoint_count(&conn), 1);
    }

    fn history_snapshot(conn: &Connection) -> Vec<(i64, String, String, String)> {
        let mut stmt = conn
            .prepare(
                "SELECT version, name, applied_on, checksum \
                 FROM refinery_schema_history ORDER BY version",
            )
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    /// The fork repair matches only the fork's name at V72/V81. Another name,
    /// or the upstream name with a wrong checksum, still fails closed; the
    /// fork's name is the legitimate control. Each version is checked on its
    /// own, so a repair that ignores the name at one of them still fails.
    #[test]
    fn only_the_fork_chunk_progress_name_is_repaired() {
        for version in FORK_CHUNK_PROGRESS_VERSIONS {
            let mut conn = Connection::open_in_memory().unwrap();
            run_to(&mut conn, 80).unwrap();
            let (upstream_name, upstream_checksum) = embedded_name_and_checksum(version);
            if version == 81 {
                conn.execute(
                    "INSERT INTO refinery_schema_history (version, name, applied_on, checksum) \
                     VALUES (81, ?1, '2026-10-15T00:00:00Z', ?2)",
                    params![upstream_name, upstream_checksum],
                )
                .unwrap();
            }

            conn.execute(
                "UPDATE refinery_schema_history SET name = 'something_else' WHERE version = ?1",
                [version],
            )
            .unwrap();
            assert_divergent(run(&mut conn), &format!("another name at V{version}"));

            conn.execute(
                "UPDATE refinery_schema_history SET name = ?1, checksum = '0' WHERE version = ?2",
                params![upstream_name, version],
            )
            .unwrap();
            assert_divergent(
                run(&mut conn),
                &format!("upstream name with a wrong checksum at V{version}"),
            );

            conn.execute(
                "UPDATE refinery_schema_history \
                 SET name = 'consolidation_chunk_progress', checksum = '0' WHERE version = ?1",
                [version],
            )
            .unwrap();
            run(&mut conn)
                .unwrap_or_else(|e| panic!("the fork's V{version} row must be repaired: {e}"));
            assert_eq!(
                history_row(&conn, version),
                embedded_name_and_checksum(version)
            );
        }
    }

    #[test]
    fn v97_rebuilds_both_fts_indexes_so_a_directory_path_matches_its_basename() {
        use crate::fts_query::{FtsStopwords, prepare_fts5_query};

        const TABLES: [&str; 2] = ["pages_fts", "observations_fts"];

        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 96).unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "proj", None).unwrap();
        let session_id = SessionId::new();
        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: session_id,
                workspace_id,
                project_id,
                agent_kind: AgentKind::Codex,
                cwd: None,
                actor_user: None,
            },
        )
        .unwrap();
        let observation = |body: &str| NewObservation {
            occurred_at: None,
            session_id,
            workspace_id,
            project_id,
            kind: ObservationKind::PostToolUse,
            extension: None,
            source_event: None,
            title: "tool file".into(),
            body: body.into(),
            importance: 5,
            file_path: None,
        };
        let body = "edited crates/ai-memory-store/src/reader.rs in ai-memory";
        let page = [4_u8; 16];
        conn.execute(
            "INSERT INTO pages \
             (id, workspace_id, project_id, path, title, tier, body, body_sha256, \
              frontmatter_json, is_latest, pinned, created_at, updated_at) \
             VALUES (?1, ?2, ?3, 'notes/edit.md', 'title', 'semantic', ?4, zeroblob(32), '{}', \
                     1, 0, 1, 1)",
            params![
                page.as_slice(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
                body,
            ],
        )
        .unwrap();
        crate::ops::insert_observation(&mut conn, &observation(body)).unwrap();

        let stopwords = FtsStopwords::default();
        let matches = |conn: &Connection, table: &str, raw: &str| -> i64 {
            let query = prepare_fts5_query(raw, &stopwords);
            conn.query_row(
                &format!("SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?1"),
                [query],
                |row| row.get(0),
            )
            .unwrap()
        };
        for table in TABLES {
            assert_eq!(matches(&conn, table, "reader.rs"), 0, "{table} at V96");
        }

        run(&mut conn).unwrap();

        for table in TABLES {
            for raw in ["reader.rs", "reader", "src/reader.rs", "ai-memory"] {
                assert_eq!(matches(&conn, table, raw), 1, "{table} query {raw:?}");
            }
        }

        // The recreated triggers keep the index in step with later writes.
        conn.execute(
            "UPDATE pages SET body = 'nothing here' WHERE id = ?1",
            params![page.as_slice()],
        )
        .unwrap();
        assert_eq!(matches(&conn, "pages_fts", "reader.rs"), 0);
        crate::ops::insert_observation(&mut conn, &observation("see src/ops.rs")).unwrap();
        assert_eq!(matches(&conn, "observations_fts", "ops.rs"), 1);
    }

    const V100_TABLES: [&str; 2] = ["pages_fts", "observations_fts"];

    /// Seed a V99 store with one page and one observation whose body holds a
    /// hyphenated and an underscored word, written under the pre-V100 index.
    /// Returns the page id.
    fn seed_v99_hyphenated_rows(conn: &mut Connection) -> [u8; 16] {
        const BODY: &str = "the zebrafinch-marker-alpha tag and the zebra_finch_keeper log";

        run_to(conn, 99).unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(conn, &workspace_id, "proj", None).unwrap();
        let session_id = SessionId::new();
        crate::ops::begin_session(
            conn,
            &NewSession {
                occurred_at: None,
                id: session_id,
                workspace_id,
                project_id,
                agent_kind: AgentKind::Codex,
                cwd: None,
                actor_user: None,
            },
        )
        .unwrap();
        let page = [4_u8; 16];
        conn.execute(
            "INSERT INTO pages \
             (id, workspace_id, project_id, path, title, tier, body, body_sha256, \
              frontmatter_json, is_latest, pinned, created_at, updated_at) \
             VALUES (?1, ?2, ?3, 'notes/markers.md', 'title', 'semantic', ?4, zeroblob(32), '{}', \
                     1, 0, 1, 1)",
            params![
                page.as_slice(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
                BODY,
            ],
        )
        .unwrap();
        crate::ops::insert_observation(
            conn,
            &NewObservation {
                occurred_at: None,
                session_id,
                workspace_id,
                project_id,
                kind: ObservationKind::PostToolUse,
                extension: None,
                source_event: None,
                title: "tool file".into(),
                body: BODY.into(),
                importance: 5,
                file_path: None,
            },
        )
        .unwrap();
        page
    }

    fn v100_matches(conn: &Connection, table: &str, raw: &str) -> i64 {
        use crate::fts_query::{FtsStopwords, prepare_fts5_query};

        let query = prepare_fts5_query(raw, &FtsStopwords::default());
        conn.query_row(
            &format!("SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?1"),
            [query],
            |row| row.get(0),
        )
        .unwrap()
    }

    /// V100 gives both indexes a second spelling of `title`/`body` with `-`
    /// and `_` turned into spaces, so the parts of a hyphenated word match.
    /// A store that already holds such rows gets them re-indexed by the
    /// rebuild, and the whole token keeps matching (#2013).
    #[test]
    fn v100_indexes_the_parts_of_a_hyphenated_word_on_an_existing_store() {
        let mut conn = Connection::open_in_memory().unwrap();
        let page = seed_v99_hyphenated_rows(&mut conn);
        for table in V100_TABLES {
            assert_eq!(v100_matches(&conn, table, "alpha"), 0, "{table} at V99");
        }

        run(&mut conn).unwrap();

        for table in V100_TABLES {
            for part in ["alpha", "marker", "zebrafinch", "keeper"] {
                assert_eq!(
                    v100_matches(&conn, table, part),
                    1,
                    "{table} query {part:?}"
                );
            }
            // The whole token keeps matching: the tokenizer was not touched.
            assert_eq!(
                v100_matches(&conn, table, "zebrafinch-marker-alpha"),
                1,
                "{table} whole token"
            );
        }

        // The recreated triggers keep both spellings in step with later writes.
        conn.execute(
            "UPDATE pages SET body = 'nothing hyphenated here' WHERE id = ?1",
            params![page.as_slice()],
        )
        .unwrap();
        assert_eq!(v100_matches(&conn, "pages_fts", "marker"), 0);
        assert_eq!(v100_matches(&conn, "pages_fts", "hyphenated"), 1);
    }

    /// V100 rebuilds `pages_fts` before it reaches `observations`. When the
    /// observations half fails, the pages half must roll back with it: the
    /// store stays a working V99 store (old columns, old triggers, whole-token
    /// search intact), and a re-run after the cause is removed converges.
    #[test]
    fn v100_failing_partway_rolls_back_to_a_searchable_v99_store() {
        let mut conn = Connection::open_in_memory().unwrap();
        seed_v99_hyphenated_rows(&mut conn);
        // V100 adds `observations.body_parts`; a column already holding that
        // name fails the migration after the pages half has run.
        conn.execute("ALTER TABLE observations ADD COLUMN body_parts TEXT", [])
            .unwrap();

        let err = run(&mut conn).unwrap_err();
        assert!(matches!(err, StoreError::Migration(_)), "{err:?}");
        assert_eq!(applied_versions(&conn).last(), Some(&99));
        let pages_parts: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_xinfo('pages') \
                 WHERE name IN ('title_parts', 'body_parts')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(pages_parts, 0, "the pages half must roll back");
        for table in V100_TABLES {
            assert_eq!(
                v100_matches(&conn, table, "zebrafinch-marker-alpha"),
                1,
                "{table} keeps its V99 index"
            );
            assert_eq!(v100_matches(&conn, table, "alpha"), 0, "{table} at V99");
        }

        conn.execute("ALTER TABLE observations DROP COLUMN body_parts", [])
            .unwrap();
        run(&mut conn).unwrap();
        assert_eq!(
            applied_versions(&conn).last(),
            Some(&i64::from(max_supported_version().unwrap()))
        );
        for table in V100_TABLES {
            assert_eq!(
                v100_matches(&conn, table, "alpha"),
                1,
                "{table} after re-run"
            );
        }
    }

    /// V99 adds `file_path` to a store that already holds observations: an
    /// existing row keeps its columns and reads NULL, a new row stores its
    /// path, and a path lookup uses the partial index.
    #[test]
    fn v99_adds_a_nullable_file_path_to_existing_observations() {
        let mut conn = Connection::open_in_memory().unwrap();
        run_to(&mut conn, 98).unwrap();
        let workspace_id = crate::ops::get_or_create_workspace(&mut conn, "default").unwrap();
        let project_id =
            crate::ops::get_or_create_project(&mut conn, &workspace_id, "proj", None).unwrap();
        let session_id = SessionId::new();
        crate::ops::begin_session(
            &mut conn,
            &NewSession {
                occurred_at: None,
                id: session_id,
                workspace_id,
                project_id,
                agent_kind: AgentKind::Codex,
                cwd: None,
                actor_user: None,
            },
        )
        .unwrap();
        conn.execute(
            "INSERT INTO observations \
             (id, session_id, workspace_id, project_id, kind, title, body, importance, created_at) \
             VALUES (?1, ?2, ?3, ?4, 'post-tool-use', 'tool file', 'before V99', 5, 1)",
            params![
                [7_u8; 16].as_slice(),
                session_id.as_bytes(),
                workspace_id.as_bytes(),
                project_id.as_bytes(),
            ],
        )
        .unwrap();

        run(&mut conn).unwrap();

        let existing: (String, Option<String>) = conn
            .query_row("SELECT body, file_path FROM observations", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert_eq!(existing, ("before V99".to_owned(), None));
        crate::ops::insert_observation(
            &mut conn,
            &NewObservation {
                occurred_at: None,
                session_id,
                workspace_id,
                project_id,
                kind: ObservationKind::PostToolUse,
                extension: None,
                source_event: None,
                title: "tool file".into(),
                body: "after V99".into(),
                importance: 5,
                file_path: Some("src/lib.rs".into()),
            },
        )
        .unwrap();
        let plan: String = conn
            .query_row(
                "EXPLAIN QUERY PLAN SELECT id FROM observations \
                 WHERE workspace_id = ?1 AND project_id = ?2 AND file_path = 'src/lib.rs'",
                params![workspace_id.as_bytes(), project_id.as_bytes()],
                |row| row.get(3),
            )
            .unwrap();
        assert!(plan.contains("idx_observations_file_path"), "{plan}");
    }

    /// The boot-time pre-migration backup warning reads the schema version
    /// without migrating: absent database and blank file report `None` (a
    /// fresh install never warns), a current store reports nothing pending,
    /// and a store stopped partway through the embedded migrations reports
    /// the exact applied ceiling with the gap to the target.
    #[test]
    fn schema_status_reports_absent_fresh_and_old_stores_without_touching_them() {
        // Absent database file.
        let absent = tempfile::TempDir::new().unwrap();
        assert!(
            matches!(schema_status(absent.path()), Ok(None)),
            "an absent database must report nothing to warn about"
        );

        // A file with no applied migration history: a blank database is a
        // fresh install, and the read must not have created or grown it.
        let blank_dir = tempfile::TempDir::new().unwrap();
        let blank = blank_dir.path().join("db").join("memory.sqlite");
        std::fs::create_dir_all(blank.parent().unwrap()).unwrap();
        std::fs::write(&blank, b"").unwrap();
        assert!(matches!(schema_status(blank_dir.path()), Ok(None)));
        assert_eq!(std::fs::read(&blank).unwrap(), b"");

        // A current store: every embedded migration applied, nothing pending.
        let current_dir = tempfile::TempDir::new().unwrap();
        let current = current_dir.path().join("db").join("memory.sqlite");
        std::fs::create_dir_all(current.parent().unwrap()).unwrap();
        {
            let mut conn = Connection::open(&current).unwrap();
            run(&mut conn).unwrap();
        }
        let status = schema_status(current_dir.path()).unwrap().unwrap();
        assert_eq!(status.current, max_supported_version().unwrap());
        assert_eq!(status.target, max_supported_version().unwrap());
        assert!(!status.has_pending_migrations());

        // An older store: only part of the embedded migrations applied.
        let old_dir = tempfile::TempDir::new().unwrap();
        let old = old_dir.path().join("db").join("memory.sqlite");
        std::fs::create_dir_all(old.parent().unwrap()).unwrap();
        {
            let mut conn = Connection::open(&old).unwrap();
            run_to(&mut conn, 50).unwrap();
        }
        let status = schema_status(old_dir.path()).unwrap().unwrap();
        assert_eq!(status.current, 50);
        assert!(status.has_pending_migrations());
        assert!(status.current < status.target);
    }

    /// A store written by the PREVIOUS driver — bundled SQLite 3.46.0
    /// (libsqlite3-sys 0.30.1), refinery 0.8.16, rusqlite 0.32.1, i.e. the
    /// 3.4.4 stack — must open under the upgraded driver (bundled SQLite
    /// 3.54.1, refinery 0.10, rusqlite 0.41) and pass the strict migration
    /// validation (abort_divergent + abort_missing) over every one of the
    /// 101 history rows the old refinery wrote. A fresh-store round-trip
    /// would not prove cross-driver file compatibility, so the input is a
    /// pinned artifact, generated by the scratch `legacy-db-producer`
    /// against the pinned old stack with fully synthetic data (one
    /// workspace, project, session and one FTS-indexed page; no real
    /// store is ever read).
    #[tokio::test]
    async fn a_store_written_by_the_previous_driver_opens_migrates_reopens_and_backs_up() {
        use sha2::{Digest, Sha256};

        const EXPECTED_SHA256: [u8; 32] = [
            0x15, 0x5a, 0x9f, 0x90, 0x27, 0x1a, 0x7b, 0x31, 0x0e, 0x0e, 0x5a, 0x14, 0x3a, 0x44,
            0x04, 0x27, 0xce, 0x49, 0x4d, 0x60, 0x1d, 0x0d, 0x2a, 0xe1, 0xd9, 0x7e, 0xdf, 0xb7,
            0xa5, 0x62, 0xed, 0x92,
        ];
        let fixture: &[u8] = include_bytes!("testdata/legacy-store-v101-sqlite346.db");
        let mut hasher = Sha256::new();
        hasher.update(fixture);
        assert_eq!(
            hasher.finalize().as_slice(),
            &EXPECTED_SHA256[..],
            "the legacy fixture must not change silently"
        );

        let data_dir = tempfile::TempDir::new().unwrap();
        let db_path = data_dir.path().join("db").join(crate::DB_FILENAME);
        std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();
        std::fs::write(&db_path, fixture).unwrap();

        // 1. The upgraded driver opens the legacy store; the strict runner
        //    re-validates every legacy history row, and the boot backfills
        //    are no-ops on an already-complete store.
        let store = crate::Store::open(data_dir.path()).unwrap();

        // 2. The schema sits at the embedded maximum with nothing pending.
        let status = schema_status(data_dir.path()).unwrap().unwrap();
        assert_eq!(status.current, max_supported_version().unwrap());
        assert!(!status.has_pending_migrations());

        // 3. The synthetic page is readable and still FTS5-indexed by the
        //    new driver.
        {
            let conn = Connection::open(store.db_path()).unwrap();
            let body: String = conn
                .query_row(
                    "SELECT body FROM pages WHERE path = 'notes/legacy-fixture.md'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(
                body.contains("upgraded-driver compatibility test"),
                "{body}"
            );
            let fts: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM pages_fts WHERE pages_fts MATCH 'legacy'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(fts, 1, "the legacy page must stay FTS5-retrievable");
            drop(conn);
        }

        // 4. The online backup API (what `ai-memory backup` takes) works on
        //    the legacy file, and the snapshot restores through
        //    `open_snapshot` (the `ai-memory restore` entry point), which
        //    re-runs the strict migration validation on the restored file.
        let backup_path = data_dir.path().join("db").join("legacy-backup.sqlite");
        store.reader.snapshot_to(backup_path.clone()).await.unwrap();
        let snapshot_dir = tempfile::TempDir::new().unwrap();
        let snapshot_db = snapshot_dir.path().join("db").join(crate::DB_FILENAME);
        std::fs::create_dir_all(snapshot_db.parent().unwrap()).unwrap();
        std::fs::copy(&backup_path, &snapshot_db).unwrap();
        let restored = crate::Store::open_snapshot(snapshot_dir.path()).unwrap();
        let body: String = {
            let conn = Connection::open(restored.db_path()).unwrap();
            conn.query_row(
                "SELECT body FROM pages WHERE path = 'notes/legacy-fixture.md'",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert!(
            body.contains("upgraded-driver compatibility test"),
            "{body}"
        );

        // 5. Re-opening the original store is a no-op the second time.
        drop(restored);
        let _ = crate::Store::open(data_dir.path()).unwrap();
        let status = schema_status(data_dir.path()).unwrap().unwrap();
        assert!(!status.has_pending_migrations());
    }
}
