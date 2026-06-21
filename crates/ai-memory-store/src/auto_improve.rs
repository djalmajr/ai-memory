#![allow(missing_docs)]

use std::str::FromStr;

use ai_memory_core::{
    ActorContext, AutoImproveProposalId, AutoImproveRunId, NewPage, PageId, PagePath, ProjectId,
    SessionId, UserId, WorkspaceId,
};
use jiff::Timestamp;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{StoreError, StoreResult};
use crate::ops;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoImproveProposalStatus {
    Pending,
    Approved,
    Rejected,
    Conflict,
    Failed,
}

impl AutoImproveProposalStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Rejected => "rejected",
            Self::Conflict => "conflict",
            Self::Failed => "failed",
        }
    }
}

impl FromStr for AutoImproveProposalStatus {
    type Err = StoreError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "approved" => Ok(Self::Approved),
            "rejected" => Ok(Self::Rejected),
            "conflict" => Ok(Self::Conflict),
            "failed" => Ok(Self::Failed),
            other => Err(StoreError::MalformedRecord(format!(
                "unknown auto-improve proposal status: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoImproveProposalOperation {
    Create,
    Update,
}

impl AutoImproveProposalOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Update => "update",
        }
    }
}

impl FromStr for AutoImproveProposalOperation {
    type Err = StoreError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "create" => Ok(Self::Create),
            "update" => Ok(Self::Update),
            other => Err(StoreError::MalformedRecord(format!(
                "unknown auto-improve proposal operation: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StageAutoImproveRun {
    pub workspace_id: WorkspaceId,
    pub project_id: ProjectId,
    pub session_id: Option<SessionId>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub summary: Option<String>,
    pub warnings_json: serde_json::Value,
    pub rejected_candidates_json: serde_json::Value,
    pub config_json: serde_json::Value,
    pub proposal_actor: ActorContext,
    pub proposals: Vec<NewAutoImproveProposal>,
}

#[derive(Debug, Clone)]
pub struct NewAutoImproveProposal {
    pub operation: AutoImproveProposalOperation,
    pub target_path: PagePath,
    pub kind: String,
    pub title: String,
    pub confidence: f64,
    pub rationale: String,
    pub evidence_json: serde_json::Value,
    pub body_markdown: String,
    pub artifact_sha256: Option<[u8; 32]>,
    pub edit_mode: Option<String>,
    pub patch_json: Option<serde_json::Value>,
    pub expected_base_body_sha256: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StagedAutoImproveRun {
    pub run_id: AutoImproveRunId,
    pub proposal_ids: Vec<AutoImproveProposalId>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoImproveProposalSummary {
    pub id: AutoImproveProposalId,
    pub run_id: AutoImproveRunId,
    pub workspace_id: WorkspaceId,
    pub project_id: ProjectId,
    pub status: AutoImproveProposalStatus,
    pub operation: AutoImproveProposalOperation,
    pub target_path: PagePath,
    pub kind: String,
    pub title: String,
    pub confidence: f64,
    pub staged_at: i64,
    pub decided_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoImproveProposalDetail {
    pub summary: AutoImproveProposalSummary,
    pub rationale: String,
    pub evidence_json: serde_json::Value,
    pub body_markdown: String,
    pub body_sha256: [u8; 32],
    pub artifact_path: String,
    pub artifact_sha256: Option<[u8; 32]>,
    pub target_latest_page_id_at_stage: Option<PageId>,
    pub target_body_sha256_at_stage: Option<[u8; 32]>,
    pub target_updated_at_at_stage: Option<i64>,
    pub decision_reason: Option<String>,
    pub decided_by_author_id: Option<UserId>,
    pub decided_by_actor_json: Option<serde_json::Value>,
    pub applied_page_id: Option<PageId>,
    pub checkpoint: Option<String>,
    pub edit_mode: String,
    pub patch_json: Option<serde_json::Value>,
    pub expected_base_body_sha256: Option<[u8; 32]>,
    pub materialized_base_body_sha256: Option<[u8; 32]>,
    pub events: Vec<AutoImproveProposalEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoImproveProposalEvent {
    pub id: i64,
    pub proposal_id: AutoImproveProposalId,
    pub event: String,
    pub actor_json: serde_json::Value,
    pub author_id: Option<UserId>,
    pub detail_json: serde_json::Value,
    pub at: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AutoImproveRejectionSummary {
    pub id: String,
    pub workspace_id: WorkspaceId,
    pub project_id: ProjectId,
    pub target_path: Option<String>,
    pub kind: Option<String>,
    pub operation: Option<String>,
    pub edit_mode: Option<String>,
    pub reason: String,
    pub normalized_fingerprint: String,
    pub summary: String,
    pub evidence_json: serde_json::Value,
    pub source_run_id: Option<AutoImproveRunId>,
    pub source_proposal_id: Option<AutoImproveProposalId>,
    pub created_at: i64,
}

#[derive(Debug, Clone)]
pub struct RejectAutoImproveProposal {
    pub workspace_id: WorkspaceId,
    pub project_id: ProjectId,
    pub proposal_id: AutoImproveProposalId,
    pub reason: String,
    pub actor: ActorContext,
    pub author_id: Option<UserId>,
}

#[derive(Debug, Clone)]
pub struct FailAutoImproveProposal {
    pub workspace_id: WorkspaceId,
    pub project_id: ProjectId,
    pub proposal_id: AutoImproveProposalId,
    pub reason: String,
    pub actor: ActorContext,
    pub author_id: Option<UserId>,
}

#[derive(Debug, Clone)]
pub struct ApproveAutoImproveProposal {
    pub workspace_id: WorkspaceId,
    pub project_id: ProjectId,
    pub proposal_id: AutoImproveProposalId,
    pub page: NewPage,
    pub actor: ActorContext,
    pub author_id: Option<UserId>,
    pub checkpoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApproveAutoImproveProposalResult {
    Approved { page_id: PageId },
    Conflict,
}

pub fn artifact_path_for(proposal_id: AutoImproveProposalId) -> String {
    format!("_pending/auto-improve/{proposal_id}.md")
}

pub fn ensure_scheduler_state(
    conn: &mut Connection,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let watermark_ended_at = conn.query_row(
        "SELECT COALESCE(MAX(ended_at), 0) FROM sessions \
         WHERE workspace_id = ?1 AND project_id = ?2 AND ended_at IS NOT NULL",
        params![workspace_id.as_bytes(), project_id.as_bytes()],
        |row| row.get::<_, i64>(0),
    )?;
    conn.execute(
        "INSERT INTO auto_improve_scheduler_state \
         (workspace_id, project_id, watermark_ended_at, initialized_at, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?4) \
         ON CONFLICT(workspace_id, project_id) DO NOTHING",
        params![
            workspace_id.as_bytes(),
            project_id.as_bytes(),
            watermark_ended_at,
            now,
        ],
    )?;
    Ok(())
}

pub fn claim_scheduler_session(
    conn: &mut Connection,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    session_id: SessionId,
    ended_at: i64,
) -> StoreResult<bool> {
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    let inserted = tx.execute(
        "INSERT OR IGNORE INTO auto_improve_scheduler_claims \
         (workspace_id, project_id, session_id, claimed_at) \
         SELECT ?1, ?2, ?3, ?4 \
         WHERE EXISTS ( \
             SELECT 1 FROM auto_improve_scheduler_state st \
             JOIN sessions s \
               ON s.workspace_id = st.workspace_id \
              AND s.project_id = st.project_id \
             WHERE st.workspace_id = ?1 \
               AND st.project_id = ?2 \
               AND s.id = ?3 \
               AND s.ended_at = ?5 \
               AND s.ended_at > st.watermark_ended_at \
         ) \
           AND NOT EXISTS ( \
               SELECT 1 FROM auto_improve_runs r \
               WHERE r.workspace_id = ?1 \
                 AND r.project_id = ?2 \
                 AND r.session_id = ?3 \
           )",
        params![
            workspace_id.as_bytes(),
            project_id.as_bytes(),
            session_id.as_bytes(),
            now,
            ended_at,
        ],
    )?;
    if inserted == 1 {
        tx.execute(
            "UPDATE auto_improve_scheduler_state \
             SET updated_at = ?3 \
             WHERE workspace_id = ?1 AND project_id = ?2",
            params![workspace_id.as_bytes(), project_id.as_bytes(), now],
        )?;
    }
    tx.commit()?;
    Ok(inserted == 1)
}

pub fn stage_run(
    conn: &mut Connection,
    input: &StageAutoImproveRun,
) -> StoreResult<StagedAutoImproveRun> {
    let now = Timestamp::now().as_microsecond();
    let run_id = AutoImproveRunId::new();
    let actor_json = serde_json::to_string(&input.proposal_actor)?;
    let warnings_json = serde_json::to_string(&input.warnings_json)?;
    let rejected_json = serde_json::to_string(&input.rejected_candidates_json)?;
    let config_json = serde_json::to_string(&input.config_json)?;
    let tx = conn.transaction()?;
    if let Some(session_id) = input.session_id {
        tx.query_row(
            "SELECT 1 FROM sessions WHERE id = ?1 AND workspace_id = ?2 AND project_id = ?3",
            params![
                session_id.as_bytes(),
                input.workspace_id.as_bytes(),
                input.project_id.as_bytes(),
            ],
            |_| Ok(()),
        )
        .optional()?
        .ok_or_else(|| {
            StoreError::InvalidState("auto-improve session is not in proposal scope".into())
        })?;
    }
    tx.execute(
        "INSERT INTO auto_improve_runs \
         (id, workspace_id, project_id, session_id, provider, model, summary, warnings_json, \
          rejected_candidates_json, config_json, proposal_actor_json, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            run_id.as_bytes(),
            input.workspace_id.as_bytes(),
            input.project_id.as_bytes(),
            input.session_id.map(|id| id.as_bytes().to_vec()),
            input.provider.as_deref(),
            input.model.as_deref(),
            input.summary.as_deref(),
            warnings_json,
            rejected_json,
            config_json,
            actor_json,
            now,
        ],
    )?;
    insert_rejected_candidates_in_tx(&tx, input, run_id, now)?;
    let mut proposal_ids = Vec::with_capacity(input.proposals.len());
    for proposal in &input.proposals {
        let id = AutoImproveProposalId::new();
        let artifact_path = artifact_path_for(id);
        let evidence_json = serde_json::to_string(&proposal.evidence_json)?;
        let body_sha256 = sha256(proposal.body_markdown.as_bytes());
        let edit_mode = proposal.edit_mode.as_deref().unwrap_or("full_page");
        let patch_json = proposal
            .patch_json
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let target_snapshot = latest_target_snapshot(
            &tx,
            input.workspace_id,
            input.project_id,
            proposal.target_path.as_str(),
        )?;
        let (
            target_latest_page_id_at_stage,
            target_body_sha256_at_stage,
            target_updated_at_at_stage,
        ) = match (proposal.operation, target_snapshot) {
            (AutoImproveProposalOperation::Create, None) => (None, None, None),
            (AutoImproveProposalOperation::Create, Some(_)) => {
                return Err(StoreError::InvalidState(format!(
                    "create proposal target already exists: {}",
                    proposal.target_path
                )));
            }
            (AutoImproveProposalOperation::Update, Some((id, body_hash, updated_at))) => {
                (Some(id), Some(bytes32(body_hash)?), Some(updated_at))
            }
            (AutoImproveProposalOperation::Update, None) => {
                return Err(StoreError::InvalidState(format!(
                    "update proposal target does not exist: {}",
                    proposal.target_path
                )));
            }
        };
        if edit_mode == "patch" {
            if proposal.operation != AutoImproveProposalOperation::Update {
                return Err(StoreError::InvalidState(format!(
                    "patch proposal must use update operation: {}",
                    proposal.target_path
                )));
            }
            if patch_json.is_none() {
                return Err(StoreError::InvalidState(format!(
                    "patch proposal missing patch_json: {}",
                    proposal.target_path
                )));
            }
            let Some(expected) = proposal.expected_base_body_sha256 else {
                return Err(StoreError::InvalidState(format!(
                    "patch proposal missing expected base body hash: {}",
                    proposal.target_path
                )));
            };
            match target_body_sha256_at_stage {
                Some(current) if current == expected => {}
                Some(_) => {
                    return Err(StoreError::InvalidState(format!(
                        "auto-improve proposal target changed since patch materialization: {}",
                        proposal.target_path
                    )));
                }
                None => {
                    return Err(StoreError::InvalidState(format!(
                        "patch proposal target does not exist: {}",
                        proposal.target_path
                    )));
                }
            }
        }
        tx.execute(
            "INSERT INTO auto_improve_proposals \
             (id, run_id, workspace_id, project_id, status, operation, target_path, kind, title, \
              confidence, rationale, evidence_json, body_markdown, body_sha256, artifact_path, \
              artifact_sha256, target_latest_page_id_at_stage, target_body_sha256_at_stage, \
              target_updated_at_at_stage, staged_at, edit_mode, patch_json, \
              expected_base_body_sha256, materialized_base_body_sha256) \
             VALUES (?1, ?2, ?3, ?4, 'pending', ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, \
                     ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23)",
            params![
                id.as_bytes(),
                run_id.as_bytes(),
                input.workspace_id.as_bytes(),
                input.project_id.as_bytes(),
                proposal.operation.as_str(),
                proposal.target_path.as_str(),
                proposal.kind.as_str(),
                proposal.title.as_str(),
                proposal.confidence,
                proposal.rationale.as_str(),
                evidence_json,
                proposal.body_markdown.as_str(),
                body_sha256.as_slice(),
                artifact_path,
                proposal.artifact_sha256.map(|h| h.to_vec()),
                target_latest_page_id_at_stage.map(|id| id.as_bytes().to_vec()),
                target_body_sha256_at_stage.map(|h| h.to_vec()),
                target_updated_at_at_stage,
                now,
                edit_mode,
                patch_json,
                proposal.expected_base_body_sha256.map(|h| h.to_vec()),
                proposal.expected_base_body_sha256.map(|h| h.to_vec()),
            ],
        )?;
        insert_event_in_tx(
            &tx,
            id,
            "staged",
            &input.proposal_actor,
            None,
            &serde_json::json!({}),
            now,
        )?;
        proposal_ids.push(id);
    }
    tx.commit()?;
    Ok(StagedAutoImproveRun {
        run_id,
        proposal_ids,
    })
}

pub fn fail_proposal(conn: &mut Connection, input: &FailAutoImproveProposal) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let actor_json = serde_json::to_string(&input.actor)?;
    let tx = conn.transaction()?;
    let changed = tx.execute(
        "UPDATE auto_improve_proposals \
         SET status = 'failed', decided_at = ?1, decision_reason = ?2, \
             decided_by_author_id = ?3, decided_by_actor_json = ?4 \
         WHERE id = ?5 AND workspace_id = ?6 AND project_id = ?7 AND status = 'pending'",
        params![
            now,
            input.reason.as_str(),
            input.author_id.map(|id| id.as_bytes().to_vec()),
            actor_json,
            input.proposal_id.as_bytes(),
            input.workspace_id.as_bytes(),
            input.project_id.as_bytes(),
        ],
    )?;
    if changed != 1 {
        return Err(StoreError::InvalidState(
            "auto-improve proposal is not pending or not in scope".into(),
        ));
    }
    insert_rejection_for_proposal_in_tx(&tx, input.proposal_id, &input.reason, now)?;
    insert_event_in_tx(
        &tx,
        input.proposal_id,
        "failed",
        &input.actor,
        input.author_id,
        &serde_json::json!({ "reason": input.reason.as_str() }),
        now,
    )?;
    tx.commit()?;
    Ok(())
}

pub fn reject_proposal(
    conn: &mut Connection,
    input: &RejectAutoImproveProposal,
) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let actor_json = serde_json::to_string(&input.actor)?;
    let tx = conn.transaction()?;
    let changed = tx.execute(
        "UPDATE auto_improve_proposals \
         SET status = 'rejected', decided_at = ?1, decision_reason = ?2, \
             decided_by_author_id = ?3, decided_by_actor_json = ?4 \
         WHERE id = ?5 AND workspace_id = ?6 AND project_id = ?7 AND status = 'pending'",
        params![
            now,
            input.reason.as_str(),
            input.author_id.map(|id| id.as_bytes().to_vec()),
            actor_json,
            input.proposal_id.as_bytes(),
            input.workspace_id.as_bytes(),
            input.project_id.as_bytes(),
        ],
    )?;
    if changed != 1 {
        return Err(StoreError::InvalidState(
            "auto-improve proposal is not pending or not in scope".into(),
        ));
    }
    insert_rejection_for_proposal_in_tx(&tx, input.proposal_id, &input.reason, now)?;
    insert_event_in_tx(
        &tx,
        input.proposal_id,
        "rejected",
        &input.actor,
        input.author_id,
        &serde_json::json!({ "reason": input.reason.as_str() }),
        now,
    )?;
    tx.commit()?;
    Ok(())
}

pub fn approve_proposal(
    conn: &mut Connection,
    input: &ApproveAutoImproveProposal,
) -> StoreResult<ApproveAutoImproveProposalResult> {
    if input.page.workspace_id != input.workspace_id || input.page.project_id != input.project_id {
        return Err(StoreError::InvalidState(
            "approval page scope does not match proposal scope".into(),
        ));
    }
    if input.page.author_id != input.author_id {
        return Err(StoreError::InvalidState(
            "approval page author does not match approver author".into(),
        ));
    }
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    let proposal = tx
        .query_row(
            "SELECT operation, target_path, target_latest_page_id_at_stage, \
                    target_body_sha256_at_stage, target_updated_at_at_stage \
             FROM auto_improve_proposals \
             WHERE id = ?1 AND workspace_id = ?2 AND project_id = ?3 AND status = 'pending'",
            params![
                input.proposal_id.as_bytes(),
                input.workspace_id.as_bytes(),
                input.project_id.as_bytes(),
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<Vec<u8>>>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                ))
            },
        )
        .optional()?;
    let Some((operation, target_path, staged_page_id, staged_body_hash, staged_updated_at)) =
        proposal
    else {
        return Err(StoreError::InvalidState(
            "auto-improve proposal is not pending or not in scope".into(),
        ));
    };
    if input.page.path.as_str() != target_path {
        return Err(StoreError::InvalidState(
            "approval page path does not match proposal target".into(),
        ));
    }

    let current = latest_target_snapshot(&tx, input.workspace_id, input.project_id, &target_path)?;
    let conflict = match AutoImproveProposalOperation::from_str(&operation)? {
        AutoImproveProposalOperation::Create => current.is_some(),
        AutoImproveProposalOperation::Update => match current {
            Some((id, body_hash, updated_at)) => {
                Some(id.as_bytes().to_vec()) != staged_page_id
                    || Some(body_hash) != staged_body_hash
                    || Some(updated_at) != staged_updated_at
            }
            None => true,
        },
    };
    if conflict {
        insert_rejection_for_proposal_in_tx(
            &tx,
            input.proposal_id,
            "target changed since proposal was staged",
            now,
        )?;
        mark_decision_in_tx(
            &tx,
            input,
            "conflict",
            None,
            Some("target changed since proposal was staged"),
            now,
        )?;
        insert_event_in_tx(
            &tx,
            input.proposal_id,
            "conflict",
            &input.actor,
            input.author_id,
            &serde_json::json!({ "reason": "target changed since proposal was staged" }),
            now,
        )?;
        tx.commit()?;
        return Ok(ApproveAutoImproveProposalResult::Conflict);
    }

    let page_id = ops::upsert_page_in_tx(&tx, &input.page, now)?;
    mark_decision_in_tx(&tx, input, "approved", Some(page_id), None, now)?;
    insert_event_in_tx(
        &tx,
        input.proposal_id,
        "approved",
        &input.actor,
        input.author_id,
        &serde_json::json!({ "applied_page_id": page_id.to_string() }),
        now,
    )?;
    tx.commit()?;
    Ok(ApproveAutoImproveProposalResult::Approved { page_id })
}

fn mark_decision_in_tx(
    tx: &rusqlite::Transaction<'_>,
    input: &ApproveAutoImproveProposal,
    status: &str,
    applied_page_id: Option<PageId>,
    reason: Option<&str>,
    now: i64,
) -> StoreResult<()> {
    let actor_json = serde_json::to_string(&input.actor)?;
    tx.execute(
        "UPDATE auto_improve_proposals \
         SET status = ?1, decided_at = ?2, decision_reason = ?3, decided_by_author_id = ?4, \
             decided_by_actor_json = ?5, applied_page_id = ?6, checkpoint = ?7 \
         WHERE id = ?8 AND workspace_id = ?9 AND project_id = ?10 AND status = 'pending'",
        params![
            status,
            now,
            reason,
            input.author_id.map(|id| id.as_bytes().to_vec()),
            actor_json,
            applied_page_id.map(|id| id.as_bytes().to_vec()),
            input.checkpoint.as_deref(),
            input.proposal_id.as_bytes(),
            input.workspace_id.as_bytes(),
            input.project_id.as_bytes(),
        ],
    )?;
    Ok(())
}

fn latest_target_snapshot(
    tx: &rusqlite::Transaction<'_>,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    target_path: &str,
) -> StoreResult<Option<(PageId, Vec<u8>, i64)>> {
    let row = tx
        .query_row(
            "SELECT id, body_sha256, updated_at FROM pages \
             WHERE workspace_id = ?1 AND project_id = ?2 AND path = ?3 AND is_latest = 1",
            params![workspace_id.as_bytes(), project_id.as_bytes(), target_path],
            |row| {
                Ok((
                    PageId::from_slice(&row.get::<_, Vec<u8>>(0)?).map_err(to_sql_err)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()?;
    Ok(row)
}

fn insert_event_in_tx(
    tx: &rusqlite::Transaction<'_>,
    proposal_id: AutoImproveProposalId,
    event: &str,
    actor: &ActorContext,
    author_id: Option<UserId>,
    detail: &serde_json::Value,
    at: i64,
) -> StoreResult<()> {
    tx.execute(
        "INSERT INTO auto_improve_proposal_events \
         (proposal_id, event, actor_json, author_id, detail_json, at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            proposal_id.as_bytes(),
            event,
            serde_json::to_string(actor)?,
            author_id.map(|id| id.as_bytes().to_vec()),
            serde_json::to_string(detail)?,
            at,
        ],
    )?;
    Ok(())
}

fn insert_rejected_candidates_in_tx(
    tx: &rusqlite::Transaction<'_>,
    input: &StageAutoImproveRun,
    run_id: AutoImproveRunId,
    now: i64,
) -> StoreResult<()> {
    let Some(candidates) = input.rejected_candidates_json.as_array() else {
        return Ok(());
    };
    for candidate in candidates {
        let reason = candidate
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .trim();
        if reason.is_empty() {
            continue;
        }
        let target_path = string_field(candidate, "target_path")
            .or_else(|| string_field(candidate, "path"))
            .or_else(|| {
                string_field(candidate, "evidence").filter(|value| PagePath::new(value).is_ok())
            });
        let summary = string_field(candidate, "summary")
            .or_else(|| string_field(candidate, "evidence"))
            .unwrap_or_else(|| reason.to_string());
        let record = NewAutoImproveRejectionRecord {
            workspace_id: input.workspace_id,
            project_id: input.project_id,
            target_path,
            kind: string_field(candidate, "kind"),
            operation: string_field(candidate, "operation"),
            edit_mode: string_field(candidate, "edit_mode"),
            reason: reason.to_string(),
            summary,
            evidence_json: candidate.clone(),
            source_run_id: Some(run_id),
            source_proposal_id: None,
        };
        insert_rejection_record_in_tx(tx, &record, now)?;
    }
    Ok(())
}

fn string_field(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
}

struct NewAutoImproveRejectionRecord {
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    target_path: Option<String>,
    kind: Option<String>,
    operation: Option<String>,
    edit_mode: Option<String>,
    reason: String,
    summary: String,
    evidence_json: serde_json::Value,
    source_run_id: Option<AutoImproveRunId>,
    source_proposal_id: Option<AutoImproveProposalId>,
}

fn insert_rejection_for_proposal_in_tx(
    tx: &rusqlite::Transaction<'_>,
    proposal_id: AutoImproveProposalId,
    reason: &str,
    now: i64,
) -> StoreResult<()> {
    let record = tx.query_row(
        "SELECT run_id, workspace_id, project_id, operation, target_path, kind, title, rationale, \
                evidence_json, edit_mode \
         FROM auto_improve_proposals WHERE id = ?1",
        params![proposal_id.as_bytes()],
        |row| {
            let run_id =
                AutoImproveRunId::from_slice(&row.get::<_, Vec<u8>>(0)?).map_err(to_sql_err)?;
            let workspace_id =
                WorkspaceId::from_slice(&row.get::<_, Vec<u8>>(1)?).map_err(to_sql_err)?;
            let project_id =
                ProjectId::from_slice(&row.get::<_, Vec<u8>>(2)?).map_err(to_sql_err)?;
            let evidence_raw: String = row.get(8)?;
            let title: String = row.get(6)?;
            let rationale: String = row.get(7)?;
            Ok(NewAutoImproveRejectionRecord {
                workspace_id,
                project_id,
                target_path: Some(row.get(4)?),
                kind: Some(row.get(5)?),
                operation: Some(row.get(3)?),
                edit_mode: Some(row.get(9)?),
                reason: reason.to_string(),
                summary: if title.trim().is_empty() {
                    rationale
                } else {
                    title
                },
                evidence_json: serde_json::from_str(&evidence_raw).map_err(to_sql_err)?,
                source_run_id: Some(run_id),
                source_proposal_id: Some(proposal_id),
            })
        },
    )?;
    insert_rejection_record_in_tx(tx, &record, now)
}

fn insert_rejection_record_in_tx(
    tx: &rusqlite::Transaction<'_>,
    record: &NewAutoImproveRejectionRecord,
    now: i64,
) -> StoreResult<()> {
    let id = Uuid::new_v4();
    let evidence_json = serde_json::to_string(&record.evidence_json)?;
    let fingerprint = rejection_fingerprint(record);
    tx.execute(
        "INSERT INTO auto_improve_rejections \
         (id, workspace_id, project_id, target_path, kind, operation, edit_mode, reason, \
          normalized_fingerprint, summary, evidence_json, source_run_id, source_proposal_id, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            id.as_bytes().as_slice(),
            record.workspace_id.as_bytes(),
            record.project_id.as_bytes(),
            record.target_path.as_deref(),
            record.kind.as_deref(),
            record.operation.as_deref(),
            record.edit_mode.as_deref(),
            record.reason.as_str(),
            fingerprint,
            record.summary.as_str(),
            evidence_json,
            record.source_run_id.map(|id| id.as_bytes().to_vec()),
            record.source_proposal_id.map(|id| id.as_bytes().to_vec()),
            now,
        ],
    )?;
    Ok(())
}

fn rejection_fingerprint(record: &NewAutoImproveRejectionRecord) -> String {
    let input = [
        normalize_fp(record.target_path.as_deref().unwrap_or("")),
        normalize_fp(record.kind.as_deref().unwrap_or("")),
        normalize_fp(record.operation.as_deref().unwrap_or("")),
        normalize_fp(record.edit_mode.as_deref().unwrap_or("")),
        normalize_fp(&record.reason),
        normalize_fp(&record.summary),
    ]
    .join("\n");
    hex_sha256(input.as_bytes())
}

fn normalize_fp(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn hex_sha256(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

pub(crate) fn summary_from_row(row: &Row<'_>) -> rusqlite::Result<AutoImproveProposalSummary> {
    let status: String = row.get(4)?;
    let operation: String = row.get(5)?;
    let id = AutoImproveProposalId::from_slice(&row.get::<_, Vec<u8>>(0)?).map_err(to_sql_err)?;
    let run_id = AutoImproveRunId::from_slice(&row.get::<_, Vec<u8>>(1)?).map_err(to_sql_err)?;
    let workspace_id = WorkspaceId::from_slice(&row.get::<_, Vec<u8>>(2)?).map_err(to_sql_err)?;
    let project_id = ProjectId::from_slice(&row.get::<_, Vec<u8>>(3)?).map_err(to_sql_err)?;
    let target_path = PagePath::new(row.get::<_, String>(6)?).map_err(to_sql_err)?;
    Ok(AutoImproveProposalSummary {
        id,
        run_id,
        workspace_id,
        project_id,
        status: AutoImproveProposalStatus::from_str(&status).map_err(to_sql_err)?,
        operation: AutoImproveProposalOperation::from_str(&operation).map_err(to_sql_err)?,
        target_path,
        kind: row.get(7)?,
        title: row.get(8)?,
        confidence: row.get(9)?,
        staged_at: row.get(10)?,
        decided_at: row.get(11)?,
    })
}

pub(crate) fn bytes32(bytes: Vec<u8>) -> StoreResult<[u8; 32]> {
    bytes
        .try_into()
        .map_err(|_| StoreError::MalformedRecord("invalid sha256 length".into()))
}

pub(crate) fn opt_bytes32(bytes: Option<Vec<u8>>) -> StoreResult<Option<[u8; 32]>> {
    bytes.map(bytes32).transpose()
}

pub(crate) fn to_sql_err<E: std::error::Error + Send + Sync + 'static>(err: E) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Blob, Box::new(err))
}
