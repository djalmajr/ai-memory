//! Dry-run reviewer for optional auto-improvement proposals.
//!
//! This module is intentionally read-only. It inspects one completed session,
//! asks the configured LLM for structured wiki edit proposals, validates those
//! proposals, and returns a report. Staging and approval live in a later phase.

use std::collections::BTreeSet;

use ai_memory_core::{Observation, ObservationKind, PagePath, ProjectId, SessionId, WorkspaceId};
use ai_memory_llm::{ChatMessage, ChatRequest, LlmError, LlmProvider, Role, complete_structured};
use ai_memory_store::{BriefingPage, ReaderPool, StoredPageBody};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de};
use thiserror::Error;

const CHARS_PER_TOKEN: usize = 4;
const PROMPT_RESERVE_TOKENS: usize = 1_000;
const MAX_PROPOSAL_BODY_CHARS: usize = 32_000;
const DEFAULT_REVIEW_MAX_TOKENS: u32 = 16_000;
const MAX_SESSION_PAGE_CHARS: usize = 32_000;
const MAX_OBSERVATION_BODY_CHARS: usize = 1_500;
const SAMPLE_LIMIT_WITH_SESSION_PAGE: usize = 48;
const SAMPLE_LIMIT_WITHOUT_SESSION_PAGE: usize = 72;
const EVEN_SAMPLE_BUCKETS: usize = 16;
const PROMPT_SCAFFOLD_RESERVE_CHARS: usize = 4_000;

/// Default confidence floor for staged auto-improvement proposals.
pub const DEFAULT_AUTO_IMPROVE_MIN_CONFIDENCE: f32 = 0.75;
/// Default minimum observation count before a session is reviewed.
pub const DEFAULT_AUTO_IMPROVE_MIN_OBSERVATIONS: usize = 8;
/// Default minimum session duration before a session is reviewed.
pub const DEFAULT_AUTO_IMPROVE_MIN_SESSION_DURATION_SECS: u64 = 120;
/// Default approximate input token budget for one review.
pub const DEFAULT_AUTO_IMPROVE_MAX_INPUT_TOKENS: usize = 24_000;
/// Default maximum validated proposals per review.
pub const DEFAULT_AUTO_IMPROVE_MAX_PROPOSALS: usize = 5;
/// Default synthetic actor name for autonomous proposal provenance.
pub const DEFAULT_AUTO_IMPROVE_PROPOSAL_ACTOR: &str = "auto_improve";
/// Default wiki-relative folder for future pending proposal markdown.
pub const DEFAULT_AUTO_IMPROVE_PENDING_PATH: &str = "_pending/auto-improve";

/// Configuration for one dry-run auto-improvement review.
#[derive(Debug, Clone)]
pub struct AutoImproveReviewConfig {
    /// Minimum observations before a session is worth reviewing.
    pub min_observations: usize,
    /// Minimum session duration before a session is worth reviewing.
    pub min_session_duration_secs: u64,
    /// Minimum model confidence accepted by validation.
    pub min_confidence: f32,
    /// Approximate input token budget, using chars/4.
    pub max_input_tokens: usize,
    /// Maximum validated proposals returned from one run.
    pub max_proposals_per_run: usize,
    /// Whether raw fallback content may be considered. Reserved for prompt
    /// visibility in this dry-run slice; hooks still provide raw observations.
    pub include_raw_fallback: bool,
    /// Synthetic actor name used for future proposal provenance.
    pub proposal_actor: String,
    /// Wiki-relative pending proposal folder once staging ships.
    pub pending_path: String,
}

/// Errors raised by the dry-run reviewer.
#[derive(Debug, Error)]
pub enum AutoImproveError {
    /// Underlying store error.
    #[error(transparent)]
    Store(#[from] ai_memory_store::StoreError),
    /// Underlying LLM error.
    #[error(transparent)]
    Llm(#[from] LlmError),
    /// Domain parsing error.
    #[error(transparent)]
    Memory(#[from] ai_memory_core::MemoryError),
    /// Session was not found.
    #[error("session not found: {0}")]
    SessionNotFound(SessionId),
    /// Session belongs to a different scope than the request selected.
    #[error("session {session_id} belongs to a different workspace/project")]
    SessionOutOfScope {
        /// Session that failed the scope check.
        session_id: SessionId,
    },
}

/// Result alias for auto-improvement review.
pub type AutoImproveResult<T> = Result<T, AutoImproveError>;

/// One evidence quote cited by a proposal.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct AutoImproveEvidence {
    /// Source page or observation label, such as `sessions/<id>.md`.
    pub page: String,
    /// Bounded quote supporting the proposed durable edit.
    pub quote: String,
}

impl<'de> Deserialize<'de> for AutoImproveEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum EvidenceInput {
            Object {
                #[serde(default)]
                page: String,
                #[serde(default)]
                quote: String,
            },
            Quote(String),
        }

        match EvidenceInput::deserialize(deserializer)? {
            EvidenceInput::Object { page, quote } => Ok(Self {
                page: if page.trim().is_empty() && !quote.trim().is_empty() {
                    "unspecified".into()
                } else {
                    page
                },
                quote,
            }),
            EvidenceInput::Quote(quote) => Ok(Self {
                page: "unspecified".into(),
                quote,
            }),
        }
    }
}

/// One proposed wiki edit returned by the LLM and accepted by validation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AutoImproveProposal {
    /// Currently only `create_or_update` is supported.
    #[serde(default = "default_operation")]
    pub operation: String,
    /// Relative wiki path that would be created or updated.
    #[serde(default)]
    pub path: String,
    /// Human title for the proposed page.
    #[serde(default)]
    pub title: String,
    /// Semantic kind: gotcha, decision, concept, procedure, rule, fact, note, or slot.
    #[serde(default)]
    pub kind: String,
    /// Model confidence from 0.0 to 1.0.
    #[serde(default)]
    pub confidence: f32,
    /// Why the lesson is durable enough to propose.
    #[serde(default)]
    pub rationale: String,
    /// Evidence quotes that justify the proposal.
    #[serde(default)]
    pub evidence: Vec<AutoImproveEvidence>,
    /// Markdown body without frontmatter.
    #[serde(default, alias = "body", alias = "markdown", alias = "content")]
    pub body_markdown: String,
}

fn default_operation() -> String {
    "create_or_update".into()
}

/// A candidate the reviewer or validator rejected.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AutoImproveRejectedCandidate {
    /// Machine-readable-ish reason for rejection.
    #[serde(default)]
    pub reason: String,
    /// Evidence label or short detail.
    #[serde(default)]
    pub evidence: String,
}

/// Structured response requested from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AutoImproveLlmResponse {
    /// Short summary of the review.
    #[serde(default)]
    pub summary: String,
    /// Candidate edits.
    #[serde(default)]
    pub proposals: Vec<AutoImproveProposal>,
    /// Candidates the model chose not to promote.
    #[serde(default)]
    pub rejected_candidates: Vec<AutoImproveRejectedCandidate>,
}

/// Report returned by the dry-run reviewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoImproveReport {
    /// Always true for this first implementation slice.
    pub dry_run: bool,
    /// Reviewed session id.
    pub session_id: String,
    /// Number of observations read from the session.
    pub observations_considered: usize,
    /// Approximate session span, first observation to last observation.
    pub session_duration_secs: u64,
    /// Approximate prompt tokens sent to the LLM, chars/4.
    pub estimated_input_tokens: usize,
    /// LLM provider name, or `none` when the preflight filter skipped the call.
    pub provider: String,
    /// LLM model, or `none` when the preflight filter skipped the call.
    pub model: String,
    /// Configured confidence floor used by validation.
    pub min_confidence: f32,
    /// Actor name reserved for future staged proposal provenance.
    pub proposal_actor: String,
    /// Wiki-relative pending path reserved for future staged proposal markdown.
    pub pending_path: String,
    /// Review summary.
    pub summary: String,
    /// Validated proposals. No wiki writes have occurred.
    pub proposals: Vec<AutoImproveProposal>,
    /// Model and validator rejections.
    pub rejected_candidates: Vec<AutoImproveRejectedCandidate>,
    /// Non-fatal validation or budget notes.
    pub warnings: Vec<String>,
}

/// Run a dry-run auto-improvement review for one session.
///
/// # Errors
/// Returns store, scope, or LLM errors. This function never writes wiki files or
/// SQLite rows.
pub async fn run_auto_improve_review(
    reader: &ReaderPool,
    llm: &(dyn LlmProvider + 'static),
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    session_id: SessionId,
    cfg: AutoImproveReviewConfig,
) -> AutoImproveResult<AutoImproveReport> {
    match reader.session_project_ids(session_id).await? {
        Some((session_ws, session_proj))
            if session_ws == workspace_id && session_proj == project_id => {}
        Some(_) => return Err(AutoImproveError::SessionOutOfScope { session_id }),
        None => return Err(AutoImproveError::SessionNotFound(session_id)),
    }

    let observations = reader.observations_for_session(session_id).await?;
    let duration = session_duration_secs(&observations);
    if let Some(rejection) = preflight_rejection(&observations, duration, &cfg) {
        return Ok(AutoImproveReport {
            dry_run: true,
            session_id: session_id.to_string(),
            observations_considered: observations.len(),
            session_duration_secs: duration,
            estimated_input_tokens: 0,
            provider: "none".into(),
            model: "none".into(),
            min_confidence: cfg.min_confidence,
            proposal_actor: cfg.proposal_actor,
            pending_path: cfg.pending_path,
            summary: "session skipped by preflight filters".into(),
            proposals: Vec::new(),
            rejected_candidates: vec![rejection],
            warnings: Vec::new(),
        });
    }

    let briefing = reader
        .briefing_for_project(workspace_id, project_id, 100)
        .await?;
    let session_page_path = format!("sessions/{session_id}.md");
    let session_page = reader
        .page_body_by_ids(workspace_id, project_id, &session_page_path)
        .await?;
    let existing_index = ExistingPageIndex::from_pages(&briefing.recent_pages);
    let prompt_input = build_prompt_input(
        session_id,
        &observations,
        duration,
        session_page.as_ref(),
        &briefing.recent_pages,
        &cfg,
    );
    let estimated_input_tokens = estimate_tokens(&prompt_input.prompt);
    let request = ChatRequest {
        system: Some(AUTO_IMPROVE_SYSTEM_PROMPT.to_string()),
        messages: vec![ChatMessage {
            role: Role::User,
            content: prompt_input.prompt,
        }],
        max_tokens: DEFAULT_REVIEW_MAX_TOKENS,
        temperature: Some(0.1),
    };
    let raw: AutoImproveLlmResponse = complete_structured(llm, request).await?;
    let (proposals, mut rejected_candidates, mut warnings) =
        validate_response(raw, &cfg, &existing_index);
    rejected_candidates.extend(prompt_input.rejected_candidates);
    warnings.extend(prompt_input.warnings);

    Ok(AutoImproveReport {
        dry_run: true,
        session_id: session_id.to_string(),
        observations_considered: observations.len(),
        session_duration_secs: duration,
        estimated_input_tokens,
        provider: llm.name().to_string(),
        model: llm.model().to_string(),
        min_confidence: cfg.min_confidence,
        proposal_actor: cfg.proposal_actor,
        pending_path: cfg.pending_path,
        summary: if proposals.is_empty() {
            "review completed; no validated proposals".into()
        } else {
            format!(
                "review completed; {} proposal(s) validated",
                proposals.len()
            )
        },
        proposals,
        rejected_candidates,
        warnings,
    })
}

struct PromptInput {
    prompt: String,
    rejected_candidates: Vec<AutoImproveRejectedCandidate>,
    warnings: Vec<String>,
}

#[derive(Debug, Default)]
struct ExistingPageIndex {
    paths: BTreeSet<String>,
    titles: BTreeSet<String>,
}

impl ExistingPageIndex {
    fn from_pages(pages: &[BriefingPage]) -> Self {
        Self {
            paths: pages.iter().map(|page| page.path.clone()).collect(),
            titles: pages
                .iter()
                .map(|page| normalize_title(&page.title))
                .filter(|title| !title.is_empty())
                .collect(),
        }
    }

    fn contains_path(&self, path: &str) -> bool {
        self.paths.contains(path)
    }

    fn contains_title(&self, title: &str) -> bool {
        let normalized = normalize_title(title);
        !normalized.is_empty() && self.titles.contains(&normalized)
    }
}

fn normalize_title(title: &str) -> String {
    title
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn build_prompt_input(
    session_id: SessionId,
    observations: &[Observation],
    duration_secs: u64,
    session_page: Option<&StoredPageBody>,
    recent_pages: &[BriefingPage],
    cfg: &AutoImproveReviewConfig,
) -> PromptInput {
    let mut warnings = Vec::new();
    let mut rejected_candidates = Vec::new();
    let usable_tokens = cfg.max_input_tokens.saturating_sub(PROMPT_RESERVE_TOKENS);
    let usable_chars = usable_tokens.saturating_mul(CHARS_PER_TOKEN);

    let recent = render_recent_pages(recent_pages);
    let session_page_budget = session_page
        .map(|_| usable_chars / 3)
        .unwrap_or(0)
        .min(MAX_SESSION_PAGE_CHARS);
    let session_page_section =
        render_session_page(session_page, session_page_budget, &mut warnings);
    let observation_budget = usable_chars
        .saturating_sub(recent.len())
        .saturating_sub(session_page_section.len())
        .saturating_sub(PROMPT_SCAFFOLD_RESERVE_CHARS);
    let (rendered_observations, selected_count) = render_sampled_observations(
        observations,
        session_page.is_some(),
        observation_budget,
        &mut warnings,
    );

    if selected_count < observations.len() {
        rejected_candidates.push(AutoImproveRejectedCandidate {
            reason: "input_budget_sampled".into(),
            evidence: format!(
                "selected {selected_count} of {} observations for scalable review",
                observations.len()
            ),
        });
    }

    let prompt = format!(
        "Review one completed ai-memory session and propose only durable wiki edits.\n\
         Session: {session_id}\n\
         Observation count: {}\n\
         Session duration seconds: {duration_secs}\n\
         Minimum confidence: {}\n\
         Max proposals: {}\n\
         Include raw fallback: {}\n\
         Pending path for future staging: {}\n\n\
         Existing project pages, for duplicate avoidance. Do not target these existing paths or titles; diff-based update proposals are a later phase:\n{recent}\n\n\
         Consolidated session page, primary source when present:\n{session_page_section}\n\n\
         Selected observations, sampled for scale and supporting evidence:\n{rendered_observations}\n\n\
         Return proposals only for durable lessons. Prefer gotchas/, decisions/, concepts/, procedures/, _rules/, _slots/current-focus.md, or notes/. Use _rules/ exactly for rules; never rules/. Body markdown must begin with '# <title>'. Reject no-activity sessions, smoke tests, release markers, one-off task narratives, transient failures, generic routing snippets, and agent instruction text unless the session explicitly changed project policy.",
        observations.len(),
        cfg.min_confidence,
        cfg.max_proposals_per_run,
        cfg.include_raw_fallback,
        cfg.pending_path,
    );

    PromptInput {
        prompt,
        rejected_candidates,
        warnings,
    }
}

fn render_session_page(
    page: Option<&StoredPageBody>,
    max_chars: usize,
    warnings: &mut Vec<String>,
) -> String {
    let Some(page) = page else {
        return "(none; relying on sampled observations)".into();
    };
    if max_chars == 0 {
        warnings.push("session page exists but input budget left no room for it".into());
        return "(omitted by input budget)".into();
    }
    let (body, truncated) = truncate_with_marker(&page.body, max_chars, "[session page truncated]");
    if truncated {
        warnings.push(format!(
            "session page body truncated to {max_chars} chars before review"
        ));
    }
    format!(
        "title: {}\ntier: {}\npinned: {}\nbody:\n{}\n",
        page.title, page.tier, page.pinned, body
    )
}

fn render_sampled_observations(
    observations: &[Observation],
    has_session_page: bool,
    max_chars: usize,
    warnings: &mut Vec<String>,
) -> (String, usize) {
    if observations.is_empty() {
        return ("(none)".into(), 0);
    }
    if max_chars == 0 {
        warnings.push("observation sample omitted because session page and page index filled the input budget".into());
        return ("(omitted by input budget)".into(), 0);
    }

    let limit = if has_session_page {
        SAMPLE_LIMIT_WITH_SESSION_PAGE
    } else {
        SAMPLE_LIMIT_WITHOUT_SESSION_PAGE
    };
    let indices = select_observation_indices(observations, limit);
    let mut rendered = String::new();
    let mut kept = 0usize;

    for idx in indices {
        let Some(obs) = observations.get(idx) else {
            continue;
        };
        let chunk = render_observation(idx, observations.len(), obs);
        if !rendered.is_empty() && rendered.len().saturating_add(chunk.len()) > max_chars {
            warnings.push(format!(
                "observation sample stopped after {kept} selected observations to stay under max_input_tokens"
            ));
            break;
        }
        rendered.push_str(&chunk);
        kept += 1;
    }

    if kept < observations.len() {
        warnings.push(format!(
            "observation input sampled {kept} of {} observations for scalable review",
            observations.len()
        ));
    }

    if rendered.is_empty() {
        rendered.push_str("(none fit the input budget)");
    }
    (rendered, kept)
}

fn select_observation_indices(observations: &[Observation], limit: usize) -> Vec<usize> {
    if observations.len() <= limit {
        return (0..observations.len()).collect();
    }
    let even = even_sample_indices(observations.len());
    let mut scored: Vec<(i32, usize)> = observations
        .iter()
        .enumerate()
        .map(|(idx, obs)| {
            let mut score = observation_score(obs, idx, observations.len());
            if even.contains(&idx) {
                score += 40;
            }
            (score, idx)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let selected: BTreeSet<usize> = scored.into_iter().take(limit).map(|(_, idx)| idx).collect();
    selected.into_iter().collect()
}

fn even_sample_indices(total: usize) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    if total == 0 {
        return out;
    }
    if total == 1 {
        out.insert(0);
        return out;
    }
    let buckets = EVEN_SAMPLE_BUCKETS.min(total);
    for bucket in 0..buckets {
        let idx = bucket.saturating_mul(total - 1) / (buckets - 1).max(1);
        out.insert(idx);
    }
    out
}

fn observation_score(obs: &Observation, idx: usize, total: usize) -> i32 {
    let mut score = i32::from(obs.importance);
    score += match obs.kind {
        ObservationKind::UserPrompt => 80,
        ObservationKind::SessionEnd => 70,
        ObservationKind::Stop => 55,
        ObservationKind::PreCompact => 45,
        ObservationKind::PostToolUse => 30,
        ObservationKind::Notification => 25,
        ObservationKind::SessionStart => 20,
        ObservationKind::Other => 15,
        ObservationKind::PreToolUse => 5,
    };
    if idx == 0 || idx + 1 == total {
        score += 100;
    }

    let body_prefix = obs.body.chars().take(4_000).collect::<String>();
    let text = format!("{}\n{}", obs.title, body_prefix).to_ascii_lowercase();
    for keyword in [
        "root cause",
        "fix",
        "fixed",
        "failed",
        "failure",
        "error",
        "bug",
        "regression",
        "decision",
        "decided",
        "gotcha",
        "rule",
        "always",
        "never",
        "migration",
        "scope",
        "workspace",
        "project",
        "auth",
        "test",
        "clippy",
        "release",
    ] {
        if text.contains(keyword) {
            score += 15;
        }
    }
    if obs.body.contains("```") {
        score += 8;
    }
    if text.contains("long-term memory (ai-memory)")
        || text.contains("install ai-memory routing")
        || text.contains("memory_query searches only one project")
    {
        score -= 80;
    }
    score
}

fn render_recent_pages(pages: &[BriefingPage]) -> String {
    if pages.is_empty() {
        return "(none)".into();
    }
    let mut out = String::new();
    for page in pages {
        out.push_str(&format!(
            "- {} | {} | {} | updated {}\n",
            page.path, page.title, page.kind, page.updated_at
        ));
    }
    out
}

fn render_observation(idx: usize, total: usize, obs: &Observation) -> String {
    let (body, _) = truncate_with_marker(
        &obs.body,
        MAX_OBSERVATION_BODY_CHARS,
        "[observation truncated]",
    );
    format!(
        "\n--- observation {}/{} {} ---\nkind: {}\ntitle: {}\nimportance: {}\ncreated_at: {}\nbody:\n{}\n",
        idx + 1,
        total,
        obs.id,
        obs.kind.as_str(),
        obs.title,
        obs.importance,
        obs.created_at,
        body
    )
}

fn truncate_with_marker(input: &str, max_bytes: usize, marker: &str) -> (String, bool) {
    if input.len() <= max_bytes {
        return (input.to_string(), false);
    }
    let mut end = max_bytes.min(input.len());
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = input[..end].to_string();
    out.push('\n');
    out.push_str(marker);
    (out, true)
}

fn preflight_rejection(
    observations: &[Observation],
    duration_secs: u64,
    cfg: &AutoImproveReviewConfig,
) -> Option<AutoImproveRejectedCandidate> {
    if observations.len() < cfg.min_observations {
        return Some(AutoImproveRejectedCandidate {
            reason: "too_few_observations".into(),
            evidence: format!(
                "{} observations below configured minimum {}",
                observations.len(),
                cfg.min_observations
            ),
        });
    }
    if duration_secs < cfg.min_session_duration_secs {
        return Some(AutoImproveRejectedCandidate {
            reason: "session_too_short".into(),
            evidence: format!(
                "{duration_secs}s below configured minimum {}s",
                cfg.min_session_duration_secs
            ),
        });
    }
    None
}

fn validate_response(
    raw: AutoImproveLlmResponse,
    cfg: &AutoImproveReviewConfig,
    existing_index: &ExistingPageIndex,
) -> (
    Vec<AutoImproveProposal>,
    Vec<AutoImproveRejectedCandidate>,
    Vec<String>,
) {
    let mut proposals = Vec::new();
    let mut rejected = raw.rejected_candidates;
    let mut warnings = Vec::new();

    for mut proposal in raw.proposals {
        if proposals.len() >= cfg.max_proposals_per_run {
            rejected.push(AutoImproveRejectedCandidate {
                reason: "max_proposals_exceeded".into(),
                evidence: proposal.path,
            });
            continue;
        }
        normalize_proposal(&mut proposal, &mut warnings);
        match validate_proposal(&proposal, cfg, existing_index) {
            Ok(()) => proposals.push(proposal),
            Err(reason) => rejected.push(AutoImproveRejectedCandidate {
                reason,
                evidence: proposal.path,
            }),
        }
    }

    if raw.summary.trim().is_empty() {
        warnings.push("LLM returned an empty review summary".into());
    }

    (proposals, rejected, warnings)
}

fn normalize_proposal(proposal: &mut AutoImproveProposal, warnings: &mut Vec<String>) {
    if proposal.body_markdown.trim_start().starts_with("# ") {
        return;
    }
    let title = proposal.title.trim();
    let body = proposal.body_markdown.trim_start();
    if title.is_empty() || body.is_empty() {
        return;
    }
    proposal.body_markdown = format!("# {title}\n\n{body}");
    warnings.push(format!(
        "proposal {} body lacked an H1; prepended title as H1 before validation",
        proposal.path
    ));
}

fn validate_proposal(
    proposal: &AutoImproveProposal,
    cfg: &AutoImproveReviewConfig,
    existing_index: &ExistingPageIndex,
) -> Result<(), String> {
    if proposal.operation != "create_or_update" {
        return Err("unsupported_operation".into());
    }
    if proposal.confidence < cfg.min_confidence {
        return Err("confidence_below_threshold".into());
    }
    if proposal.rationale.trim().is_empty() {
        return Err("missing_rationale".into());
    }
    if proposal.evidence.is_empty()
        || proposal
            .evidence
            .iter()
            .any(|e| e.page.trim().is_empty() || e.quote.trim().is_empty())
    {
        return Err("missing_evidence".into());
    }
    if proposal.body_markdown.trim().is_empty() {
        return Err("empty_body".into());
    }
    if proposal.body_markdown.len() > MAX_PROPOSAL_BODY_CHARS {
        return Err("body_too_large".into());
    }
    if !proposal.body_markdown.trim_start().starts_with("# ") {
        return Err("body_missing_h1".into());
    }
    let path = PagePath::new(proposal.path.clone()).map_err(|_| "invalid_path".to_string())?;
    if !allowed_target_path(path.as_str()) {
        return Err("unsupported_path_prefix".into());
    }
    if path.as_str() != "_slots/current-focus.md" && existing_index.contains_path(path.as_str()) {
        return Err("duplicate_existing_path".into());
    }
    if path.as_str() != "_slots/current-focus.md" && existing_index.contains_title(&proposal.title)
    {
        return Err("duplicate_existing_title".into());
    }
    if !kind_matches_path(&proposal.kind, path.as_str()) {
        return Err("kind_path_mismatch".into());
    }
    Ok(())
}

fn allowed_target_path(path: &str) -> bool {
    path.starts_with("gotchas/")
        || path.starts_with("decisions/")
        || path.starts_with("concepts/")
        || path.starts_with("procedures/")
        || path.starts_with("_rules/")
        || path.starts_with("notes/")
        || path == "_slots/current-focus.md"
}

fn kind_matches_path(kind: &str, path: &str) -> bool {
    match kind {
        "gotcha" => path.starts_with("gotchas/"),
        "decision" => path.starts_with("decisions/"),
        "concept" => path.starts_with("concepts/"),
        "procedure" => path.starts_with("procedures/"),
        "rule" => path.starts_with("_rules/"),
        "slot" => path.starts_with("_slots/"),
        "fact" | "note" => path.starts_with("notes/") || path.starts_with("concepts/"),
        _ => false,
    }
}

fn session_duration_secs(observations: &[Observation]) -> u64 {
    let (Some(first), Some(last)) = (observations.first(), observations.last()) else {
        return 0;
    };
    let diff_us = last
        .created_at
        .as_microsecond()
        .saturating_sub(first.created_at.as_microsecond());
    u64::try_from(diff_us / 1_000_000).unwrap_or(0)
}

fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(CHARS_PER_TOKEN)
}

const AUTO_IMPROVE_SYSTEM_PROMPT: &str = r#"You are ai-memory's dry-run auto-improvement reviewer.

Return structured JSON matching the schema. You are proposing wiki edits, not applying them.

Only propose durable, future-useful knowledge:
- gotchas: reproducible pitfalls with a root cause and mitigation
- decisions: choices with rationale and consequences
- concepts: stable architecture or domain knowledge
- procedures: reusable multi-step workflows
- rules: explicit always/never instructions
- notes: useful facts that do not fit stronger categories

Reject transient setup failures, smoke tests, release markers, one-off task narratives, broad negative tool claims, and failures that were resolved without a reusable lesson.

Every proposal must include bounded evidence quotes, confidence, rationale, a valid path, and markdown beginning with an H1. Rule paths must start with `_rules/`, not `rules/`. Do not target sessions/ or _pending/. Treat the consolidated session page as primary when present; selected observations are supporting context. Do not promote ai-memory routing snippets, AGENTS.md/CLAUDE.md instructions, or generic agent tool guidance unless the session explicitly changed project policy."#;

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_core::{AgentKind, NewObservation, NewSession, ObservationId, ObservationKind};
    use ai_memory_llm::{ChatResponse, LlmResult};
    use ai_memory_store::Store;
    use jiff::Timestamp;
    use tempfile::TempDir;

    struct FakeLlm;

    #[async_trait::async_trait]
    impl LlmProvider for FakeLlm {
        fn name(&self) -> &'static str {
            "fake"
        }

        fn model(&self) -> &str {
            "fake-model"
        }

        async fn complete(&self, _request: ChatRequest) -> LlmResult<ChatResponse> {
            Ok(ChatResponse {
                text: "unused".into(),
                usage: None,
                model: "fake-model".into(),
            })
        }

        async fn complete_structured_raw(
            &self,
            _request: ChatRequest,
            _schema: serde_json::Value,
        ) -> LlmResult<serde_json::Value> {
            Ok(serde_json::json!({
                "summary": "found one durable procedure",
                "proposals": [{
                    "operation": "create_or_update",
                    "path": "procedures/release.md",
                    "title": "Release Procedure",
                    "kind": "procedure",
                    "confidence": 0.91,
                    "rationale": "The session repeated a release workflow with verification.",
                    "evidence": [{"page": "sessions/test.md", "quote": "run the full gate before release"}],
                    "body_markdown": "# Release Procedure\n\nRun the full gate before release."
                }],
                "rejected_candidates": []
            }))
        }
    }

    fn cfg() -> AutoImproveReviewConfig {
        AutoImproveReviewConfig {
            min_observations: 3,
            min_session_duration_secs: 60,
            min_confidence: 0.75,
            max_input_tokens: 24_000,
            max_proposals_per_run: 2,
            include_raw_fallback: false,
            proposal_actor: "auto_improve".into(),
            pending_path: "_pending/auto-improve".into(),
        }
    }

    fn proposal(path: &str, kind: &str, confidence: f32) -> AutoImproveProposal {
        AutoImproveProposal {
            operation: "create_or_update".into(),
            path: path.into(),
            title: "Test".into(),
            kind: kind.into(),
            confidence,
            rationale: "durable lesson".into(),
            evidence: vec![AutoImproveEvidence {
                page: "sessions/abc.md".into(),
                quote: "quote".into(),
            }],
            body_markdown: "# Test\n\nBody".into(),
        }
    }

    fn obs(
        idx: usize,
        kind: ObservationKind,
        title: &str,
        body: &str,
        importance: u8,
    ) -> Observation {
        Observation {
            id: ObservationId::new(),
            workspace_id: WorkspaceId::new(),
            project_id: ProjectId::new(),
            session_id: SessionId::new(),
            kind,
            extension: None,
            source_event: None,
            title: title.into(),
            body: body.into(),
            importance,
            created_at: Timestamp::from_microsecond(idx as i64 * 1_000_000)
                .expect("test timestamp is valid"),
        }
    }

    #[tokio::test]
    async fn reviewer_returns_validated_llm_proposals_without_writes() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "proj", None)
            .await
            .unwrap();
        let session_id = ai_memory_core::SessionId::new();
        store
            .writer
            .begin_session(NewSession {
                id: session_id,
                workspace_id: ws,
                project_id: proj,
                agent_kind: AgentKind::Other,
                cwd: None,
            })
            .await
            .unwrap();
        for i in 0..3 {
            store
                .writer
                .insert_observation(NewObservation {
                    session_id,
                    workspace_id: ws,
                    project_id: proj,
                    kind: if i == 0 {
                        ObservationKind::SessionStart
                    } else {
                        ObservationKind::UserPrompt
                    },
                    extension: None,
                    source_event: None,
                    title: format!("event {i}"),
                    body: "run the full gate before release".into(),
                    importance: 5,
                })
                .await
                .unwrap();
        }

        let report = run_auto_improve_review(
            &store.reader,
            &FakeLlm,
            ws,
            proj,
            session_id,
            AutoImproveReviewConfig {
                min_session_duration_secs: 0,
                ..cfg()
            },
        )
        .await
        .unwrap();

        assert!(report.dry_run);
        assert_eq!(report.provider, "fake");
        assert_eq!(report.model, "fake-model");
        assert_eq!(report.proposals.len(), 1);
        assert_eq!(report.proposals[0].path, "procedures/release.md");
        assert!(report.rejected_candidates.is_empty());
    }

    #[test]
    fn validation_accepts_procedure_pages() {
        let raw = AutoImproveLlmResponse {
            summary: "ok".into(),
            proposals: vec![proposal("procedures/release.md", "procedure", 0.91)],
            rejected_candidates: Vec::new(),
        };
        let (accepted, rejected, warnings) =
            validate_response(raw, &cfg(), &ExistingPageIndex::default());
        assert_eq!(accepted.len(), 1);
        assert!(rejected.is_empty());
        assert!(warnings.is_empty());
    }

    #[test]
    fn evidence_deserializes_bare_string_quotes() {
        let raw: AutoImproveLlmResponse = serde_json::from_value(serde_json::json!({
            "summary": "ok",
            "proposals": [{
                "operation": "create_or_update",
                "path": "procedures/release.md",
                "title": "Release Procedure",
                "kind": "procedure",
                "confidence": 0.91,
                "rationale": "The session repeated a release workflow with verification.",
                "evidence": ["run the full gate before release"],
                "body_markdown": "# Release Procedure\n\nRun the full gate before release."
            }],
            "rejected_candidates": []
        }))
        .unwrap();

        assert_eq!(raw.proposals[0].evidence.len(), 1);
        assert_eq!(raw.proposals[0].evidence[0].page, "unspecified");
        assert_eq!(
            raw.proposals[0].evidence[0].quote,
            "run the full gate before release"
        );

        let (accepted, rejected, warnings) =
            validate_response(raw, &cfg(), &ExistingPageIndex::default());
        assert_eq!(accepted.len(), 1);
        assert!(rejected.is_empty());
        assert!(warnings.is_empty());
    }

    #[test]
    fn missing_operation_defaults_to_create_or_update() {
        let raw: AutoImproveLlmResponse = serde_json::from_value(serde_json::json!({
            "summary": "ok",
            "proposals": [{
                "path": "procedures/release.md",
                "title": "Release Procedure",
                "kind": "procedure",
                "confidence": 0.91,
                "rationale": "The session repeated a release workflow with verification.",
                "evidence": [{"quote": "run the full gate before release"}],
                "body_markdown": "# Release Procedure\n\nRun the full gate before release."
            }],
            "rejected_candidates": []
        }))
        .unwrap();

        assert_eq!(raw.proposals[0].operation, "create_or_update");
        assert_eq!(raw.proposals[0].evidence[0].page, "unspecified");
        assert_eq!(
            raw.proposals[0].evidence[0].quote,
            "run the full gate before release"
        );

        let (accepted, rejected, warnings) =
            validate_response(raw, &cfg(), &ExistingPageIndex::default());
        assert_eq!(accepted.len(), 1);
        assert!(rejected.is_empty());
        assert!(warnings.is_empty());
    }

    #[test]
    fn malformed_proposal_missing_path_is_rejected_not_parse_fatal() {
        let raw: AutoImproveLlmResponse = serde_json::from_value(serde_json::json!({
            "summary": "ok",
            "proposals": [{
                "title": "Release Procedure",
                "kind": "procedure",
                "confidence": 0.91,
                "rationale": "The session repeated a release workflow with verification.",
                "evidence": ["run the full gate before release"],
                "body_markdown": "# Release Procedure\n\nRun the full gate before release."
            }],
            "rejected_candidates": []
        }))
        .unwrap();

        let (accepted, rejected, warnings) =
            validate_response(raw, &cfg(), &ExistingPageIndex::default());
        assert!(accepted.is_empty());
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].reason, "invalid_path");
        assert!(warnings.is_empty());
    }

    #[test]
    fn proposal_body_accepts_common_markdown_aliases() {
        let raw: AutoImproveLlmResponse = serde_json::from_value(serde_json::json!({
            "summary": "ok",
            "proposals": [{
                "path": "procedures/release.md",
                "title": "Release Procedure",
                "kind": "procedure",
                "confidence": 0.91,
                "rationale": "The session repeated a release workflow with verification.",
                "evidence": ["run the full gate before release"],
                "body": "# Release Procedure\n\nRun the full gate before release."
            }],
            "rejected_candidates": []
        }))
        .unwrap();

        assert!(
            raw.proposals[0]
                .body_markdown
                .starts_with("# Release Procedure")
        );

        let (accepted, rejected, warnings) =
            validate_response(raw, &cfg(), &ExistingPageIndex::default());
        assert_eq!(accepted.len(), 1);
        assert!(rejected.is_empty());
        assert!(warnings.is_empty());
    }

    #[test]
    fn prompt_uses_session_page_and_samples_high_signal_long_sessions() {
        let session_id = SessionId::new();
        let mut observations: Vec<Observation> = (0..200)
            .map(|idx| {
                obs(
                    idx,
                    ObservationKind::PostToolUse,
                    "routine",
                    "boring output",
                    3,
                )
            })
            .collect();
        observations[180] = obs(
            180,
            ObservationKind::PostToolUse,
            "root cause found",
            "root cause was a stale scope cache; fixed by re-resolving project scope after write errors",
            10,
        );
        let page = StoredPageBody {
            title: "Large Session Summary".into(),
            body: "# Large Session Summary\n\nThe session audited scalable project isolation."
                .into(),
            frontmatter_json: "{}".into(),
            tier: "episodic".into(),
            pinned: false,
        };

        let prompt = build_prompt_input(
            session_id,
            &observations,
            3_600,
            Some(&page),
            &[],
            &AutoImproveReviewConfig {
                max_input_tokens: 8_000,
                ..cfg()
            },
        );

        assert!(prompt.prompt.contains("Large Session Summary"));
        assert!(prompt.prompt.contains("root cause was a stale scope cache"));
        assert!(prompt.warnings.iter().any(|w| w.contains("sampled")));
        assert!(
            prompt
                .rejected_candidates
                .iter()
                .any(|r| r.reason == "input_budget_sampled")
        );
    }

    #[test]
    fn validation_prepends_title_when_body_is_missing_h1() {
        let mut candidate = proposal("gotchas/missing-h1.md", "gotcha", 0.91);
        candidate.body_markdown = "Body without a heading.".into();
        let raw = AutoImproveLlmResponse {
            summary: "ok".into(),
            proposals: vec![candidate],
            rejected_candidates: Vec::new(),
        };

        let (accepted, rejected, warnings) =
            validate_response(raw, &cfg(), &ExistingPageIndex::default());
        assert_eq!(accepted.len(), 1);
        assert!(rejected.is_empty());
        assert!(accepted[0].body_markdown.starts_with("# Test\n\n"));
        assert!(warnings.iter().any(|w| w.contains("prepended title")));
    }

    #[test]
    fn validation_rejects_existing_path_duplicates() {
        let existing = ExistingPageIndex::from_pages(&[BriefingPage {
            path: "gotchas/existing.md".into(),
            title: "Existing Lesson".into(),
            kind: "gotcha".into(),
            updated_at: "2026-06-15T00:00:00Z".into(),
        }]);
        let raw = AutoImproveLlmResponse {
            summary: "ok".into(),
            proposals: vec![
                proposal("gotchas/existing.md", "gotcha", 0.91),
                AutoImproveProposal {
                    title: "Existing Lesson".into(),
                    ..proposal("gotchas/same-title.md", "gotcha", 0.91)
                },
            ],
            rejected_candidates: Vec::new(),
        };

        let (accepted, rejected, _warnings) = validate_response(raw, &cfg(), &existing);
        assert!(accepted.is_empty());
        assert_eq!(rejected.len(), 2);
        assert_eq!(rejected[0].reason, "duplicate_existing_path");
        assert_eq!(rejected[1].reason, "duplicate_existing_title");
    }

    #[test]
    fn validation_rejects_low_confidence_and_bad_prefix() {
        let raw = AutoImproveLlmResponse {
            summary: "ok".into(),
            proposals: vec![
                proposal("gotchas/good.md", "gotcha", 0.2),
                proposal("sessions/new.md", "fact", 0.9),
            ],
            rejected_candidates: Vec::new(),
        };
        let (accepted, rejected, _warnings) =
            validate_response(raw, &cfg(), &ExistingPageIndex::default());
        assert!(accepted.is_empty());
        assert_eq!(rejected.len(), 2);
        assert_eq!(rejected[0].reason, "confidence_below_threshold");
        assert_eq!(rejected[1].reason, "unsupported_path_prefix");
    }

    #[test]
    fn validation_caps_proposal_count() {
        let raw = AutoImproveLlmResponse {
            summary: "ok".into(),
            proposals: vec![
                proposal("gotchas/one.md", "gotcha", 0.9),
                proposal("decisions/two.md", "decision", 0.9),
                proposal("concepts/three.md", "concept", 0.9),
            ],
            rejected_candidates: Vec::new(),
        };
        let (accepted, rejected, _warnings) =
            validate_response(raw, &cfg(), &ExistingPageIndex::default());
        assert_eq!(accepted.len(), 2);
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].reason, "max_proposals_exceeded");
    }
}
