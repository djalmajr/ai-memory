//! M8 lint pass — rule-based wiki health check + optional LLM-driven
//! contradiction detection.
//!
//! Two layers:
//!
//! 1. **Rule-based** (no LLM, always on): stale episodic pages
//!    (>30d old with zero accesses), pages with empty bodies,
//!    duplicate-by-title across paths.
//! 2. **LLM-driven** (opt-in via the provider): clusters the latest
//!    semantic pages, feeds them to the LLM with a structured-output
//!    prompt asking for contradictions / stale claims.
//!
//! Findings are written to `wiki/_lint/<YYYY-MM-DD>.md` so they're
//! grep-able and tracked in git.

/// System prompt for the contradiction-detection lint pass. Loaded
/// at compile time from `prompts/lint_system.md`.
const LINT_SYSTEM_PROMPT: &str = include_str!("../prompts/lint_system.md");

use ai_memory_core::{PagePath, ProjectId, Tier, WorkspaceId};
use ai_memory_llm::{ChatMessage, ChatRequest, LlmProvider, Role, complete_structured};
use ai_memory_store::{DecayCandidate, ReaderPool};
use ai_memory_wiki::{Wiki, WritePageRequest};
use jiff::Timestamp;
use jiff::tz::TimeZone;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::warn;

/// One lint finding (rule-based or LLM-emitted).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LintFinding {
    /// Discriminator: `contradiction` | `stale` | `duplicate` | `empty` | `other`.
    pub kind: String,
    /// `info` | `warning` | `error`.
    pub severity: String,
    /// One-paragraph description.
    pub message: String,
    /// Wiki paths the finding refers to.
    #[serde(default)]
    pub pages: Vec<String>,
}

/// Structured output the LLM produces.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LintReport {
    /// Findings the LLM identified.
    pub findings: Vec<LintFinding>,
}

/// Errors raised by the lint pass.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LintError {
    /// Underlying store error.
    #[error(transparent)]
    Store(#[from] ai_memory_store::StoreError),
    /// Underlying wiki error.
    #[error(transparent)]
    Wiki(#[from] ai_memory_wiki::WikiError),
    /// Underlying LLM error.
    #[error(transparent)]
    Llm(#[from] ai_memory_llm::LlmError),
    /// Domain error (e.g. invalid page path).
    #[error(transparent)]
    Memory(#[from] ai_memory_core::MemoryError),
}

const US_PER_DAY: f64 = 86_400_000_000.0;
/// Cap on pages fed to the LLM contradiction pass (token budget).
pub const LLM_CLUSTER_CAP: usize = 20;
/// Stale threshold: an unused page this many days old is flagged.
pub const STALE_DAYS: f64 = 30.0;

/// Run the lint pass.
///
/// * `llm` — when `Some`, the contradiction pass runs; otherwise the
///   report contains only rule-based findings.
/// * `dry_run` — when `true`, no file is written.
///
/// # Errors
/// Returns [`LintError`] for any store / wiki / LLM failure.
pub async fn run_lint(
    reader: &ReaderPool,
    wiki: &Wiki,
    llm: Option<&std::sync::Arc<dyn LlmProvider>>,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    dry_run: bool,
) -> Result<LintReport, LintError> {
    let candidates = reader.decay_candidates(workspace_id, project_id).await?;
    let mut findings = rule_based_findings(&candidates);

    if let Some(provider) = llm {
        match contradiction_pass(provider.clone(), wiki, &candidates).await {
            Ok(mut extra) => findings.append(&mut extra),
            Err(e) => warn!(error = %e, "lint LLM contradiction pass failed"),
        }
    }

    let report = LintReport { findings };

    if !dry_run && !report.findings.is_empty() {
        write_report_page(wiki, workspace_id, project_id, &report).await?;
    }

    Ok(report)
}

fn rule_based_findings(candidates: &[DecayCandidate]) -> Vec<LintFinding> {
    let now_us = Timestamp::now().as_microsecond();
    let mut out = Vec::new();
    let mut titles: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for c in candidates {
        // Stale: episodic, >30d, zero accesses.
        #[allow(clippy::cast_precision_loss)]
        let age_days = (now_us - c.updated_at_us) as f64 / US_PER_DAY;
        if c.tier == Tier::Episodic && age_days > STALE_DAYS && c.access_count == 0 {
            out.push(LintFinding {
                kind: "stale".into(),
                severity: "info".into(),
                message: format!(
                    "Episodic page {} is {:.0} days old with zero accesses",
                    c.path, age_days,
                ),
                pages: vec![c.path.as_str().to_string()],
            });
        }
        // M20: rule-shaped pages get a "consider adding to
        // CLAUDE.md" suggestion. Two signals are checked:
        //   1. Frontmatter `kind: rule` — set by the consolidator
        //      when it classifies an observation as a rule.
        //   2. Page path starts with `_rules/` — same routing
        //      target. Either signal suffices.
        let frontmatter: Option<serde_json::Value> = serde_json::from_str(&c.frontmatter_json).ok();
        let kind_is_rule = frontmatter
            .as_ref()
            .and_then(|fm| fm.get("kind"))
            .and_then(serde_json::Value::as_str)
            == Some("rule");
        let path_str = c.path.as_str();
        let path_is_rule = path_str.starts_with("_rules/");
        if kind_is_rule || path_is_rule {
            out.push(LintFinding {
                kind: "rule_suggestion".into(),
                severity: "info".into(),
                message: format!(
                    "Page {path_str} looks like a durable project rule. \
                     Consider copying it into your project's CLAUDE.md / \
                     AGENTS.md so the agent sees it on every turn, not \
                     just when it remembers to call memory_query."
                ),
                pages: vec![path_str.to_string()],
            });
        }
        // Duplicate-title tracking: peek the frontmatter for a `title` field.
        if let Some(fm) = frontmatter.as_ref()
            && let Some(t) = fm.get("title").and_then(serde_json::Value::as_str)
        {
            titles
                .entry(t.to_lowercase())
                .or_default()
                .push(c.path.as_str().to_string());
        }
    }

    for (title, paths) in titles {
        if paths.len() > 1 {
            out.push(LintFinding {
                kind: "duplicate".into(),
                severity: "warning".into(),
                message: format!("Multiple pages share title {title:?}"),
                pages: paths,
            });
        }
    }

    out
}

async fn contradiction_pass(
    provider: std::sync::Arc<dyn LlmProvider>,
    wiki: &Wiki,
    candidates: &[DecayCandidate],
) -> Result<Vec<LintFinding>, LintError> {
    // Focus on semantic / procedural pages — those are the ones the
    // user actually compounds knowledge on.
    let mut subset: Vec<&DecayCandidate> = candidates
        .iter()
        .filter(|c| matches!(c.tier, Tier::Semantic | Tier::Procedural))
        .collect();
    if subset.len() < 2 {
        return Ok(Vec::new());
    }
    // Prefer high-access pages so the LLM sees the canonical knowledge.
    subset.sort_by_key(|c| std::cmp::Reverse(c.access_count));
    subset.truncate(LLM_CLUSTER_CAP);

    let mut prompt = String::new();
    prompt.push_str(
        "Audit the following wiki pages for contradictions, stale claims, or \
         duplicate information. Return a LintReport with one finding per issue.\n\n",
    );
    for c in &subset {
        let preview = wiki
            .read_page(&c.path)
            .map(|md| md.body.chars().take(400).collect::<String>())
            .unwrap_or_else(|_| "(unable to read)".into());
        prompt.push_str(&format!("## `{}`\n\n{}\n\n---\n\n", c.path, preview));
    }

    let request = ChatRequest {
        system: Some(LINT_SYSTEM_PROMPT.into()),
        messages: vec![ChatMessage {
            role: Role::User,
            content: prompt,
        }],
        // Generous output budget so multi-finding reports don't
        // truncate mid-JSON. Same rationale as consolidator/bootstrap.
        max_tokens: 32_000,
        temperature: Some(0.1),
    };
    let report: LintReport = complete_structured(&*provider, request).await?;
    Ok(report.findings)
}

async fn write_report_page(
    wiki: &Wiki,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    report: &LintReport,
) -> Result<(), LintError> {
    let date = Timestamp::now()
        .to_zoned(TimeZone::UTC)
        .strftime("%Y-%m-%d")
        .to_string();
    let path = PagePath::new(format!("_lint/{date}.md"))?;
    let title = format!("Lint report {date}");
    let body = render_markdown(report);
    wiki.write_page(WritePageRequest {
        workspace_id,
        project_id,
        path,
        frontmatter: serde_json::json!({
            "title": title.clone(),
            "tier": "semantic",
            "kind": "lint-report",
        }),
        body,
        tier: Tier::Semantic,
        pinned: false,
        title: Some(title),
    })
    .await?;
    Ok(())
}

fn render_markdown(report: &LintReport) -> String {
    let mut buf = String::new();
    buf.push_str("# Lint findings\n\n");
    if report.findings.is_empty() {
        buf.push_str("_No findings._\n");
        return buf;
    }
    buf.push_str(&format!("{} finding(s).\n\n", report.findings.len()));
    for (i, f) in report.findings.iter().enumerate() {
        buf.push_str(&format!("## {} — {} ({})\n\n", i + 1, f.kind, f.severity));
        buf.push_str(&format!("{}\n\n", f.message));
        if !f.pages.is_empty() {
            buf.push_str("Pages:\n");
            for p in &f.pages {
                buf.push_str(&format!("- `{p}`\n"));
            }
            buf.push('\n');
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_pass_flags_stale_episodic() {
        let very_old = Timestamp::now().as_microsecond() - (90 * 86_400_000_000i64);
        let candidates = vec![DecayCandidate {
            id: ai_memory_core::PageId::new(),
            path: ai_memory_core::PagePath::new("sessions/old.md").unwrap(),
            tier: Tier::Episodic,
            pinned: false,
            updated_at_us: very_old,
            access_count: 0,
            last_accessed_at_us: None,
            frontmatter_json: "{}".into(),
        }];
        let findings = rule_based_findings(&candidates);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, "stale");
    }

    #[test]
    fn rule_pass_flags_duplicate_titles() {
        let a = DecayCandidate {
            id: ai_memory_core::PageId::new(),
            path: ai_memory_core::PagePath::new("concepts/a.md").unwrap(),
            tier: Tier::Semantic,
            pinned: false,
            updated_at_us: Timestamp::now().as_microsecond(),
            access_count: 0,
            last_accessed_at_us: None,
            frontmatter_json: r#"{"title": "Karpathy Wiki"}"#.into(),
        };
        let b = DecayCandidate {
            path: ai_memory_core::PagePath::new("concepts/b.md").unwrap(),
            ..a.clone()
        };
        let findings = rule_based_findings(&[a, b]);
        let dupes: Vec<_> = findings.iter().filter(|f| f.kind == "duplicate").collect();
        assert_eq!(dupes.len(), 1);
        assert_eq!(dupes[0].pages.len(), 2);
    }

    /// M20: a page tagged `kind: rule` in its frontmatter triggers
    /// a rule_suggestion finding pointing the user at CLAUDE.md.
    #[test]
    fn rule_pass_flags_rule_kind_frontmatter() {
        let candidate = DecayCandidate {
            id: ai_memory_core::PageId::new(),
            path: ai_memory_core::PagePath::new("concepts/no-impl-without-test.md").unwrap(),
            tier: Tier::Semantic,
            pinned: false,
            updated_at_us: Timestamp::now().as_microsecond(),
            access_count: 0,
            last_accessed_at_us: None,
            frontmatter_json: r#"{"title": "Never ship code without a test", "kind": "rule"}"#
                .into(),
        };
        let findings = rule_based_findings(&[candidate]);
        let rules: Vec<_> = findings
            .iter()
            .filter(|f| f.kind == "rule_suggestion")
            .collect();
        assert_eq!(rules.len(), 1, "expected one rule_suggestion finding");
        assert!(rules[0].message.contains("CLAUDE.md"));
    }

    /// M20: a page under `_rules/` also triggers the suggestion
    /// even when the frontmatter is missing/empty — the path
    /// itself is enough signal.
    #[test]
    fn rule_pass_flags_rules_path() {
        let candidate = DecayCandidate {
            id: ai_memory_core::PageId::new(),
            path: ai_memory_core::PagePath::new("_rules/no-impl-without-test.md").unwrap(),
            tier: Tier::Semantic,
            pinned: false,
            updated_at_us: Timestamp::now().as_microsecond(),
            access_count: 0,
            last_accessed_at_us: None,
            frontmatter_json: "{}".into(),
        };
        let findings = rule_based_findings(&[candidate]);
        assert!(
            findings.iter().any(|f| f.kind == "rule_suggestion"),
            "expected a rule_suggestion finding for _rules/ page",
        );
    }

    /// Defensive: a normal concept page (no rule signal) does NOT
    /// emit the suggestion. Without this guard, every fact-tagged
    /// page would noise up the lint report.
    #[test]
    fn rule_pass_skips_non_rule_pages() {
        let candidate = DecayCandidate {
            id: ai_memory_core::PageId::new(),
            path: ai_memory_core::PagePath::new("concepts/karpathy-wiki.md").unwrap(),
            tier: Tier::Semantic,
            pinned: false,
            updated_at_us: Timestamp::now().as_microsecond(),
            access_count: 5,
            last_accessed_at_us: None,
            frontmatter_json: r#"{"title": "Karpathy Wiki", "kind": "fact"}"#.into(),
        };
        let findings = rule_based_findings(&[candidate]);
        assert!(
            findings.iter().all(|f| f.kind != "rule_suggestion"),
            "non-rule page must not produce a rule_suggestion finding",
        );
    }
}
