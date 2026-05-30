//! Core domain types and errors for ai-memory.
//!
//! This crate is the closure of the project's vocabulary: identifiers, agent
//! kinds, the workspace-wide error type, and the privacy strip (which is
//! pure-compute, no IO). Nothing in here performs I/O, which keeps it
//! trivially unit-testable and free of platform concerns.

pub mod active_project;
pub mod actor;
pub mod error;
pub mod handoff;
pub mod ids;
pub mod observation;
pub mod page;
pub mod routing_snippet;
pub mod sanitize;
pub mod user;

/// Default workspace name used by the single-workspace v1 flow.
pub const DEFAULT_WORKSPACE_NAME: &str = "default";

/// Defensive project fallback used only when no cwd/project is available.
pub const DEFAULT_PROJECT_NAME: &str = "scratch";

pub use active_project::ActiveProject;
pub use actor::{ActorContext, AuthLevel};
pub use error::{MemoryError, MemoryResult};
pub use handoff::{Handoff, HandoffState, NewHandoff};
pub use ids::{
    AgentKind, HandoffId, ObservationId, PageId, PagePath, ProjectId, SessionId, UserId,
    WorkspaceId,
};
pub use observation::{NewObservation, NewSession, Observation, ObservationKind};
pub use page::{LinkTarget, NewPage, Page, Tier};
pub use routing_snippet::{MARKER_END, MARKER_START, SNIPPET_BODY, full_block};
pub use sanitize::{SanitizeConfig, Sanitized, Sanitizer};
pub use user::{MAX_EMAIL_LEN, MAX_USERNAME_LEN, NewUser, User, validate_email, validate_username};
