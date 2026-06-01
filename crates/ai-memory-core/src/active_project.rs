//! A process-shared pointer to the project the user is currently active in.
//!
//! ## Why this exists (issue #2)
//!
//! The MCP protocol carries no working-directory context: a `memory_query`
//! call arrives with its arguments and nothing else, so a tool handler has
//! no way to know which project the agent is sitting in. The lifecycle hooks
//! *do* know — every `/hook` event carries the agent's `cwd`, and the hook
//! router resolves it to the correct per-cwd `(workspace_id, project_id)`.
//!
//! In HTTP mode the `/hook` ingress and the `/mcp` endpoint live in the same
//! process, so the hook router can publish "the project the user is currently
//! active in" to this shared pointer, and the MCP tools can read it as their
//! default instead of falling back to the server's static `--project` (which
//! defaults to `scratch` and made the read tools return empty memory even
//! when the hooks were correctly populating a real project).
//!
//! The pointer reflects the most recently resolved cwd-based project. For the
//! common single-user, one-project-at-a-time workflow this is exactly right.
//! For the rarer case of one shared server fielding several projects
//! concurrently, the MCP tools also accept an explicit `project` argument that
//! takes precedence over this pointer.
//!
//! The lock is held only for the microseconds it takes to copy a small tuple;
//! callers never `.await` while holding it, so a plain `std::sync::RwLock` is
//! the right primitive (no async lock needed).

use std::sync::{Arc, RwLock};

use crate::ids::{ProjectId, WorkspaceId};

/// Cheap, cloneable handle to the shared "currently active project" slot.
///
/// Clones share the same underlying slot. Starts empty; the hook router
/// fills it as events arrive.
#[derive(Clone, Default)]
pub struct ActiveProject {
    inner: Arc<RwLock<Option<(WorkspaceId, ProjectId)>>>,
}

impl ActiveProject {
    /// Create an empty pointer (no project resolved yet).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish the project the agent is currently active in. Called by the
    /// hook router after it resolves an event's `cwd` to a real project.
    pub fn set(&self, workspace_id: WorkspaceId, project_id: ProjectId) {
        // A poisoned lock means a writer panicked mid-update. The slot holds
        // only a Copy tuple, so there's no torn state to recover — recover the
        // guard and overwrite rather than propagating the panic into a hook
        // handler that must stay fire-and-forget.
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        *guard = Some((workspace_id, project_id));
    }

    /// Read the currently active project, if any has been published yet.
    #[must_use]
    pub fn get(&self) -> Option<(WorkspaceId, ProjectId)> {
        let guard = self.inner.read().unwrap_or_else(|e| e.into_inner());
        *guard
    }

    /// Forget the active project. Called after an admin operation invalidates
    /// the published pointer (e.g. a `move-project` whose copy-purge path gives
    /// the project a NEW id, so the old pointer no longer resolves).
    pub fn clear(&self) {
        let mut guard = self.inner.write().unwrap_or_else(|e| e.into_inner());
        *guard = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        assert!(ActiveProject::new().get().is_none());
    }

    #[test]
    fn set_then_get_round_trips() {
        let ap = ActiveProject::new();
        let ws = WorkspaceId::new();
        let proj = ProjectId::new();
        ap.set(ws, proj);
        assert_eq!(ap.get(), Some((ws, proj)));
    }

    #[test]
    fn set_overwrites_previous() {
        let ap = ActiveProject::new();
        ap.set(WorkspaceId::new(), ProjectId::new());
        let ws2 = WorkspaceId::new();
        let proj2 = ProjectId::new();
        ap.set(ws2, proj2);
        assert_eq!(ap.get(), Some((ws2, proj2)));
    }

    #[test]
    fn clones_share_one_slot() {
        let a = ActiveProject::new();
        let b = a.clone();
        let ws = WorkspaceId::new();
        let proj = ProjectId::new();
        a.set(ws, proj);
        assert_eq!(b.get(), Some((ws, proj)), "clone must see the same slot");
    }
}
