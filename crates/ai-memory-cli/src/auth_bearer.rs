//! Bearer-token resolution shared by lifecycle hooks and thin HTTP clients.
//!
//! Static CLI/config tokens always win. When they are absent, a stored OIDC
//! device-flow token is loaded from `auth.json`, refreshed if stale, and used
//! as the bearer for server HTTP calls.

use std::path::Path;

use ai_memory_llm::{OidcToken, refresh_access_token};
use secrecy::ExposeSecret as _;

/// Resolve the bearer for one server request: explicit/static token first,
/// stored OIDC token second, and no token last.
pub async fn resolve_bearer(
    client: &reqwest::Client,
    auth_path: &Path,
    static_token: Option<&str>,
) -> Option<String> {
    match static_token.filter(|t| !t.is_empty()) {
        Some(t) => Some(t.to_string()),
        None => resolve_oidc(client, auth_path).await,
    }
}

/// Load the stored OIDC token, refreshing and persisting it when stale.
///
/// Returns the access token, or `None` when there is no token or refresh
/// failed. Failing open to "no bearer" preserves the old unauthenticated CLI
/// behavior and lets the server return the authoritative auth error.
pub async fn resolve_oidc(client: &reqwest::Client, auth_path: &Path) -> Option<String> {
    let mut token = OidcToken::load(auth_path).ok().flatten()?;
    if token.needs_refresh() {
        let Ok(refreshed) = refresh_access_token(client, &token).await else {
            return None;
        };
        let _ = refreshed.save(auth_path);
        token = refreshed;
    }
    Some(token.access.expose_secret().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    use secrecy::SecretString;

    fn save_oidc_token(path: &Path, access: &str) {
        let token = OidcToken {
            access: SecretString::from(access.to_string()),
            refresh: SecretString::from("refresh-token".to_string()),
            expires_at_ms: u64::MAX,
            issuer: "https://issuer.example.com/realms/team".to_string(),
            client_id: "ai-memory-cli".to_string(),
            token_endpoint: "https://issuer.example.com/token".to_string(),
        };
        token.save(path).expect("save test OIDC token");
    }

    #[tokio::test]
    async fn static_token_wins_over_stored_oidc() {
        let tmp = tempfile::tempdir().unwrap();
        let auth_path = tmp.path().join("auth.json");
        save_oidc_token(&auth_path, "oidc-access");
        let client = reqwest::Client::new();

        let bearer = resolve_bearer(&client, &auth_path, Some("static-token")).await;

        assert_eq!(bearer.as_deref(), Some("static-token"));
    }

    #[tokio::test]
    async fn stored_oidc_is_used_when_static_token_is_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let auth_path = tmp.path().join("auth.json");
        save_oidc_token(&auth_path, "oidc-access");
        let client = reqwest::Client::new();

        let bearer = resolve_bearer(&client, &auth_path, None).await;

        assert_eq!(bearer.as_deref(), Some("oidc-access"));
    }

    #[tokio::test]
    async fn empty_when_static_and_oidc_are_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let auth_path = tmp.path().join("auth.json");
        let client = reqwest::Client::new();

        let bearer = resolve_bearer(&client, &auth_path, None).await;

        assert!(bearer.is_none());
    }
}
