//! Startup warning for proxy variables the HTTP client cannot use.
//!
//! reqwest reads `HTTPS_PROXY` and its siblings through hyper-util, which
//! skips a value that is not UTF-8 or does not parse as a URI and carries on
//! without a proxy. Provider, OIDC and upgrade requests then go out directly:
//! a timeout on a network that blocks direct egress, or traffic that bypasses
//! a required proxy on one that does not. The checks below mirror
//! hyper-util 0.1's acceptance rules (`parse_env_uri`) so the operator hears
//! about it; the value itself is never logged, since it often carries a
//! password.

use std::ffi::OsStr;

use crate::config::RuntimeEnv;

/// Variables hyper-util consults, in its lookup order.
const PROXY_VARS: [&str; 6] = [
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
];

/// Log one warning per proxy variable that is set but unusable.
pub(crate) fn warn_unusable(env: &RuntimeEnv) {
    for (name, reason) in unusable(|name| env.var_os(name)) {
        tracing::warn!(
            variable = name,
            "{name} {reason}, so the HTTP client ignores it and connects without that proxy"
        );
    }
}

fn unusable(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Vec<(&'static str, &'static str)> {
    PROXY_VARS
        .iter()
        // Windows matches variable names case-insensitively, so the lower-case
        // spelling is the same variable there.
        .filter(|name| !(cfg!(windows) && name.bytes().any(|b| b.is_ascii_lowercase())))
        .filter_map(|&name| Some((name, problem(&lookup(name)?)?)))
        .collect()
}

fn problem(value: &OsStr) -> Option<&'static str> {
    let Some(text) = value.to_str() else {
        return Some("is not valid UTF-8");
    };
    // hyper-util treats an empty value as unset, and so does every shell user.
    if text.is_empty() {
        return None;
    }
    let Ok(uri) = text.parse::<http::Uri>() else {
        return Some(
            "is not a valid URI (percent-encode spaces and non-ASCII characters in \
             the user name and password)",
        );
    };
    match uri.scheme_str() {
        None | Some("http" | "https" | "socks4" | "socks4a" | "socks5" | "socks5h") => {}
        Some(_) => return Some("uses an unsupported proxy scheme"),
    }
    if uri.authority().is_none() {
        return Some("names no proxy host");
    }
    None
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    fn only(name: &'static str, value: OsString) -> impl Fn(&str) -> Option<OsString> {
        move |asked| (asked == name).then(|| value.clone())
    }

    #[test]
    fn usable_values_raise_no_warning() {
        for value in [
            "http://proxy.example:3128",
            "http://user:p%C3%A7ss@proxy.example:3128",
            "https://proxy.example",
            "proxy.example:3128",
            "socks5h://127.0.0.1:1080",
            "",
        ] {
            assert!(
                unusable(only("HTTPS_PROXY", value.into())).is_empty(),
                "{value:?} was reported as unusable"
            );
        }
    }

    #[test]
    fn values_the_client_skips_are_reported_by_name() {
        for value in [
            "http://user:p\u{e7}ss@proxy.example:3128",
            "http://user:p ss@proxy.example:3128",
            " http://proxy.example:3128",
            "ftp://proxy.example:21",
        ] {
            let found = unusable(only("HTTPS_PROXY", value.into()));
            assert_eq!(found.len(), 1, "{value:?} was not reported");
            assert_eq!(found[0].0, "HTTPS_PROXY");
        }
    }

    /// A Latin-1 password byte (`ç` is 0xE7) makes the whole value non-UTF-8.
    #[cfg(unix)]
    #[test]
    fn a_non_utf8_value_is_reported_without_its_bytes() {
        use std::os::unix::ffi::OsStringExt as _;
        let value = OsString::from_vec(b"http://user:a\xe7\xe3o@proxy.example:3128".to_vec());
        let found = unusable(only("http_proxy", value));
        assert_eq!(found, vec![("http_proxy", "is not valid UTF-8")]);
        assert!(!found[0].1.contains("proxy.example"));
    }

    #[test]
    fn unset_variables_raise_no_warning() {
        assert!(unusable(|_| None).is_empty());
    }
}
