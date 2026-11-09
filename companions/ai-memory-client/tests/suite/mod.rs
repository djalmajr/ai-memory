use super::*;
use serde_json::json;

#[test]
fn bounded_body_schema_refuses_authority_depth_and_size_with_control() {
    let control = json!({"session_id":"sessão-界", "cwd":"/work", "_ai_memory_capture":{"actor":"untrusted provenance"}});
    check_body(&control).unwrap();
    assert!(check_body(&json!([control.clone()])).is_err());
    for field in ["workspace", "project", "actor", "author_id", "headers"] {
        let mut attack = control.clone();
        attack[field] = json!("forged");
        assert!(
            check_body(&attack).is_err(),
            "body authority must be refused"
        );
    }
    let mut deep = control.clone();
    for _ in 0..65 {
        deep = json!({"nested": deep});
    }
    assert!(check_body(&deep).is_err(), "deep bodies must be refused");
    let large = json!({"first":"x".repeat(140_000),"second":"x".repeat(140_000)});
    assert!(
        check_body(&large).is_err(),
        "body byte cap must be enforced"
    );
    let mut large_text = json!("x".repeat(256 * 1024 + 1));
    assert!(sanitize_external_value(&mut large_text).is_err());
    check_body(&control).unwrap();
}

#[test]
fn complete_literal_redaction_markers_survive_repeated_scrubbing() {
    for marker in [
        "[REDACTED]",
        "[REDACTED CREDENTIAL]",
        "[REDACTED PRIVATE KEY]",
    ] {
        let control = format!("界 token={marker} literal {marker}");
        assert_eq!(sanitize_external_text(&control), control);
        assert_eq!(
            sanitize_external_text(&sanitize_external_text(&control)),
            control
        );
    }
}

#[test]
fn secrets_are_rejected_in_all_envelope_positions_without_exposing_them() {
    for secret in [
        "Bearer abcdefghijklmnop",
        "github_pat_1234567890abcdefghijklmnop",
        "password=abcdefghi",
        "unsafe\u{1b}text",
    ] {
        assert_ne!(sanitize_external_text(secret), secret);
        for envelope in [
            json!(secret),
            json!({"session_id": secret}),
            json!({"cwd": secret}),
            json!({"source": secret}),
            json!({"metadata": [secret]}),
        ] {
            let error = reject_sensitive(&envelope).unwrap_err();
            assert!(!error.to_string().contains(secret));
        }
    }
    assert!(reject_sensitive(&json!({"nested": {"api_key": "short"}})).is_err());
    assert!(
        reject_sensitive(
            &json!({"session_id": "native", "cwd": "/work", "prompt": "ordinary text"})
        )
        .is_ok()
    );
}

#[test]
fn redacted_nested_values_validate_idempotently() {
    let body = json!({"session_id": "native", "cwd": "/work", "nested": {"token": "[REDACTED]", "api-key": "[REDACTED]"}});
    reject_sensitive(&body).unwrap();
    check_body(&body).unwrap();
    let mut again = body.clone();
    sanitize_external_value(&mut again).unwrap();
    assert_eq!(again, body);
}

#[test]
fn sanitation_keeps_native_identity_and_refuses_body_authority() {
    let mut body = json!({"session_id":"native", "cwd":"/work", "output":"\u{1b}[32m token=getToken() Bearer abcdefghijklmnop", "nested":{"password":"process.env.X"}});
    sanitize_external_value(&mut body).unwrap();
    assert_eq!(body["session_id"], "native");
    assert_eq!(body["cwd"], "/work");
    assert_eq!(body["nested"]["password"], "[REDACTED]");
    assert!(!body["output"].as_str().unwrap().contains(['\u{1b}']));
    check_body(&body).unwrap();
    for field in ["workspace", "project", "actor", "author_id", "headers"] {
        let mut forged = body.clone();
        forged[field] = "foreign".into();
        assert!(check_body(&forged).is_err());
    }
}

#[test]
fn sensitive_key_forms_redact_opaque_values_and_preserve_numeric_metrics() {
    let keys = [
        "accessToken",
        "refreshToken",
        "clientSecret",
        "privateKey",
        "x-api-key",
        "OPENAI_API_KEY",
        "accessKey",
        "HTTPAuthorization",
        "cookie",
        "sessionCookies",
        "credential",
        "credentials",
        "password",
        "token_count",
        "token_usage",
        "max_tokens",
        "input_tokens",
    ];
    for key in keys {
        let raw = json!({"nested": [{key: "fixture-value"}]});
        assert_eq!(sanitize_external_text("fixture-value"), "fixture-value");
        assert!(reject_sensitive(&raw).is_err());
        let mut safe = raw;
        sanitize_external_value(&mut safe).unwrap();
        assert_eq!(safe["nested"][0][key], "[REDACTED]");
        reject_sensitive(&safe).unwrap();
        let once = safe.clone();
        sanitize_external_value(&mut safe).unwrap();
        assert_eq!(safe, once);
    }
    let metrics = json!({"max_tokens": 12, "input_tokens": 123456789012345_u64, "token_count": 0, "token_usage": 1.25});
    let mut safe = metrics.clone();
    sanitize_external_value(&mut safe).unwrap();
    assert_eq!(safe, metrics);
    assert_eq!(
        serde_json::to_string(&safe).unwrap(),
        serde_json::to_string(&metrics).unwrap()
    );
    reject_sensitive(&safe).unwrap();
}

#[test]
fn unlisted_numeric_token_counts_survive_while_their_string_form_redacts() {
    let counts = json!({"cached_tokens": 7, "thinking_tokens": 1024, "cacheTokens": 3});
    let mut safe = counts.clone();
    sanitize_external_value(&mut safe).unwrap();
    assert_eq!(
        safe, counts,
        "numeric usage counts keep their value and type"
    );
    reject_sensitive(&safe).unwrap();

    let mut strings = json!({"cached_tokens": "fixture-value", "thinking_tokens": "fixture-value"});
    assert!(reject_sensitive(&strings).is_err());
    sanitize_external_value(&mut strings).unwrap();
    assert_eq!(strings["cached_tokens"], "[REDACTED]");
    assert_eq!(strings["thinking_tokens"], "[REDACTED]");
}

#[test]
fn numeric_token_metrics_preserve_representation_without_credential_bypass() {
    let metrics = json!({"max_tokens":12,"input_tokens":34,"token_count":0,"token_usage":1.25,"output_tokens":9007199254740993_u64,"total_tokens":89,"totalTokens":89,"cache_read_input_tokens":2,"cache_creation_input_tokens":3,"prompt_tokens":5,"completion_tokens":7,"reasoning_tokens":11});
    let raw = json!({"usage":metrics,"nested":[metrics.clone()]});
    let mut safe = raw.clone();
    sanitize_external_value(&mut safe).unwrap();
    assert!(
        safe == raw,
        "numeric metric values and types must survive sanitation"
    );
    assert!(serde_json::to_vec(&safe).unwrap() == serde_json::to_vec(&raw).unwrap());
    reject_sensitive(&safe).unwrap();
    let once = safe.clone();
    sanitize_external_value(&mut safe).unwrap();
    assert!(safe == once);
    for key in metrics.as_object().unwrap().keys() {
        let mut opaque = json!({key:"fixture-value"});
        sanitize_external_value(&mut opaque).unwrap();
        assert!(
            opaque[key] == "[REDACTED]",
            "opaque metric values must be redacted"
        );
    }
    for key in [
        "token",
        "access_tokens",
        "private_token_count",
        "auth_token_usage",
        "authorization_token_limit",
        "private_key",
        "accessKey",
    ] {
        let mut credential = json!({key:123456});
        sanitize_external_value(&mut credential).unwrap();
        assert!(
            credential[key] == "[REDACTED]",
            "numeric credential values must be redacted"
        );
    }
    let mut usage_object = json!({"token_usage":{"input_tokens":12,"output_tokens":34}});
    sanitize_external_value(&mut usage_object).unwrap();
    assert!(usage_object["token_usage"] == "[REDACTED]");
}

#[test]
fn common_credential_components_redact_opaque_values_with_author_controls() {
    for key in [
        "auth",
        "jwt",
        "pwd",
        "bearer",
        "session_key",
        "signature",
        "clientAuth",
        "HTTP_JWT",
        "HTTPJwt",
        "JWT",
        "client-pwd",
        "Bearer",
        "sessionKey",
        "Session-Key",
        "requestSignature",
    ] {
        for value in [json!("fixture-value"), json!(123456)] {
            let raw = json!({"nested":[{key:value}]});
            assert!(
                reject_sensitive(&raw).is_err(),
                "raw credential component must be refused"
            );
            let mut safe = raw;
            sanitize_external_value(&mut safe).unwrap();
            assert!(safe["nested"][0][key] == "[REDACTED]");
            reject_sensitive(&safe).unwrap();
            let once = safe.clone();
            sanitize_external_value(&mut safe).unwrap();
            assert!(safe == once);
        }
    }
    let control =
        json!({"author":"fixture-value","authority":"fixture-value","coauthor":"fixture-value"});
    assert!(sanitize_external_text("fixture-value") == "fixture-value");
    let mut safe = control.clone();
    sanitize_external_value(&mut safe).unwrap();
    assert!(
        safe == control,
        "author and authority are legitimate components"
    );
    reject_sensitive(&safe).unwrap();
}

#[test]
fn session_key_pairs_are_sensitive_with_native_session_metadata_control() {
    for key in [
        "session_key",
        "sessionKey",
        "Session-Key",
        "HTTP_SESSION_KEY",
    ] {
        for value in [json!("fixture-value"), json!(123456)] {
            let raw = json!({"nested":[{key:value}]});
            assert!(reject_sensitive(&raw).is_err());
            let mut safe = raw;
            sanitize_external_value(&mut safe).unwrap();
            assert!(safe["nested"][0][key] == "[REDACTED]");
            reject_sensitive(&safe).unwrap();
            let once = safe.clone();
            sanitize_external_value(&mut safe).unwrap();
            assert!(safe == once);
        }
    }
    let control = json!({"session":"fixture-value","session_id":"native","key":"fixture-value"});
    let mut safe = control.clone();
    sanitize_external_value(&mut safe).unwrap();
    assert!(safe == control);
    reject_sensitive(&safe).unwrap();
}

#[test]
fn plural_credential_key_pairs_redact_opaque_values_recursively() {
    assert!(sanitize_external_text("fixture-value") == "fixture-value");
    for key in [
        "api_keys",
        "access_keys",
        "private_keys",
        "session_keys",
        "apiKeys",
        "accessKeys",
        "privateKeys",
        "sessionKeys",
        "API_KEYS",
        "session-keys",
    ] {
        for value in [
            json!("fixture-value"),
            json!(["fixture-value", {"label":"opaque-value"}]),
        ] {
            let raw = json!({"nested":[{"details":{key:value}}]});
            assert!(
                reject_sensitive(&raw).is_err(),
                "plural key pair must be refused before sanitation"
            );
            let mut safe = raw;
            sanitize_external_value(&mut safe).unwrap();
            assert!(safe["nested"][0]["details"][key] == "[REDACTED]");
            check_body(&safe).unwrap();
            let once = safe.clone();
            sanitize_external_value(&mut safe).unwrap();
            assert!(safe == once);
        }
    }
}

#[test]
fn plural_key_guard_preserves_lone_keys_native_ids_and_metrics() {
    let control = json!({
        "key":"fixture-value", "keys":["fixture-value"],
        "session":"fixture-value", "session_id":"native-session", "event_id":"native-event",
        "author":"fixture-value", "authority":"fixture-value",
        "metrics":{"max_tokens":12,"input_tokens":34,"token_count":0,"token_usage":1.25,"output_tokens":9007199254740993_u64,"total_tokens":89,"totalTokens":89,"cache_read_input_tokens":2,"cache_creation_input_tokens":3,"prompt_tokens":5,"completion_tokens":7,"reasoning_tokens":11}
    });
    let bytes = serde_json::to_vec(&control).unwrap();
    check_body(&control).unwrap();
    let mut safe = control.clone();
    sanitize_external_value(&mut safe).unwrap();
    assert!(safe == control);
    assert!(serde_json::to_vec(&safe).unwrap() == bytes);
    sanitize_external_value(&mut safe).unwrap();
    assert!(safe == control);
}

#[test]
fn redaction_marker_suffix_never_exempts_a_credential() {
    for marker in [
        "[REDACTED]",
        "[REDACTED CREDENTIAL]",
        "[REDACTED PRIVATE KEY]",
    ] {
        let control = format!("token={marker}");
        assert_eq!(sanitize_external_text(&control), control);
        for attack in [
            format!("token=abcdefghijklmnop{marker}"),
            format!("token={marker}abcdefghijklmnop"),
        ] {
            assert!(reject_sensitive(&json!(attack)).is_err());
            assert_eq!(sanitize_external_text(&attack), "token=[REDACTED]");
            let safe = sanitize_external_text(&attack);
            assert_eq!(sanitize_external_text(&safe), safe);
        }
        assert_eq!(sanitize_external_text(&control), control);
    }
}

/// The vectors the server sanitizer runs too (`ai-memory-core::sanitize`), so
/// the two copies cannot drift: credentials split by an invisible character,
/// private-key blocks with and without an end line, and text that must pass
/// through unchanged.
#[test]
fn shared_sanitizer_vectors_hold() {
    let vectors: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/sanitizer_vectors.json"
    ))
    .unwrap();
    let strings = |key: &str| -> Vec<String> {
        vectors[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect()
    };
    for secret in strings("credentials") {
        let (head, tail) = secret.split_at(secret.len() / 2);
        for invisible in strings("invisibles") {
            let out = sanitize_external_text(&format!("key {head}{invisible}{tail} end"));
            assert!(
                out.contains("[REDACTED") && !out.contains(head) && !out.contains(tail),
                "{invisible:?} inside {secret}: {out:?}"
            );
            assert!(out.starts_with("key ") && out.ends_with(" end"), "{out:?}");
        }
    }
    for field in vectors["credential_fields"].as_array().unwrap() {
        let input = field["input"].as_str().unwrap();
        let out = sanitize_external_text(input);
        assert!(
            out.contains("[REDACTED") && !out.contains(field["secret"].as_str().unwrap()),
            "{input:?} -> {out:?}"
        );
        for kept in field["kept"].as_array().into_iter().flatten() {
            assert!(out.contains(kept.as_str().unwrap()), "{input:?} -> {out:?}");
        }
    }
    let body_line = vectors["key_body_line"].as_str().unwrap();
    for block in vectors["key_blocks"].as_array().unwrap() {
        let input = block["input"].as_str().unwrap();
        let out = sanitize_external_text(input);
        assert!(
            out.contains("[REDACTED PRIVATE KEY]") && !out.contains(body_line),
            "{input:?} -> {out:?}"
        );
        for kept in block["kept"].as_array().unwrap() {
            assert!(out.contains(kept.as_str().unwrap()), "{input:?} -> {out:?}");
        }
    }
    for text in strings("unchanged") {
        assert_eq!(sanitize_external_text(&text), text);
    }
    // The structured tool-response cases the shared loop's `key … end`
    // wrap cannot express: exact input, exact output, and idempotency.
    // The core runs the same block first, so the two copies cannot drift on
    // the serialized-control spellings either. The server labels the marker
    // `[REDACTED:ai_memory_key]`; the client replaces with its own unlabeled
    // marker, so its expected output is the same vector with the label
    // swapped.
    for case in vectors["native_key_cases"].as_array().unwrap() {
        let input = case["input"].as_str().unwrap();
        let expected = case["output"]
            .as_str()
            .unwrap()
            .replace("[REDACTED:ai_memory_key]", "[REDACTED CREDENTIAL]");
        let out = sanitize_external_text(input);
        assert_eq!(out, expected, "{input:?}");
        assert_eq!(
            sanitize_external_text(&out),
            out,
            "{input:?} must be a no-op twice"
        );
    }
}

#[test]
fn characters_that_hide_text_or_split_a_credential_are_removed() {
    let hidden: String = "hidden note"
        .chars()
        .map(|c| char::from_u32(0xe0000 + u32::from(c)).unwrap())
        .collect();
    assert_eq!(
        sanitize_external_text(&format!("visible{hidden} text")),
        "visible text"
    );
    // A control character inside a credential no longer keeps it whole.
    let out = sanitize_external_text("sk-leak-012345\u{0}6789abcdef0123");
    assert_eq!(out, "[REDACTED CREDENTIAL]");
}

/// Control characters are stripped before matching, so a word glued onto a key
/// by a removed character used to keep the key whole: the `aim_`/`ams_` and
/// `AKIA` rules required a left word boundary the removal had destroyed, and
/// `amk_` had no rule (GHSA-6pmr-q79m-87vp, item 5, the client mirror). The
/// own-prefix and GitHub rules are now left-unanchored; AWS accepts a
/// percent-encoded byte.
#[test]
fn a_key_glued_onto_a_word_by_a_stripped_character_is_still_redacted() {
    let tail = "FAKEfake0123-_xy".repeat(3); // 48 base64url chars, > the 40 floor
    for prefix in ["aim_", "ams_", "amk_"] {
        let key = format!("{prefix}{tail}");
        for glued in [
            format!("title\u{7}{key}"),      // a removed BEL
            format!("title\u{1b}[32m{key}"), // a removed ANSI escape
            format!("word{key}"),            // a letter directly before it
        ] {
            let out = sanitize_external_text(&glued);
            // A stripped character leaves a word boundary, so the anchored
            // 22-character own-prefix rule redacts the first two cases with
            // its labeled marker; a bare glued letter falls through to the
            // left-unanchored 40-character rule. Either way the tail must be
            // gone.
            assert!(
                out.contains("[REDACTED") && !out.contains(&tail),
                "glued key must be redacted: {glued:?} -> {out:?}"
            );
        }
    }
    // A GitHub token glued onto a word by a removed BEL, and an AWS key behind
    // a percent-encoded byte.
    for (input, leaked) in [
        ("word\u{7}ghp_FAKE0123456789ABCDEFGHIJ0123", "ghp_FAKE"),
        ("url?k=%20AKIAFAKEFAKEFAKEFAKE", "FAKEFAKE"),
    ] {
        let out = sanitize_external_text(input);
        assert!(
            out.contains("[REDACTED CREDENTIAL]") && !out.contains(leaked),
            "credential must be redacted: {input:?} -> {out:?}"
        );
    }
}

/// An own-prefix key whose body holds a vendor fragment used to lose only
/// the fragment to the vendor rule, leaving the head of the key in clear
/// text whenever the anchored own-prefix rule could not see the key (a word
/// glued onto it, a split). The credential-set union collects every member's
/// spans over the same pre-rewrite text and redacts the overlap once, so the
/// whole key goes.
#[test]
fn an_own_prefix_key_holding_a_vendor_fragment_is_redacted_whole() {
    let head = "Z".repeat(19);
    for prefix in ["aim_", "amk_", "ams_"] {
        for fragment in [
            format!("_sk-{}", "A".repeat(20)),
            format!("_ghp_{}", "B".repeat(20)),
            // The client has no Stripe rule: this fragment matches no client
            // vendor rule, and the union must still take the whole key.
            format!("_sk_live_{}", "C".repeat(16)),
        ] {
            let key = format!("{prefix}{head}{fragment}");
            for (shape, input, expected_prefix) in [
                // Glued onto a word: the anchored rule loses its boundary.
                ("glued", format!("word{key}"), "word"),
                // A plain space: both own-prefix rules see the key.
                ("space", format!("word {key}"), "word "),
                // A removed BEL glues the word onto the key.
                ("bel", format!("word\u{7}{key}"), "word"),
                // A removed ANSI escape glues the word onto the key (the
                // visible tail of the sequence stays).
                ("ansi", format!("word\u{1b}[32m{key}"), "word"),
                // A NUL inside the prefix is removed and glues the word on.
                (
                    "nul",
                    format!(
                        "word{first}\u{0}{rest}",
                        first = &key[..1],
                        rest = &key[1..]
                    ),
                    "word",
                ),
            ] {
                let out = sanitize_external_text(&input);
                assert!(
                    out == format!("{expected_prefix}[REDACTED CREDENTIAL]")
                        || out.ends_with("[REDACTED CREDENTIAL]"),
                    "{shape} {prefix}: {input:?} -> {out:?}"
                );
                assert!(
                    !out.contains(prefix) && !out.contains(&head) && !out.contains(&fragment),
                    "{shape} {prefix}: a credential part survives in {out:?}"
                );
                // The scrubber runs before and after the value boundary: a
                // second pass must be a no-op, not a fresh fragment.
                assert_eq!(
                    sanitize_external_text(&out),
                    out,
                    "{shape} {prefix}: not idempotent: {out:?}"
                );
            }
        }
    }
    // Controls: a lone vendor key and an ordinary word keep their shape.
    let vendor = format!("use sk-{} here", "D".repeat(24));
    assert_eq!(
        sanitize_external_text(&vendor),
        "use [REDACTED CREDENTIAL] here"
    );
    let word = "the aimless stop at the gate";
    assert_eq!(sanitize_external_text(word), word);
}

/// The other direction: a vendor key whose body holds an own-prefix fragment
/// used to come out as `sk-QQQQ[REDACTED CREDENTIAL]`, the vendor head in
/// clear. The union redacts the vendor key whole. The client's `sk-` rule
/// covers the `sk-`/`sk-proj-` family and its `github_pat_` alternative
/// covers the GitHub PAT family; a token the client's own rules do not
/// recognize (a `ghp_` body with underscores) was never a match here and is
/// out of scope for the union.
#[test]
fn a_vendor_key_holding_an_own_prefix_fragment_is_redacted_whole() {
    let families = ["sk-", "sk-proj-", "github_pat_"];
    for prefix in ["aim_", "amk_", "ams_"] {
        for family in families {
            for head in ["Q".repeat(8), "Q".repeat(24)] {
                for tail in ["Z".repeat(40), "Z".repeat(43)] {
                    let input = format!("use {family}{head}{prefix}{tail} here");
                    let out = sanitize_external_text(&input);
                    assert_eq!(
                        out, "use [REDACTED CREDENTIAL] here",
                        "family {family} {prefix}: {input:?} -> {out:?}"
                    );
                    assert!(
                        !out.contains(prefix) && !out.contains(&head) && !out.contains(&tail),
                        "family {family} {prefix}: a credential part survives in {out:?}"
                    );
                    assert_eq!(sanitize_external_text(&out), out, "not idempotent: {out:?}");
                }
            }
        }
    }
}

/// The union must not reach beyond the credential set: two own-prefix keys
/// side by side stay two separate redactions, a vendor key plus a separate
/// own-prefix key keeps both, and a native key in a `Bearer` header stays on
/// the bearer marker the client uses for it.
#[test]
fn the_union_keeps_disjoint_credentials_and_the_bearer_marker() {
    let a = format!("aim_{}", "J".repeat(43));
    let b = format!("amk_{}", "K".repeat(43));
    assert_eq!(
        sanitize_external_text(&format!("{a} {b}")),
        "[REDACTED CREDENTIAL] [REDACTED CREDENTIAL]"
    );
    let vendor = format!("sk-{}", "L".repeat(24));
    assert_eq!(
        sanitize_external_text(&format!("{vendor} and {a}")),
        "[REDACTED CREDENTIAL] and [REDACTED CREDENTIAL]"
    );
    assert_eq!(
        sanitize_external_text(&format!("Authorization: Bearer {a}")),
        "Authorization: Bearer [REDACTED]"
    );
}

/// The R1 probe: an own-prefix key inside a `Bearer …` match whose body also
/// holds a `ghp_` fragment. The per-segment deferral left the own-prefix span
/// to the bearer rule while the union redacted the fragment — breaking the
/// bearer match (the marker is no bearer value) and leaving the key head and
/// tail in clear. The deferral now runs per overlapping group: a group defers
/// to the bearer only when every member of it is own-prefix and a single
/// bearer match over the same pre-union text holds the merged group, so a
/// group holding a vendor fragment is redacted whole by the union instead.
/// Nothing of the head or the tail may survive, in plain text, in JSON and
/// with U+200B inside the prefix.
#[test]
fn an_own_prefix_key_holding_a_vendor_fragment_in_a_bearer_header_is_redacted_whole() {
    let fragment = format!("ghp_{}", "A".repeat(20));
    let tail = format!("_{}A", "Z".repeat(15));
    // The R1 input's head (`Q_`) plus heads of 1, 2, 3 and 8 characters
    // before the fragment.
    let heads = ["Q_", "Q", "QQ", "QQQ", "QQQQQQQQ"];
    for prefix in ["aim_", "amk_", "ams_"] {
        for head in &heads {
            let value = format!("{prefix}{head}{fragment}{tail}");
            // U+200B inside the prefix: the plain pass cannot see the key,
            // only the view through the invisible character can.
            let split_prefix = format!("{}\u{200b}{}", &prefix[..1], &prefix[1..]);
            let zwsp_value = format!("{split_prefix}{head}{fragment}{tail}");
            for (context, input, expected) in [
                (
                    "plain",
                    format!("Authorization: Bearer {value}"),
                    "Authorization: Bearer [REDACTED CREDENTIAL]",
                ),
                (
                    "json",
                    format!(
                        "{{\"out\": \"Authorization: Bearer {value}\", \"neighbor\": \"KEEP_NEIGHBOR\"}}"
                    ),
                    "{\"out\": \"Authorization: Bearer [REDACTED CREDENTIAL]\", \"neighbor\": \"KEEP_NEIGHBOR\"}",
                ),
                (
                    "zwsp",
                    format!("Authorization: Bearer {zwsp_value}"),
                    "Authorization: Bearer [REDACTED CREDENTIAL]",
                ),
            ] {
                let out = sanitize_external_text(&input);
                assert_eq!(
                    out, expected,
                    "{context} {prefix} head {head:?}: {input:?} -> {out:?}"
                );
                let zwsp_run = "Z".repeat(15);
                for leaked in [
                    prefix,
                    *head,
                    fragment.as_str(),
                    zwsp_run.as_str(),
                    split_prefix.as_str(),
                ] {
                    assert!(
                        !out.contains(leaked),
                        "{context} {prefix} head {head:?}: a credential part survives in {out:?}"
                    );
                }
                assert_eq!(
                    sanitize_external_text(&out),
                    out,
                    "{context} {prefix} head {head:?}: not idempotent: {out:?}"
                );
            }
        }
    }
    // Control: a lone own key in a header keeps the bearer marker, so the
    // group deferral still leaves plain keys to the bearer rule.
    let lone = format!("ams_{}", "M".repeat(43));
    assert_eq!(
        sanitize_external_text(&format!("Authorization: Bearer {lone}")),
        "Authorization: Bearer [REDACTED]"
    );
}

/// The deferral's failure mode: the union defers a group to the bearer rule
/// on the pre-union text, assuming the bearer match holding the group will
/// still exist when the rule runs. Two shapes break that assumption, and
/// without the post-bearer net the deferred key — the one credential the
/// union left untouched and the own-prefix rules never run (they ride on the
/// union) — would leave the scrubber whole:
///
/// 1. A vendor group redacted inside the same bearer value: the marker is no
///    bearer value, so the match is dead and the own key after it survives.
/// 2. A zero-width space splitting one value into two bearer matches: the
///    shadow pass's first match joins the split value and consumes the
///    second `Bearer`, so the plain pass no longer sees the key's match.
///
/// The union records each deferred group's exact text and the net redacts
/// whatever of it still remains after the bearer rule's two passes, with the
/// union's own marker.
#[test]
fn a_deferred_group_is_redacted_even_when_the_bearer_match_is_gone() {
    let tail = format!("{}A", "Z".repeat(42)); // 43 base64url characters
    // Case 1: the vendor group and the own key sit in one bearer value,
    // every own prefix and every value-class separator between them.
    let fragment = format!("ghp_{}", "A".repeat(20));
    for prefix in ["aim_", "amk_", "ams_"] {
        for separator in [".", "/", "+", "=", "~"] {
            let input = format!("Authorization: Bearer {fragment}{separator}{prefix}{tail}");
            let expected = format!(
                "Authorization: Bearer [REDACTED CREDENTIAL]{separator}[REDACTED CREDENTIAL]"
            );
            let out = sanitize_external_text(&input);
            assert_eq!(out, expected, "{prefix} {separator}: {input:?} -> {out:?}");
            for leaked in [prefix, fragment.as_str(), tail.as_str()] {
                assert!(
                    !out.contains(leaked),
                    "{prefix} {separator}: a credential part survives in {out:?}"
                );
            }
            assert_eq!(
                sanitize_external_text(&out),
                out,
                "{prefix} {separator}: not idempotent: {out:?}"
            );
        }
    }
    // Case 2: a zero-width space splits one value into two bearer matches;
    // the shadow pass's first match consumes the second `Bearer`.
    for prefix in ["aim_", "amk_", "ams_"] {
        let input = format!("Bearer ABCDEFGH\u{200b}Bearer {prefix}{tail}");
        let out = sanitize_external_text(&input);
        assert_eq!(
            out, "Bearer [REDACTED] [REDACTED CREDENTIAL]",
            "{prefix}: {input:?} -> {out:?}"
        );
        for leaked in [prefix, tail.as_str()] {
            assert!(
                !out.contains(leaked),
                "{prefix}: a credential part survives in {out:?}"
            );
        }
        assert_eq!(
            sanitize_external_text(&out),
            out,
            "{prefix}: not idempotent: {out:?}"
        );
        // The same split written as a JSON string, the zero-width space in
        // its serialized escape: the client does not decode JSON escapes, so
        // the plain pass alone must see the matches. The escape's trailing
        // `b` glues onto the second `Bearer` and kills its word boundary, so
        // the union redacts the key itself and the net has nothing left —
        // either way no part of the key may survive.
        let input = format!(
            "{{\"out\": \"Bearer ABCDEFGH\\u200bBearer {prefix}{tail}\", \"neighbor\": \"KEEP_NEIGHBOR\"}}"
        );
        let out = sanitize_external_text(&input);
        assert_eq!(
            out,
            "{\"out\": \"Bearer [REDACTED]\\u200bBearer [REDACTED CREDENTIAL]\", \"neighbor\": \"KEEP_NEIGHBOR\"}",
            "{prefix}: {input:?} -> {out:?}"
        );
        assert_eq!(
            sanitize_external_text(&out),
            out,
            "{prefix}: not idempotent: {out:?}"
        );
    }
    // Controls: the simple header still defers to the bearer rule and keeps
    // its marker, so the net removes only what the bearer rule itself could
    // not redact, and a plain vendor key keeps the union's marker.
    for prefix in ["aim_", "amk_", "ams_"] {
        assert_eq!(
            sanitize_external_text(&format!("Authorization: Bearer {prefix}{tail}")),
            "Authorization: Bearer [REDACTED]"
        );
    }
    assert_eq!(
        sanitize_external_text(&format!("use {fragment} here")),
        "use [REDACTED CREDENTIAL] here"
    );
}
