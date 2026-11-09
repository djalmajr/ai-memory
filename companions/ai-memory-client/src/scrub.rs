//! Local credential scrub shared by external capture clients.
//!
//! The server sanitizer (`ai-memory-core::sanitize`) re-scrubs everything
//! these clients send; this copy keeps credentials out of the local spool,
//! queue and preview files written before delivery. It follows the same
//! rules for private keys and invisible characters: key blocks are matched
//! first, including PGP armor and a block cut off before its end line, and
//! invisible formatting characters do not hide a credential from the
//! patterns. Unlike the server it does not decode JSON string escapes, so a key
//! with part of itself written as `\uXXXX` passes here and is caught there.

use regex::Regex;
use std::sync::OnceLock;

/// ANSI/VT escape sequences, removed whole like the server's sanitizer does.
/// Removing only the ESC would leave `[32m` glued to whatever follows, where
/// neither this scrub nor the server could see a key behind it.
const ESCAPE_SEQUENCES: &str =
    r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]";

fn escape_sequences() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(ESCAPE_SEQUENCES).unwrap())
}

/// A native key prefix right after a word character and something removed
/// below. Removing that alone would glue the key to the word and hide it from
/// the native rule's word boundary, here and on the server, so the match becomes
/// `$1 $2` first. The prefix is enough: the key's body can hold a removed
/// character too. Mirrors the server's sanitizer.
fn native_key_after_stripped() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"([0-9A-Za-z_])(?:{ESCAPE_SEQUENCES}|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x{{9f}}\x{{202a}}-\x{{202e}}\x{{2066}}-\x{{2069}}\x{{2061}}-\x{{2064}}\x{{feff}}\x{{e0000}}-\x{{e007f}}])+((?:aim|amk|ams)_)"
        ))
        .unwrap()
    })
}

/// `text` seen through its transport encoding: every
/// [`is_default_ignorable`] character dropped, plus where each byte of the
/// view came from, so a match found in the view maps back to the bytes it
/// covers, invisible characters included.
struct Shadow {
    text: String,
    /// The start in `text` of the character each view byte was decoded
    /// from; repeated per byte, so a match's last view byte maps to the
    /// byte after its source character.
    origin: Vec<usize>,
}

/// Build the shadow view of `text` (callers check for invisible characters
/// first, so it only runs when a match could hide in them).
fn build_shadow(text: &str) -> Shadow {
    let mut shadow = String::with_capacity(text.len());
    let mut origin = Vec::with_capacity(text.len());
    for (index, ch) in text.char_indices() {
        if !is_default_ignorable(ch) {
            shadow.push(ch);
            origin.extend(index..index + ch.len_utf8());
        }
    }
    Shadow {
        text: shadow,
        origin,
    }
}

/// The leading group a rule's replacement keeps outside the redaction:
/// `pre` for the anchored vendor rules, the serialized escape for the
/// 22-character own-prefix rule. Nothing to keep: `None`.
fn kept_group<'a>(captures: &'a regex::Captures<'a>) -> Option<regex::Match<'a>> {
    captures.name("pre").or_else(|| captures.get(1))
}

pub fn sanitize_external_text(input: &str) -> String {
    let spaced = native_key_after_stripped().replace_all(input, "$1 $2");
    let stripped = escape_sequences().replace_all(&spaced, "");
    let input = stripped.as_ref();
    let mut offset = 0;
    let mut controls_removed = String::with_capacity(input.len());
    while let Some(ch) = input[offset..].chars().next() {
        let next = offset + ch.len_utf8();
        if ch == '\u{1f3f4}' {
            let tags = emoji_tag_run_len(&input[next..]);
            controls_removed.push_str(&input[offset..next + tags]);
            offset = next + tags;
            continue;
        }
        // Removed rather than replaced, before matching: a placeholder left
        // inside a credential would split it out of reach of every pattern.
        if ch == '\n'
            || ch == '\r'
            || ch == '\t'
            || !(ch.is_control()
                || matches!(ch, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
                || is_hidden_text_char(offset, ch))
        {
            controls_removed.push(ch);
        }
        offset = next;
    }
    let rules = secret_patterns();
    let mut union_applied = false;
    let mut text = controls_removed;
    for rule in rules {
        if matches!(
            rule.union_role,
            UnionRole::Credential | UnionRole::OwnPrefix
        ) {
            if union_applied {
                // The set was already unioned when its first member ran.
                continue;
            }
            union_applied = true;
            text = union_credential_spans(&text, rules);
            continue;
        }
        // Through invisible characters first, so a split credential is
        // replaced whole; then the plain pass, which still finds a match
        // that needs an invisible character as its word boundary.
        if text.contains(is_default_ignorable) {
            text = redact_through_invisibles(&text, rule);
        }
        text = rule
            .regex
            .replace_all(&text, |captures: &regex::Captures<'_>| {
                replacement(rule, captures)
            })
            .into_owned();
    }
    text
}

fn replacement(rule: &RedactionRule, captures: &regex::Captures<'_>) -> String {
    let matched = captures.get(0).map_or("", |matched| matched.as_str());
    if captures.get(3).is_some_and(|value| {
        value.as_str().chars().count() < 6
            || [
                "[REDACTED]",
                "[REDACTED CREDENTIAL]",
                "[REDACTED PRIVATE KEY]",
            ]
            .contains(&value.as_str())
    }) {
        matched.to_owned()
    } else {
        let mut replacement = String::new();
        captures.expand(rule.replacement, &mut replacement);
        replacement
    }
}

/// Run `rule` over `text` with invisible characters removed and replace each
/// match's span in `text`, invisible characters inside it included.
fn redact_through_invisibles(text: &str, rule: &RedactionRule) -> String {
    let shadow = build_shadow(text);
    let mut output = String::with_capacity(text.len());
    let mut copied = 0;
    for captures in rule.regex.captures_iter(&shadow.text) {
        let Some(found) = captures.get(0) else {
            continue;
        };
        let start = shadow
            .origin
            .get(found.start())
            .copied()
            .unwrap_or(text.len());
        let end = match found
            .end()
            .checked_sub(1)
            .and_then(|last| shadow.origin.get(last))
        {
            Some(last) if found.end() > found.start() => last + 1,
            _ => start,
        };
        let replaced = replacement(rule, &captures);
        // A match the rule keeps stays byte for byte, invisible characters
        // and all.
        if start < copied || replaced == found.as_str() {
            continue;
        }
        output.push_str(&text[copied..start]);
        output.push_str(&replaced);
        copied = end;
    }
    output.push_str(&text[copied..]);
    output
}

/// The spans `rule` would replace in `text`, from both views the rule always
/// runs: through the invisible characters first (each view match mapped back
/// to the span of `text` it covers, the kept group outside it), then the
/// plain pass, which still finds a match that needs an invisible character as
/// its word boundary. The kept groups (`pre`, the serialized escape) stay
/// out of every span, exactly where the rule's replacement leaves them.
fn union_member_spans(
    text: &str,
    shadow: Option<&Shadow>,
    rule: &RedactionRule,
) -> Vec<(usize, usize)> {
    let whole_match = matches!(rule.union_role, UnionRole::Bearer);
    let mut spans = Vec::new();
    if let Some(shadow) = shadow {
        for captures in rule.regex.captures_iter(&shadow.text) {
            let Some(found) = captures.get(0) else {
                continue;
            };
            // The view keeps one byte per source byte, so a group end maps
            // straight through `origin`; the span's end maps to the byte
            // after the match's last source character, which the credential
            // character classes make single-byte.
            let start = if whole_match {
                found.start()
            } else {
                kept_group(&captures).map_or(found.start(), |group| group.end())
            };
            if start >= found.end() {
                continue;
            }
            let start = shadow.origin.get(start).copied();
            let end = shadow
                .origin
                .get(found.end() - 1)
                .copied()
                .map(|last| last + 1);
            match (start, end) {
                (Some(s), Some(e)) if s < e => spans.push((s, e)),
                _ => {}
            }
        }
    }
    for captures in rule.regex.captures_iter(text) {
        let Some(found) = captures.get(0) else {
            continue;
        };
        let start = if whole_match {
            found.start()
        } else {
            kept_group(&captures).map_or(found.start(), |group| group.end())
        };
        if start < found.end() {
            spans.push((start, found.end()));
        }
    }
    spans
}

/// The vendor-prefixed credential rules, applied to the same text at once,
/// before any member rewrites it: a credential's body can hold a fragment
/// another member matches (`aim_…sk-…`, `sk-…aim_…`) and, run sequentially,
/// the member that finds the fragment first redacts it and splits the other's
/// match, so part of the credential is stored in clear. Overlapping spans
/// are unioned and replaced once with `[REDACTED CREDENTIAL]`.
///
/// An own-prefix span a `Bearer …` match fully holds is left for the bearer
/// rule instead: the bearer rule runs after the `sk-` rule and before both
/// own-prefix rules, so it redacts that key with its own marker today, and
/// the union must not take it first. A vendor span inside a `Bearer …` match
/// is the member's own business — its rule runs before the bearer rule
/// today, redacting the fragment and breaking the bearer value — and stays in
/// the union.
fn union_credential_spans(text: &str, rules: &[RedactionRule]) -> String {
    let shadow = if text.contains(is_default_ignorable) {
        Some(build_shadow(text))
    } else {
        None
    };
    let bearer: Vec<(usize, usize)> = rules
        .iter()
        .filter(|rule| matches!(rule.union_role, UnionRole::Bearer))
        .flat_map(|rule| union_member_spans(text, shadow.as_ref(), rule))
        .collect();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for rule in rules.iter().filter(|rule| {
        matches!(
            rule.union_role,
            UnionRole::Credential | UnionRole::OwnPrefix
        )
    }) {
        let defer = matches!(rule.union_role, UnionRole::OwnPrefix);
        for span in union_member_spans(text, shadow.as_ref(), rule) {
            if defer && bearer.iter().any(|&(b, e)| b <= span.0 && span.1 <= e) {
                continue;
            }
            spans.push(span);
        }
    }
    if spans.is_empty() {
        return text.to_string();
    }
    spans.sort_unstable();
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut index = 0;
    while index < spans.len() {
        let (start, mut end) = spans[index];
        index += 1;
        while index < spans.len() && spans[index].0 < end {
            end = end.max(spans[index].1);
            index += 1;
        }
        out.push_str(&text[copied..start]);
        out.push_str("[REDACTED CREDENTIAL]");
        copied = end;
    }
    out.push_str(&text[copied..]);
    out
}

/// Tag characters outside an emoji flag sequence, the invisible math
/// operators, and a byte-order mark after the start of the text: characters
/// whose only realistic use in captured text is hiding it.
const fn is_hidden_text_char(offset: usize, ch: char) -> bool {
    matches!(ch, '\u{2061}'..='\u{2064}' | '\u{e0000}'..='\u{e007f}')
        || (ch == '\u{feff}' && offset > 0)
}

/// Unicode `Default_Ignorable_Code_Point` (UAX #44), the same set the server
/// sanitizer ignores while matching.
const fn is_default_ignorable(ch: char) -> bool {
    matches!(ch,
        '\u{ad}' | '\u{34f}' | '\u{61c}' | '\u{115f}' | '\u{1160}' | '\u{17b4}' | '\u{17b5}'
        | '\u{180b}'..='\u{180f}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}'
        | '\u{2060}'..='\u{206f}' | '\u{3164}' | '\u{fe00}'..='\u{fe0f}' | '\u{feff}'
        | '\u{ffa0}' | '\u{fff0}'..='\u{fff8}' | '\u{1bca0}'..='\u{1bca3}'
        | '\u{1d173}'..='\u{1d17a}' | '\u{e0000}'..='\u{e0fff}')
}

/// Byte length of the tag run that completes an emoji flag sequence at the
/// start of `rest` (tag letters, then the cancel tag), or 0.
fn emoji_tag_run_len(rest: &str) -> usize {
    let mut len = 0;
    for ch in rest.chars() {
        match ch {
            '\u{e0020}'..='\u{e007e}' => len += ch.len_utf8(),
            '\u{e007f}' if len > 0 => return len + ch.len_utf8(),
            _ => return 0,
        }
    }
    0
}

/// Which role, if any, a rule plays in the credential-set union
/// (`union_credential_spans`): the set members redact their overlapping
/// spans as one; the own-prefix members defer to the bearer rule when it
/// holds their whole span, which is how that key is redacted today; and the
/// bearer rule's spans are collected so the union can make that deferral.
#[derive(Clone, Copy, PartialEq, Eq)]
enum UnionRole {
    None,
    Credential,
    OwnPrefix,
    Bearer,
}

struct RedactionRule {
    regex: Regex,
    replacement: &'static str,
    union_role: UnionRole,
}

fn secret_patterns() -> &'static [RedactionRule] {
    static RULES: OnceLock<Vec<RedactionRule>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            RedactionRule {
                regex: Regex::new(
                    r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*?-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----",
                )
                .unwrap(),
                replacement: "[REDACTED PRIVATE KEY]",
                union_role: UnionRole::None,
            },
            // A block with no end line, once at least one key-body line
            // follows the header; the line end after it is kept.
            RedactionRule {
                regex: Regex::new(
                    r#"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----(?:(?:\r?\n|(?:\\r)?\\n)[ \t]*(?:(?:Proc-Type|DEK-Info|Version|Comment|Hash):[^\r\n\\]*)?)*?(?:(?:\r?\n|(?:\\r)?\\n)[ \t]*[A-Za-z0-9+/=]{16,}[ \t]*)+(?P<end>\r?(?m:$)|["'\\…]|\.\.\.)"#,
                )
                .unwrap(),
                replacement: "[REDACTED PRIVATE KEY]${end}",
                union_role: UnionRole::None,
            },
            // GitHub tokens: left-unanchored like the server's rule, so a
            // token glued onto a word by a stripped control character (a
            // removed BEL, `…\x07ghp_…`) is still matched.
            RedactionRule {
                regex: Regex::new(
                    r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})",
                )
                .unwrap(),
                replacement: "[REDACTED CREDENTIAL]",
                union_role: UnionRole::Credential,
            },
            // AWS access-key IDs, anchored like the server's `aws_key` rule:
            // a boundary or a percent-encoded byte before the exact 20-char
            // shape. A letter removed from directly before `AKIA` stays
            // unmatched on purpose — redacting `AKIA` mid-word would destroy
            // ordinary uppercase text irreversibly.
            RedactionRule {
                regex: Regex::new(r"(?P<pre>\b|%[0-9A-Fa-f]{2})(?:AKIA|ASIA)[0-9A-Z]{16}\b")
                    .unwrap(),
                replacement: "${pre}[REDACTED CREDENTIAL]",
                union_role: UnionRole::Credential,
            },
            // Vendor `sk-` keys, anchored like the server's `api_key` rule: a
            // letter right before the prefix means it ends a word (`task-…`),
            // so only that may not start a match. A `\b` would also skip a key
            // glued to `_`, a digit, a CJK character or a percent escape.
            RedactionRule {
                regex: Regex::new(r"(?P<pre>^|[^A-Za-z]|%[0-9A-Fa-f]{2})sk-[A-Za-z0-9_-]{16,}")
                    .unwrap(),
                replacement: "${pre}[REDACTED CREDENTIAL]",
                union_role: UnionRole::Credential,
            },
            RedactionRule {
                regex: Regex::new(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}").unwrap(),
                replacement: "$1[REDACTED]",
                // Its span is only collected for the union's own-prefix
                // deferral: the rule itself runs where it always has, between
                // the `sk-` rule and both own-prefix rules.
                union_role: UnionRole::Bearer,
            },
            // ai-memory's own keys, mirroring the server's own-prefix pair:
            // `aim_`, `amk_` and `ams_` credentials in the captured text,
            // including right after a serialized control character — the
            // one-letter escapes the serializers use (\b \t \n \f \r), the
            // numeric spellings (\uXXXX, \xNN), the ANSI CSI escape that
            // follows a serialized ESC and the percent-encoded forms. The
            // escape is captured so the replacement keeps it, and the ASCII
            // word boundary rejects `claim_…` without treating `é` as one.
            // Both own-prefix rules are members of the credential-set union
            // (`union_credential_spans`), which collects their spans — and
            // the GitHub, AWS and `sk-` members' spans — over the same text
            // before any of them rewrites it, so a key body holding a
            // `ghp_…`, `sk-…` or `AKIA…` fragment is redacted whole. The
            // union defers an own-prefix key to the bearer rule when the
            // bearer holds it whole, which is how the key is redacted today.
            // Unlike the server the client replaces with its unlabeled
            // `[REDACTED CREDENTIAL]` marker.
            RedactionRule {
                regex: Regex::new(
                    r"((?:\\(?:[nrtbf]|u[0-9A-Fa-f]{4}|x[0-9A-Fa-f]{2})(?:\[[0-9;?]*[ -/]*[@-~])?|%[0-9A-Fa-f]{2})|(?-u:\b))(?:aim|amk|ams)_[A-Za-z0-9_\-]{22,}",
                )
                .unwrap(),
                replacement: "$1[REDACTED CREDENTIAL]",
                union_role: UnionRole::OwnPrefix,
            },
            // ai-memory's own keys: native `aim_`, web-session `ams_`, and
            // external mcp-auth `amk_`. Left-unanchored, like the server's
            // unanchored own-prefix rule: a 256-bit base64url tail after the
            // prefix makes a false positive negligible, so matching the prefix
            // even when a stripped control character or the visible tail of an
            // ANSI sequence glues a word onto it is safe and necessary. The
            // 22-character rule above runs first, so a key it covers is
            // redacted there. In the union it is the member that catches a
            // key a word is glued onto, where the anchored rule cannot see
            // it, and both own-prefix spans a `Bearer …` match holds whole
            // defer to the bearer rule.
            RedactionRule {
                regex: Regex::new(r"(?:aim|ams|amk)_[A-Za-z0-9_-]{40,}").unwrap(),
                replacement: "[REDACTED CREDENTIAL]",
                union_role: UnionRole::OwnPrefix,
            },
            RedactionRule {
                regex: Regex::new(
                    r#"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password)\b((?:\\*")?\s*[:=]\s*)(\\*"[^"\\\r\n]{6,}\\*"|'[^'\r\n]{6,}'|(?:\[REDACTED(?: CREDENTIAL| PRIVATE KEY)?\]|[^\s,;])+)"#,
                )
                .unwrap(),
                replacement: "$1$2[REDACTED]",
                union_role: UnionRole::None,
            },
        ]
    })
}
