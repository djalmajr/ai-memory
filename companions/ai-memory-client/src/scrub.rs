//! Local credential scrub shared by external capture clients.
//!
//! The server sanitizer (`ai-memory-core::sanitize`) re-scrubs everything
//! these clients send; this copy keeps credentials out of the local spool,
//! queue and preview files written before delivery. It follows the same
//! rules for private keys and invisible characters: key blocks are matched
//! first, including PGP armor and a block cut off before its end line, and
//! invisible formatting characters do not hide a credential from the
//! patterns. Like the server's sanitizer the credential-set union collects
//! its spans through views of the same pre-rewrite text: the plain text, a
//! shadow copy without `Default_Ignorable_Code_Point` characters, and a
//! decoded view that decodes one layer of JSON string escapes (the
//! two-letter escapes, `\uXXXX` and a valid surrogate pair) and drops the
//! same invisibles, so a key written as `a\u0069m_…` is seen and redacted
//! whole — the escape's bytes included — here, not only on the server. The
//! decoding is one layer deep, the same semantics as the server's view: an
//! invalid or truncated escape, a lone surrogate and a `\\u…` spelling stay
//! literal text.

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
            UnionRole::Credential | UnionRole::OwnPrefix | UnionRole::Bearer
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

/// The spans `rule` would replace in `text`, from the three views the union
/// always runs: through the invisible characters first (each view match
/// mapped back to the span of `text` it covers, the kept group outside it),
/// through the one-layer JSON-escape decoded view (a match's span projected
/// onto the original bytes, the final escape's full extent included), and
/// the plain pass, which still finds a match that needs an invisible
/// character as its word boundary. The kept groups (`pre`, the serialized
/// escape, the `bearer\s+` label) stay out of every span, exactly where the
/// rule's replacement leaves them: for the bearer rule the span is the value
/// after the label, which is what the union redacts and the label is what
/// stays.
fn union_member_spans(
    text: &str,
    shadow: Option<&Shadow>,
    decoded: Option<&Shadow>,
    rule: &RedactionRule,
) -> Vec<(usize, usize)> {
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
            let start = kept_group(&captures).map_or(found.start(), |group| group.end());
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
        let start = kept_group(&captures).map_or(found.start(), |group| group.end());
        if start < found.end() {
            spans.push((start, found.end()));
        }
    }
    if let Some(decoded) = decoded {
        for captures in rule.regex.captures_iter(&decoded.text) {
            let Some(found) = captures.get(0) else {
                continue;
            };
            // Each view byte maps to the source character that produced it
            // (one per source byte), so the span's start projects to the
            // first source byte of the match's first character and the end
            // past the last source byte the match's final character covers —
            // a decoded escape's full extent, a valid surrogate pair's
            // twelve bytes included.
            let start = kept_group(&captures).map_or(found.start(), |group| group.end());
            if start < found.end() {
                let from = decoded.origin[start];
                let last = decoded.origin[found.end() - 1];
                spans.push((from, last + source_len_at(text, last)));
            }
        }
    }
    spans
}

/// The credential set — the vendor-prefixed credential rules, the own-prefix
/// rules and the bearer value rule — applied to the same text at once,
/// before any member rewrites it: a credential's body can hold a fragment
/// another member matches (`aim_…sk-…`, `sk-…aim_…`) and, run sequentially,
/// the member that finds the fragment first redacts it and splits the other's
/// match, so part of the credential is stored in clear. A bearer value the
/// same way can hold a credential plus a tail the credential span does not
/// reach (`Bearer aim_…ZZZ.A.YYYY`), and the member that finds the credential
/// first redacts only it, leaving the tail of the same bearer value in clear.
/// Overlapping spans are unioned and replaced once: a group holding a
/// credential member (a vendor-prefixed or own-prefix span) comes out
/// `[REDACTED CREDENTIAL]`, a group made of bearer value spans alone comes
/// out `[REDACTED]`, so `Authorization: Bearer xyz…` stays
/// `Authorization: Bearer [REDACTED]` and `Authorization: Bearer aim_…`
/// stays `Authorization: Bearer [REDACTED CREDENTIAL]`.
///
/// The bearer rule therefore never runs on its own after the union: its
/// matches are exactly the value spans the union collects from the same
/// three views (plain, shadow and the one-layer JSON-escape decoded view)
/// over the same pre-rewrite text, merged with the credential spans they
/// overlap. The decoded view is what a key whose prefix or separator is
/// written as `a\u0069m_…` needs: neither the plain text nor the shadow can
/// see through the escape, so without it the union would redact the vendor
/// fragment and the bearer value up to the escape's first byte and leave the
/// key's body in clear where only the server — which redacts the whole
/// header value — used to catch it. A bearer value the union does not cover
/// cannot exist — every value match the rule would find is a collected span
/// — and a value the union did redact holds no bearer characters, so the
/// marker is left alone on a second pass.
fn union_credential_spans(text: &str, rules: &[RedactionRule]) -> String {
    let shadow = if text.contains(is_default_ignorable) {
        Some(build_shadow(text))
    } else {
        None
    };
    // Built once per union and reused by every member rule: the escape
    // decoding is the union's costliest view, so it must not run per rule.
    let decoded = if text.contains('\\') {
        Some(decoded_shadow(text))
    } else {
        None
    };
    // Each span carries whether its member is a credential (a vendor-
    // prefixed or own-prefix rule) or a bearer value: the merged group's
    // marker depends on the mix.
    let mut spans: Vec<(usize, usize, bool)> = Vec::new();
    for rule in rules.iter().filter(|rule| {
        matches!(
            rule.union_role,
            UnionRole::Credential | UnionRole::OwnPrefix | UnionRole::Bearer
        )
    }) {
        let credential = !matches!(rule.union_role, UnionRole::Bearer);
        for (start, end) in union_member_spans(text, shadow.as_ref(), decoded.as_ref(), rule) {
            spans.push((start, end, credential));
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
        let (start, mut end, mut credential) = spans[index];
        index += 1;
        while index < spans.len() && spans[index].0 < end {
            end = end.max(spans[index].1);
            credential = credential || spans[index].2;
            index += 1;
        }
        out.push_str(&text[copied..start]);
        out.push_str(if credential {
            "[REDACTED CREDENTIAL]"
        } else {
            "[REDACTED]"
        });
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

/// Build the one-layer JSON-escape decoded view of `text`, dropping the same
/// [`is_default_ignorable`] characters the shadow drops. `origin` maps each
/// view byte to the source *character* that produced it (repeated for the
/// view byte's own length), so a match's last view byte projects back to the
/// full source extent of the character that decoded into it — the escape's
/// six bytes, a surrogate pair's twelve, or the character's UTF-8 length for
/// a plain one. An escape the decoder does not accept (invalid, truncated,
/// a lone surrogate) contributes its literal bytes like any other character:
/// one layer, no recursion, nothing re-serialized.
fn decoded_shadow(text: &str) -> Shadow {
    let mut shadow = String::with_capacity(text.len());
    let mut origin = Vec::with_capacity(text.len());
    let mut index = 0;
    while let Some(c) = text[index..].chars().next() {
        let (c, len) = json_escape_at(text.as_bytes(), index).unwrap_or((c, c.len_utf8()));
        if !is_default_ignorable(c) {
            shadow.push(c);
            origin.extend(std::iter::repeat_n(index, c.len_utf8()));
        }
        index += len;
    }
    Shadow {
        text: shadow,
        origin,
    }
}

/// How many source bytes the character at `index` occupies: the escape's
/// length when a JSON escape starts there (the character decoded from it),
/// otherwise the character's own UTF-8 length.
fn source_len_at(text: &str, index: usize) -> usize {
    json_escape_at(text.as_bytes(), index).map_or_else(
        || text[index..].chars().next().map_or(0, char::len_utf8),
        |(_, len)| len,
    )
}

/// The character the JSON string escape at `bytes[index..]` stands for and
/// the escape's byte length. Anything else, a lone surrogate included, is
/// text. Same semantics as the server's `json_escape_at` in
/// `crates/ai-memory-core/src/sanitize.rs`: the two-letter escapes, `\uXXXX`
/// and a `\uXXXX\uXXXX` surrogate pair; no `\xNN`, no recursion.
fn json_escape_at(bytes: &[u8], index: usize) -> Option<(char, usize)> {
    if bytes.get(index) != Some(&b'\\') {
        return None;
    }
    let decoded = match bytes.get(index + 1)? {
        b'"' => ('"', 2),
        b'\\' => ('\\', 2),
        b'/' => ('/', 2),
        b'b' => ('\u{8}', 2),
        b'f' => ('\u{c}', 2),
        b'n' => ('\n', 2),
        b'r' => ('\r', 2),
        b't' => ('\t', 2),
        b'u' => json_unicode_escape_at(bytes, index)?,
        _ => return None,
    };
    Some(decoded)
}

/// `\uXXXX`, or a `\uXXXX\uXXXX` surrogate pair, at `bytes[index..]`.
fn json_unicode_escape_at(bytes: &[u8], index: usize) -> Option<(char, usize)> {
    let unit = json_hex_unit(bytes, index + 2)?;
    if !(0xD800..=0xDBFF).contains(&unit) {
        return Some((char::from_u32(unit)?, 6));
    }
    if bytes.get(index + 6..index + 8)? != b"\\u" {
        return None;
    }
    let low = json_hex_unit(bytes, index + 8).filter(|low| (0xDC00..=0xDFFF).contains(low))?;
    let code = 0x1_0000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
    Some((char::from_u32(code)?, 12))
}

/// The four hex digits at `bytes[at..]` as a UTF-16 code unit.
fn json_hex_unit(bytes: &[u8], at: usize) -> Option<u32> {
    bytes.get(at..at + 4)?.iter().try_fold(0, |unit, b| {
        Some((unit << 4) | char::from(*b).to_digit(16)?)
    })
}

/// Which role, if any, a rule plays in the credential-set union
/// (`union_credential_spans`): the set members redact their overlapping
/// spans as one, per merged group. A group holding a `Credential` or
/// `OwnPrefix` span comes out `[REDACTED CREDENTIAL]`; a group of `Bearer`
/// spans alone comes out `[REDACTED]`. The bearer rule contributes only the
/// value after the `bearer\s+` label and never runs on its own after the
/// union: every value match is collected into the union from the same three
/// views the rule would run on (plain, shadow and decoded).

#[derive(Clone, Copy)]
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
                // Unused: the rule's matches are collected into the union as
                // value spans (the `bearer\s+` label kept out by
                // `kept_group`) and never run on their own. A bearer-only
                // group redacts under `[REDACTED]`, a group that also holds
                // a credential member under `[REDACTED CREDENTIAL]`.
                replacement: "$1[REDACTED]",
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
            // union redacts an own-prefix group even inside a `Bearer …`
            // value, so a header holding a native key comes out as
            // `Bearer [REDACTED CREDENTIAL]`; the bearer rule then runs on
            // the rewritten text and leaves the marker alone.
            // Unlike the server the client replaces with its unlabeled
            // `[REDACTED CREDENTIAL]` marker, and the union collects this
            // rule's spans together with the bearer value's: a `Bearer …`
            // value holding an own key comes out
            // `Bearer [REDACTED CREDENTIAL]` with the whole value gone,
            // not only the key's group.
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
            // it, and the union redacts the group with its own marker even
            // inside a `Bearer …` value, where it merges with the bearer
            // value's span so the whole value is redacted.
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
