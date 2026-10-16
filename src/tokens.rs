//! Per-caller access tokens.
//!
//! The tokens file holds one `label <sha256 hex>` pair per line. Only hashes
//! are stored, so a leaked file does not reveal any caller's token. A caller
//! sends its token as `Authorization: Bearer <token>`, and the proxy hashes it
//! and looks up the matching label.

use sha2::{Digest, Sha256};
use std::{fs, io, io::Read};

pub struct TokenStore {
    /// (label, sha256 hex) pairs, one per caller.
    entries: Vec<(String, String)>,
}

impl TokenStore {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("read tokens file {path}: {e}"))?;
        Self::parse(&text)
    }

    /// Parse the tokens file. Refuses an empty file, so the proxy never starts
    /// open to everyone.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut entries: Vec<(String, String)> = Vec::new();
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(label), Some(hash), None) = (parts.next(), parts.next(), parts.next()) else {
                return Err(format!(
                    "tokens file line {}: expected `label sha256hex`",
                    n + 1
                ));
            };
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!(
                    "tokens file line {}: hash must be 64 hex characters",
                    n + 1
                ));
            }
            if entries.iter().any(|(l, _)| l == label) {
                return Err(format!(
                    "tokens file line {}: duplicate label `{label}`",
                    n + 1
                ));
            }
            entries.push((label.to_string(), hash.to_ascii_lowercase()));
        }
        if entries.is_empty() {
            return Err("tokens file has no entries; refusing to start with no access".into());
        }
        Ok(Self { entries })
    }

    /// Label of the caller that owns `token`, if any.
    pub fn lookup(&self, token: &str) -> Option<&str> {
        let digest = hash_hex(token);
        self.entries
            .iter()
            .find(|(_, h)| *h == digest)
            .map(|(label, _)| label.as_str())
    }

    /// True when an entry with this label exists, whatever its token.
    pub fn has_label(&self, label: &str) -> bool {
        self.entries.iter().any(|(l, _)| l == label)
    }
}

pub fn hash_hex(token: &str) -> String {
    Sha256::digest(token.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Prefix on every token issued from now on. Secret scanners match it, and the
/// binary redacts it from logs. Tokens issued before the prefix stay valid as
/// bare hex, since lookup hashes whatever string the caller sends.
pub const TOKEN_PREFIX: &str = "ocp_";

/// A fresh random token: `ocp_` followed by 32 bytes from the OS, hex encoded.
fn new_token() -> io::Result<String> {
    let mut buf = [0u8; 32];
    fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    let hex: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!("{TOKEN_PREFIX}{hex}"))
}

/// Issue a token for `label`. Appends the label and the token's hash to the
/// tokens file and returns the token. The caller must show it once and never
/// store it on the server.
pub fn issue(path: &str, label: &str) -> Result<String, String> {
    if label.is_empty()
        || !label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("label must be non-empty and use only letters, digits, - or _".into());
    }
    let mut text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("read tokens file {path}: {e}")),
    };
    if text
        .lines()
        .any(|l| l.split_whitespace().next() == Some(label))
    {
        return Err(format!("label `{label}` already exists in {path}"));
    }
    let token = new_token().map_err(|e| format!("generate token: {e}"))?;
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&format!("{label} {}\n", hash_hex(&token)));
    fs::write(path, text).map_err(|e| format!("write tokens file {path}: {e}"))?;
    Ok(token)
}

/// Remove the entry for `label` from the tokens file. Returns `false` when the
/// label is not there. The running proxy keeps the token until it restarts.
/// Refuses a removal that would leave the file with no entries, because the
/// proxy will not start on an empty file.
pub fn revoke(path: &str, label: &str) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read tokens file {path}: {e}"))?;
    let mut kept = String::new();
    let mut removed = false;
    for raw in text.lines() {
        if raw.split_whitespace().next() == Some(label) {
            removed = true;
            continue;
        }
        kept.push_str(raw);
        kept.push('\n');
    }
    if !removed {
        return Ok(false);
    }
    TokenStore::parse(&kept)
        .map_err(|e| format!("revoking `{label}` would leave the tokens file unusable: {e}"))?;
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, kept).map_err(|e| format!("write {tmp}: {e}"))?;
    fs::rename(&tmp, path).map_err(|e| format!("replace tokens file {path}: {e}"))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(label: &str, token: &str) -> String {
        format!("{label} {}\n", hash_hex(token))
    }

    #[test]
    fn lookup_finds_matching_label() {
        let text = format!("# devs\n{}{}", line("alice", "tok-a"), line("bob", "tok-b"));
        let store = TokenStore::parse(&text).unwrap();
        assert_eq!(store.lookup("tok-a"), Some("alice"));
        assert_eq!(store.lookup("tok-b"), Some("bob"));
        assert_eq!(store.lookup("tok-c"), None);
    }

    #[test]
    fn empty_file_is_refused() {
        assert!(TokenStore::parse("").is_err());
        assert!(TokenStore::parse("# only a comment\n\n").is_err());
    }

    #[test]
    fn plaintext_token_in_file_is_refused() {
        // A line with a short value is not a hash and must not be accepted.
        assert!(TokenStore::parse("alice tok-a\n").is_err());
    }

    #[test]
    fn duplicate_label_is_refused() {
        let text = format!("{}{}", line("alice", "one"), line("alice", "two"));
        assert!(TokenStore::parse(&text).is_err());
    }

    #[test]
    fn hash_is_64_hex_and_stable() {
        let h = hash_hex("abc");
        assert_eq!(h.len(), 64);
        assert_eq!(h, hash_hex("abc"));
    }

    #[test]
    fn issue_writes_hash_and_returns_working_token() {
        let path = std::env::temp_dir().join(format!("keyproxy-issue-{}.txt", std::process::id()));
        let path_str = path.to_str().unwrap();
        let _ = fs::remove_file(&path);

        let token = issue(path_str, "carol").unwrap();
        let store = TokenStore::load(path_str).unwrap();
        assert_eq!(store.lookup(&token), Some("carol"));

        // The file must hold the hash, never the token.
        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(!on_disk.contains(&token));

        // A second issue for the same label is refused.
        assert!(issue(path_str, "carol").is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn issued_token_has_prefix_and_64_hex_digits() {
        let path = std::env::temp_dir().join(format!("keyproxy-prefix-{}.txt", std::process::id()));
        let path_str = path.to_str().unwrap();
        let _ = fs::remove_file(&path);

        let token = issue(path_str, "dave").unwrap();
        let _ = fs::remove_file(&path);

        let digits = token
            .strip_prefix(TOKEN_PREFIX)
            .expect("token must start with ocp_");
        assert_eq!(digits.len(), 64);
        assert!(digits.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn issue_rejects_bad_labels() {
        assert!(issue("/nonexistent/x", "").is_err());
        assert!(issue("/nonexistent/x", "has space").is_err());
    }

    #[test]
    fn has_label_matches_entries_only() {
        let text = format!("{}{}", line("alice", "tok-a"), line("bob", "tok-b"));
        let store = TokenStore::parse(&text).unwrap();
        assert!(store.has_label("alice"));
        assert!(!store.has_label("carol"));
        assert!(!store.has_label("tok-a"));
    }

    fn temp_tokens(name: &str, text: &str) -> String {
        let path = std::env::temp_dir().join(format!("keyproxy-{name}-{}.txt", std::process::id()));
        fs::write(&path, text).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn revoke_removes_only_that_label() {
        let path = temp_tokens(
            "revoke",
            &format!("{}{}", line("alice", "a"), line("bob", "b")),
        );
        assert_eq!(revoke(&path, "alice"), Ok(true));
        let store = TokenStore::load(&path).unwrap();
        assert!(!store.has_label("alice"));
        assert_eq!(store.lookup("a"), None);
        assert_eq!(store.lookup("b"), Some("bob"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn revoke_unknown_label_changes_nothing() {
        let before = line("alice", "a");
        let path = temp_tokens("revoke-missing", &before);
        assert_eq!(revoke(&path, "nobody"), Ok(false));
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn revoke_refuses_to_empty_the_file() {
        let before = line("alice", "a");
        let path = temp_tokens("revoke-last", &before);
        assert!(revoke(&path, "alice").is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
        let _ = fs::remove_file(&path);
    }
}
