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

/// Why a revocation did not happen. The variants are separate so a caller can
/// answer with the right status instead of matching on message text.
#[derive(Debug, PartialEq, Eq)]
pub enum RevokeError {
    /// No entry carries that label.
    Unknown,
    /// Removing it would leave the file with no usable entries, which the
    /// proxy refuses to start on.
    WouldEmpty,
    /// Reading or writing the file failed. The text is safe to log.
    Failed(String),
}

impl std::fmt::Display for RevokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RevokeError::Unknown => write!(f, "no entry with that label"),
            RevokeError::WouldEmpty => {
                write!(f, "removing it would leave the tokens file unusable")
            }
            RevokeError::Failed(e) => write!(f, "{e}"),
        }
    }
}

/// Remove the entry for `label` from the tokens file. A running proxy keeps the
/// token until it reloads the file.
pub fn revoke(path: &str, label: &str) -> Result<(), RevokeError> {
    let text = fs::read_to_string(path)
        .map_err(|e| RevokeError::Failed(format!("read tokens file {path}: {e}")))?;
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
        return Err(RevokeError::Unknown);
    }
    // The file parsed when the proxy loaded it, so removing one line can only
    // fail the parse by leaving it empty.
    if TokenStore::parse(&kept).is_err() {
        return Err(RevokeError::WouldEmpty);
    }
    let tmp = format!("{path}.tmp");
    fs::write(&tmp, kept).map_err(|e| RevokeError::Failed(format!("write {tmp}: {e}")))?;
    // A rename-replace takes the tmp file's mode and owner from the writing
    // process, so carry both over from the file being replaced. The owner
    // matters as much as the mode: a tokens file re-owned by the wrong user
    // leaves the service unable to read it, which is a crash loop rather than
    // one bad token.
    let meta = match fs::metadata(path) {
        Ok(meta) => meta,
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            return Err(RevokeError::Failed(format!("stat tokens file {path}: {e}")));
        }
    };
    let _ = fs::set_permissions(&tmp, meta.permissions());
    if let Err(e) = carry_owner(&tmp, &meta) {
        let _ = fs::remove_file(&tmp);
        return Err(RevokeError::Failed(format!(
            "carry ownership onto {tmp}: {e}"
        )));
    }
    fs::rename(&tmp, path)
        .map_err(|e| RevokeError::Failed(format!("replace tokens file {path}: {e}")))?;
    Ok(())
}

/// Give `tmp` the same owner as the file it is about to replace, so a
/// rename-replace cannot re-own it. Skipped when the owner already matches,
/// which keeps an unprivileged caller from needing a chown it would be
/// refused. Fails instead of silently re-owning the file.
#[cfg(unix)]
fn carry_owner(tmp: &str, target: &fs::Metadata) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let tmp_meta = fs::metadata(tmp).map_err(|e| format!("stat {tmp}: {e}"))?;
    if tmp_meta.uid() == target.uid() && tmp_meta.gid() == target.gid() {
        return Ok(());
    }
    std::os::unix::fs::chown(tmp, Some(target.uid()), Some(target.gid()))
        .map_err(|e| format!("chown {tmp}: {e}"))
}

#[cfg(not(unix))]
fn carry_owner(_tmp: &str, _target: &fs::Metadata) -> Result<(), String> {
    Ok(())
}

/// Remove `label` and reload `store` from the file, so the revoked token stops
/// authenticating on the next request instead of at the next restart. The
/// caller holds the write lock across both steps.
pub fn revoke_and_reload(
    store: &mut TokenStore,
    path: &str,
    label: &str,
) -> Result<(), RevokeError> {
    revoke(path, label)?;
    match TokenStore::load(path) {
        Ok(reloaded) => {
            *store = reloaded;
            Ok(())
        }
        Err(e) => Err(RevokeError::Failed(e)),
    }
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
        assert_eq!(revoke(&path, "alice"), Ok(()));
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
        assert_eq!(revoke(&path, "nobody"), Err(RevokeError::Unknown));
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn revoke_refuses_to_empty_the_file() {
        let before = line("alice", "a");
        let path = temp_tokens("revoke-last", &before);
        assert_eq!(revoke(&path, "alice"), Err(RevokeError::WouldEmpty));
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn revoke_and_reload_drops_the_token_from_memory() {
        let path = temp_tokens(
            "revoke-reload",
            &format!("{}{}", line("alice", "a"), line("bob", "b")),
        );
        let mut store = TokenStore::load(&path).unwrap();
        assert_eq!(store.lookup("a"), Some("alice"));

        assert_eq!(revoke_and_reload(&mut store, &path, "alice"), Ok(()));
        assert_eq!(store.lookup("a"), None);
        assert_eq!(store.lookup("b"), Some("bob"));

        // A label that is not there must not disturb the loaded store.
        assert_eq!(
            revoke_and_reload(&mut store, &path, "nobody"),
            Err(RevokeError::Unknown)
        );
        assert_eq!(store.lookup("b"), Some("bob"));
        let _ = fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn revoke_keeps_the_file_mode() {
        use std::os::unix::fs::PermissionsExt;
        let path = temp_tokens(
            "revoke-mode",
            &format!("{}{}", line("alice", "a"), line("bob", "b")),
        );
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

        revoke(&path, "alice").unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "revoke must not widen the file mode");
        let _ = fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn revoke_keeps_the_file_owner() {
        use std::os::unix::fs::MetadataExt;
        let path = temp_tokens(
            "revoke-owner",
            &format!("{}{}", line("alice", "a"), line("bob", "b")),
        );
        // Only a privileged caller can hand a file to another owner, so the
        // interesting half of this test needs root. That is the case that
        // matters: the deploy runs the CLI as root over a systemd
        // StateDirectory, where a re-owned file is unreadable to the service.
        let could_reown = std::os::unix::fs::chown(&path, Some(65534), Some(65534)).is_ok();
        let before = fs::metadata(&path).unwrap();

        revoke(&path, "alice").unwrap();

        let after = fs::metadata(&path).unwrap();
        assert_eq!(after.uid(), before.uid(), "revoke must not re-own the file");
        assert_eq!(after.gid(), before.gid(), "revoke must not re-own the file");
        if could_reown {
            assert_eq!(after.uid(), 65534, "root revoke must keep the owner");
        }
        let _ = fs::remove_file(&path);
    }
}
