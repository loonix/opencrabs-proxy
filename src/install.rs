//! Per-install token issuance.
//!
//! An install asks for its token once. The proxy issues one when the install
//! id is well formed and has no token yet. The install id becomes the token's
//! label, so `/stats` shows each install by name and usage can be tracked per
//! install.

use crate::tokens::{self, TokenStore};

pub const MIN_ID_LEN: usize = 8;
pub const MAX_ID_LEN: usize = 64;

/// Requests to the install route allowed per client IP per day. Every request
/// counts, including refused ones, so a loop of random ids is bounded too.
pub const DEFAULT_REQUESTS_PER_DAY: u32 = 5000;

/// The per-IP daily cap from `INSTALL_RATE_PER_DAY`. Unset, empty or not a
/// number falls back to the default. Zero is a real value and refuses all.
pub fn requests_per_day(value: Option<&str>) -> u32 {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_REQUESTS_PER_DAY)
}

#[derive(Debug, PartialEq, Eq)]
pub enum Refusal {
    InvalidId,
    AlreadyIssued,
    /// Reading or writing the tokens file failed. The text is safe to log.
    Failed(String),
}

/// An install id is a label: letters, digits, `-` and `_`, with a length bound.
pub fn valid_install_id(id: &str) -> bool {
    (MIN_ID_LEN..=MAX_ID_LEN).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// The rule, without any I/O: may `id` receive a token now?
pub fn check(id: &str, already_issued: bool) -> Result<(), Refusal> {
    if !valid_install_id(id) {
        return Err(Refusal::InvalidId);
    }
    if already_issued {
        return Err(Refusal::AlreadyIssued);
    }
    Ok(())
}

/// Issue `id` its token if the rule allows. On success the token is written
/// to `path` (hash only) and `store` is reloaded from that file, so the new
/// token authenticates without a restart.
pub fn issue_for(store: &mut TokenStore, path: &str, id: &str) -> Result<String, Refusal> {
    check(id, store.has_label(id))?;
    let token = tokens::issue(path, id).map_err(Refusal::Failed)?;
    *store = TokenStore::load(path).map_err(Refusal::Failed)?;
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const ID: &str = "install-0001";

    fn temp_file(name: &str, text: &str) -> String {
        let path = std::env::temp_dir().join(format!("keyproxy-{name}-{}.txt", std::process::id()));
        fs::write(&path, text).unwrap();
        path.to_str().unwrap().to_string()
    }

    /// A tokens file with one unrelated operator entry, so it is never empty.
    fn operator_file(name: &str) -> String {
        temp_file(
            name,
            &format!("ops {}\n", tokens::hash_hex("operator-token")),
        )
    }

    #[test]
    fn install_id_bounds_and_charset() {
        assert!(valid_install_id("abcdefgh"));
        assert!(valid_install_id("core-dev_01"));
        assert!(!valid_install_id("short"));
        assert!(!valid_install_id("has space here"));
        assert!(!valid_install_id("slash/not/allowed"));
        assert!(!valid_install_id(&"a".repeat(MAX_ID_LEN + 1)));
    }

    #[test]
    fn requests_per_day_defaults_and_parses() {
        assert_eq!(requests_per_day(None), 5000);
        assert_eq!(requests_per_day(Some("")), 5000);
        assert_eq!(requests_per_day(Some("  ")), 5000);
        assert_eq!(requests_per_day(Some("not-a-number")), 5000);
        assert_eq!(requests_per_day(Some(" 250 ")), 250);
        assert_eq!(requests_per_day(Some("0")), 0);
    }

    #[test]
    fn check_covers_each_refusal() {
        assert_eq!(check("x", false), Err(Refusal::InvalidId));
        assert_eq!(check(ID, true), Err(Refusal::AlreadyIssued));
        assert_eq!(check(ID, false), Ok(()));
    }

    #[test]
    fn new_install_gets_one_token_and_no_second() {
        let path = operator_file("install-once");
        let mut store = TokenStore::load(&path).unwrap();

        let token = issue_for(&mut store, &path, ID).unwrap();
        assert_eq!(store.lookup(&token), Some(ID));

        // A second request is refused and does not mint another token.
        assert_eq!(
            issue_for(&mut store, &path, ID),
            Err(Refusal::AlreadyIssued)
        );
        let on_disk = fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk.matches(ID).count(), 1);
        assert!(
            !on_disk.contains(&token),
            "the file must hold the hash only"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn invalid_install_id_is_refused_and_file_is_untouched() {
        let path = operator_file("install-invalid");
        let before = fs::read_to_string(&path).unwrap();
        let mut store = TokenStore::load(&path).unwrap();

        assert_eq!(
            issue_for(&mut store, &path, "bad id"),
            Err(Refusal::InvalidId)
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn revoked_install_token_stops_working_and_install_can_reissue() {
        let path = operator_file("install-revoke");
        let mut store = TokenStore::load(&path).unwrap();
        let first = issue_for(&mut store, &path, ID).unwrap();

        assert_eq!(tokens::revoke(&path, ID), Ok(true));
        let mut reloaded = TokenStore::load(&path).unwrap();
        assert_eq!(
            reloaded.lookup(&first),
            None,
            "revoked token must no longer authenticate"
        );

        // With the old entry gone, the install may be issued again.
        let second = issue_for(&mut reloaded, &path, ID).unwrap();
        assert_ne!(first, second);
        assert_eq!(reloaded.lookup(&second), Some(ID));
        let _ = fs::remove_file(&path);
    }
}
