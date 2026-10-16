//! Upstream key ring: several keys for one upstream, one active at a time.
//!
//! The proxy used to hold a single `UPSTREAM_API_KEY`. When that account ran
//! out of credit every request started failing, and the only way onto another
//! account was to edit the environment and restart. The ring keeps an ordered
//! list of keys and resolves the active one on every request, so a drained or
//! refused key is stepped over without a restart.
//!
//! Two signals drive a switch, and they are not the same thing:
//!
//! - **Balance.** The control plane reports the account's remaining credit. A
//!   background poller reads it per key and retires a key whose balance has
//!   fallen to `KEY_MIN_BALANCE`. This is the only signal that fires *before*
//!   the account is empty, which is the point of the whole exercise.
//! - **A refusal.** A `401` or `402` from the inference host means this key is
//!   no longer usable, so it is retired on the spot.
//!
//! A `429` is handled differently on purpose. The upstream answers `429` both
//! for a per-key rate limit and for `capacity_saturated`, which is the platform
//! running out of capacity and has nothing to do with this key. Rotating on
//! the second would walk the ring and retire every healthy account for a blip
//! no key can fix, so only the per-key case advances, and it applies a
//! cooldown rather than retiring the key.

use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Balance floor. A key at or below this leaves the rotation until it is
/// topped up. The unit is whatever the control plane reports (USDC).
pub const DEFAULT_MIN_BALANCE: f64 = 0.50;

/// How often each key's balance is re-read.
pub const DEFAULT_BALANCE_INTERVAL_SECS: u64 = 300;

/// Cooldown for a key that hit its own rate limit, when the response carries
/// no usable reset hint.
const DEFAULT_RATE_LIMIT_COOLDOWN_SECS: u64 = 60;

/// Bodies carrying these markers describe platform-wide capacity trouble, not
/// a per-key limit. Rotating cannot help either one.
const GLOBAL_CAPACITY_MARKERS: [&str; 2] = ["capacity_saturated", "no_capacity"];

/// Why the active key was stepped over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchReason {
    /// The upstream refused the key (`401`/`402`): bad key, or out of credit.
    Refused,
    /// This key hit its own rate limit. Temporary, unlike a refusal.
    RateLimited,
}

impl SwitchReason {
    pub fn as_str(self) -> &'static str {
        match self {
            SwitchReason::Refused => "refused",
            SwitchReason::RateLimited => "rate_limited",
        }
    }
}

/// A key change that just happened, for the log and the request record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Switch {
    pub from: String,
    pub to: String,
    pub reason: SwitchReason,
}

#[derive(Clone, Debug)]
struct UpstreamKey {
    label: String,
    secret: String,
    balance: Option<f64>,
    account: Option<String>,
    email: Option<String>,
    /// Requests served while this key was active.
    served: u64,
    /// Unix seconds of the last successful balance read.
    last_poll: Option<i64>,
    /// At or below the floor, or refused by the upstream. Out of rotation.
    drained: bool,
    /// Set by a per-key `429`. Temporary, unlike `drained`.
    cooldown_until: Option<Instant>,
}

impl UpstreamKey {
    fn new(label: String, secret: String) -> Self {
        Self {
            label,
            secret,
            balance: None,
            account: None,
            email: None,
            served: 0,
            last_poll: None,
            drained: false,
            cooldown_until: None,
        }
    }

    fn available(&self, now: Instant) -> bool {
        !self.drained && self.cooldown_until.is_none_or(|until| until <= now)
    }
}

struct Inner {
    keys: Vec<UpstreamKey>,
    active: usize,
    min_balance: f64,
}

/// The configured key ring, as reported by `/stats`.
#[derive(Serialize, Clone, Debug, Default)]
pub struct KeysSummary {
    /// Label of the key currently in use.
    pub active: String,
    /// Balance floor in the control plane's unit.
    pub min_balance: f64,
    pub keys: Vec<KeySummary>,
}

#[derive(Serialize, Clone, Debug)]
pub struct KeySummary {
    pub label: String,
    pub active: bool,
    pub balance: Option<f64>,
    pub account: Option<String>,
    pub email: Option<String>,
    pub served: u64,
    /// Out of rotation: at or below the floor, or refused.
    pub drained: bool,
    /// Cooling down after a per-key rate limit.
    pub rate_limited: bool,
    /// Age of the last balance reading, in seconds. `None` before the first
    /// successful poll.
    pub balance_age_secs: Option<i64>,
}

/// One balance reading from the control plane.
#[derive(Clone, Debug, PartialEq)]
pub struct BalanceReading {
    pub balance: f64,
    pub account: Option<String>,
    pub email: Option<String>,
}

/// A shared, ordered set of upstream keys.
#[derive(Clone)]
pub struct KeyRing {
    inner: Arc<Mutex<Inner>>,
}

/// Split `UPSTREAM_KEYS` into `(label, secret)` pairs.
///
/// Entries are comma separated. An entry may be `label:key`, which is the
/// readable form, or a bare key, which is numbered `key1`, `key2`, ... in the
/// order written. The split is on the FIRST colon, so a label may not contain
/// one, and a bare key that contains a colon would be read as `label:key`.
fn split_entries(raw: &str) -> Vec<(String, String)> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .enumerate()
        .map(|(i, entry)| match entry.split_once(':') {
            Some((label, key)) if !label.trim().is_empty() && !key.trim().is_empty() => {
                (label.trim().to_string(), key.trim().to_string())
            }
            _ => (format!("key{}", i + 1), entry.to_string()),
        })
        .collect()
}

impl KeyRing {
    /// Build the ring from `UPSTREAM_KEYS` when set, otherwise from the single
    /// `UPSTREAM_API_KEY`. A blank `UPSTREAM_KEYS` falls back to the single
    /// key rather than leaving the ring empty.
    ///
    /// Duplicate secrets are dropped: the same key twice would look like
    /// rotation while resolving to the same account.
    pub fn parse(raw: Option<&str>, fallback: &str, min_balance: f64) -> Self {
        let mut entries = raw.map(split_entries).unwrap_or_default();
        if entries.is_empty() {
            entries.push(("primary".to_string(), fallback.to_string()));
        }
        let mut keys: Vec<UpstreamKey> = Vec::with_capacity(entries.len());
        for (label, secret) in entries {
            if keys.iter().any(|k| k.secret == secret) {
                tracing::warn!("key ring: `{label}` duplicates an earlier key, dropping it");
                continue;
            }
            keys.push(UpstreamKey::new(label, secret));
        }
        assert!(!keys.is_empty(), "key ring is empty");
        Self {
            inner: Arc::new(Mutex::new(Inner {
                keys,
                active: 0,
                min_balance,
            })),
        }
    }

    /// Resolve the key to use right now and return its index, updating the
    /// stored active index when it moved.
    ///
    /// Order of preference: stay where we are while that key is available;
    /// otherwise the next available key forward from here, wrapping; if
    /// nothing is available, the richest balance, so a fully drained ring
    /// still serves instead of hard-failing.
    fn resolve(inner: &mut Inner, now: Instant) -> usize {
        let n = inner.keys.len();
        let chosen = if inner.keys[inner.active].available(now) {
            inner.active
        } else {
            (1..n)
                .map(|step| (inner.active + step) % n)
                .find(|&i| inner.keys[i].available(now))
                .unwrap_or_else(|| {
                    // Everything is drained or cooling down. Serve on the
                    // richest, keeping the current key on a tie or when no
                    // balance has been read yet.
                    let mut best = inner.active;
                    for i in 0..n {
                        let (a, b) = (inner.keys[i].balance, inner.keys[best].balance);
                        if matches!((a, b), (Some(a), Some(b)) if a > b) {
                            best = i;
                        }
                    }
                    best
                })
        };
        if chosen != inner.active {
            tracing::warn!(
                "key ring: `{}` unavailable, switching to `{}`",
                inner.keys[inner.active].label,
                inner.keys[chosen].label
            );
            inner.active = chosen;
        }
        chosen
    }

    /// The key to inject on this request, with its label, counted as served.
    ///
    /// Both come out of one lock and one resolution, so the label always names
    /// the key that actually carried the request.
    pub fn active(&self) -> (String, String) {
        let mut inner = self.inner.lock().expect("key ring lock poisoned");
        let now = Instant::now();
        let i = Self::resolve(&mut inner, now);
        inner.keys[i].served += 1;
        (inner.keys[i].label.clone(), inner.keys[i].secret.clone())
    }

    /// The secret to inject on this request.
    pub fn current(&self) -> String {
        let mut inner = self.inner.lock().expect("key ring lock poisoned");
        let now = Instant::now();
        let i = Self::resolve(&mut inner, now);
        inner.keys[i].served += 1;
        inner.keys[i].secret.clone()
    }

    /// The active key's label, for logs and the request record.
    pub fn label(&self) -> String {
        let mut inner = self.inner.lock().expect("key ring lock poisoned");
        let now = Instant::now();
        let i = Self::resolve(&mut inner, now);
        inner.keys[i].label.clone()
    }

    /// Number of configured keys.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("key ring lock poisoned")
            .keys
            .len()
    }

    /// Labels in configuration order, for the startup log.
    pub fn labels(&self) -> Vec<String> {
        self.inner
            .lock()
            .expect("key ring lock poisoned")
            .keys
            .iter()
            .map(|k| k.label.clone())
            .collect()
    }

    /// Every configured key's secret, for the balance poller.
    pub fn entries(&self) -> Vec<(String, String)> {
        self.inner
            .lock()
            .expect("key ring lock poisoned")
            .keys
            .iter()
            .map(|k| (k.label.clone(), k.secret.clone()))
            .collect()
    }

    /// React to an upstream status. Returns the switch when one happened.
    ///
    /// `cooldown` is the caller's read of the response's reset hint, used only
    /// for the per-key rate-limit case.
    pub fn on_status(
        &self,
        status: u16,
        body: Option<&str>,
        cooldown: Option<Duration>,
    ) -> Option<Switch> {
        match status {
            401 | 402 => self.retire(SwitchReason::Refused),
            429 if is_global_capacity(body) => None,
            429 => self.cool_down(cooldown),
            _ => None,
        }
    }

    /// Retire the active key and move on.
    fn retire(&self, reason: SwitchReason) -> Option<Switch> {
        let mut inner = self.inner.lock().expect("key ring lock poisoned");
        let now = Instant::now();
        let before = inner.active;
        let from = inner.keys[before].label.clone();
        inner.keys[before].drained = true;
        Self::resolve(&mut inner, now);
        if inner.active == before {
            tracing::error!(
                "key ring: `{from}` refused but no other key is available; still serving on it"
            );
            return None;
        }
        Some(Switch {
            from,
            to: inner.keys[inner.active].label.clone(),
            reason,
        })
    }

    /// Put the active key on a short cooldown after its own rate limit, and
    /// move on if there is anywhere to move to.
    fn cool_down(&self, cooldown: Option<Duration>) -> Option<Switch> {
        let mut inner = self.inner.lock().expect("key ring lock poisoned");
        let now = Instant::now();
        let wait = cooldown.unwrap_or(Duration::from_secs(DEFAULT_RATE_LIMIT_COOLDOWN_SECS));
        let before = inner.active;
        let from = inner.keys[before].label.clone();
        inner.keys[before].cooldown_until = Some(now + wait);
        let chosen = Self::resolve(&mut inner, now);
        if chosen == before {
            // Nowhere to go. Better to keep serving on the limited key than to
            // refuse the caller, so the cooldown is dropped again.
            inner.keys[before].cooldown_until = None;
            tracing::warn!(
                "key ring: `{from}` is rate limited and no other key is available; \
                 serving on it anyway"
            );
            return None;
        }
        Some(Switch {
            from,
            to: inner.keys[chosen].label.clone(),
            reason: SwitchReason::RateLimited,
        })
    }

    /// Store a balance reading. A key back above the floor returns to the
    /// rotation; one at or below it leaves.
    pub fn note_balance(&self, label: &str, reading: &BalanceReading, now_epoch: i64) {
        let mut inner = self.inner.lock().expect("key ring lock poisoned");
        let floor = inner.min_balance;
        let Some(key) = inner.keys.iter_mut().find(|k| k.label == label) else {
            tracing::warn!("key ring: balance for unknown label `{label}`");
            return;
        };
        key.balance = Some(reading.balance);
        key.last_poll = Some(now_epoch);
        if let Some(account) = &reading.account {
            key.account = Some(account.clone());
        }
        if let Some(email) = &reading.email {
            key.email = Some(email.clone());
        }
        let was = key.drained;
        key.drained = reading.balance <= floor;
        if key.drained && !was {
            tracing::warn!(
                "key ring: `{label}` at {:.4} is at or below the floor of {:.4}, retiring it",
                reading.balance,
                floor
            );
        } else if !key.drained && was {
            tracing::info!(
                "key ring: `{label}` is back at {:.4}, above the floor of {:.4}",
                reading.balance,
                floor
            );
        }
    }

    /// The ring as `/stats` reports it.
    pub fn summary(&self, now_epoch: i64) -> KeysSummary {
        let mut inner = self.inner.lock().expect("key ring lock poisoned");
        let now = Instant::now();
        let active = Self::resolve(&mut inner, now);
        let keys = inner
            .keys
            .iter()
            .enumerate()
            .map(|(i, k)| KeySummary {
                label: k.label.clone(),
                active: i == active,
                balance: k.balance,
                account: k.account.clone(),
                email: k.email.clone(),
                served: k.served,
                drained: k.drained,
                rate_limited: k.cooldown_until.is_some_and(|until| until > now),
                balance_age_secs: k.last_poll.map(|t| (now_epoch - t).max(0)),
            })
            .collect();
        KeysSummary {
            active: inner.keys[active].label.clone(),
            min_balance: inner.min_balance,
            keys,
        }
    }

    /// Test constructor with explicit keys and no env reads.
    #[cfg(test)]
    pub fn for_tests(entries: &[(&str, &str)], min_balance: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                keys: entries
                    .iter()
                    .map(|(l, s)| UpstreamKey::new(l.to_string(), s.to_string()))
                    .collect(),
                active: 0,
                min_balance,
            })),
        }
    }
}

/// True when a `429` body describes platform-wide capacity rather than this
/// key's own limit.
pub fn is_global_capacity(body: Option<&str>) -> bool {
    let Some(body) = body else { return false };
    GLOBAL_CAPACITY_MARKERS
        .iter()
        .any(|marker| body.contains(marker))
}

/// Parse a `/api/me` response into a balance reading.
pub fn parse_me(body: &str) -> Option<BalanceReading> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let raw = value.get("balances")?.get("consumer_balance")?;
    let balance = match raw {
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok()?,
        serde_json::Value::Number(n) => n.as_f64()?,
        _ => return None,
    };
    let field = |name: &str| value.get(name).and_then(|v| v.as_str()).map(str::to_string);
    Some(BalanceReading {
        balance,
        account: field("id"),
        email: field("email"),
    })
}

/// Balance poll interval from `KEY_BALANCE_INTERVAL_SECS`. Zero or unparseable
/// falls back to the default rather than disabling the poller by accident.
pub fn interval_from(raw: Option<&str>) -> Duration {
    Duration::from_secs(
        raw.and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_BALANCE_INTERVAL_SECS),
    )
}

/// Balance floor from `KEY_MIN_BALANCE`. A negative or unparseable value falls
/// back to the default.
pub fn min_balance_from(raw: Option<&str>) -> f64 {
    raw.and_then(|s| s.parse::<f64>().ok())
        .filter(|n| *n >= 0.0)
        .unwrap_or(DEFAULT_MIN_BALANCE)
}

/// Read one key's balance from the control plane.
///
/// The explicit User-Agent is not decoration: some control planes answer `403`
/// to request libraries' default agents, which looks exactly like a dead key.
async fn read_balance(client: &reqwest::Client, url: &str, secret: &str) -> Option<BalanceReading> {
    let resp = client
        .get(url)
        .header(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {secret}"),
        )
        .header(
            axum::http::header::USER_AGENT,
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        )
        .send()
        .await
        .map_err(|e| tracing::warn!("balance poll failed for {url}: {e}"))
        .ok()?;
    let status = resp.status();
    let body = resp.text().await.ok()?;
    if !status.is_success() {
        tracing::warn!(
            "balance poll: {url} answered {status}: {}",
            body.chars().take(120).collect::<String>()
        );
        return None;
    }
    match parse_me(&body) {
        Some(reading) => Some(reading),
        None => {
            tracing::warn!("balance poll: {url} body carried no consumer_balance");
            None
        }
    }
}

/// Start the balance poller. Returns immediately; the loop runs for the life
/// of the process. The caller only reaches this when `KEY_BALANCE_URL` is set,
/// so an unconfigured deployment has no poller at all.
///
/// The first pass runs at once, so the ring has real balances before the first
/// interval elapses instead of spending five minutes blind.
pub fn spawn_poller(
    ring: KeyRing,
    client: reqwest::Client,
    url: String,
    interval: Duration,
    tracker: crate::tracker::Tracker,
) {
    let labels = ring.labels();
    tracing::info!(
        "balance poller every {}s against {url} for [{}]",
        interval.as_secs(),
        labels.join(",")
    );
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            for (label, secret) in ring.entries() {
                let Some(reading) = read_balance(&client, &url, &secret).await else {
                    continue;
                };
                let now = chrono::Utc::now().timestamp();
                ring.note_balance(&label, &reading, now);
                tracker.record_key_balance(
                    &label,
                    reading.account.as_deref(),
                    reading.email.as_deref(),
                    reading.balance,
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(balance: f64) -> BalanceReading {
        BalanceReading {
            balance,
            account: None,
            email: None,
        }
    }

    #[test]
    fn labeled_entries_keep_their_labels() {
        let ring = KeyRing::parse(Some("oc-prod:sk-aaa,oc-staging:sk-bbb"), "x", 0.5);
        assert_eq!(ring.labels(), vec!["oc-prod", "oc-staging"]);
        assert_eq!(
            ring.entries(),
            vec![
                ("oc-prod".to_string(), "sk-aaa".to_string()),
                ("oc-staging".to_string(), "sk-bbb".to_string()),
            ]
        );
    }

    #[test]
    fn bare_entries_are_numbered_in_order() {
        let ring = KeyRing::parse(Some("sk-aaa, sk-bbb ,sk-ccc"), "x", 0.5);
        assert_eq!(ring.labels(), vec!["key1", "key2", "key3"]);
    }

    #[test]
    fn blank_list_falls_back_to_the_single_key() {
        // A blank UPSTREAM_KEYS must not leave the ring empty.
        let ring = KeyRing::parse(Some("  , , "), "solo", 0.5);
        assert_eq!(ring.labels(), vec!["primary"]);
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn unset_list_falls_back_to_the_single_key() {
        let ring = KeyRing::parse(None, "solo", 0.5);
        assert_eq!(ring.labels(), vec!["primary"]);
    }

    #[test]
    fn duplicate_secrets_are_dropped() {
        // The same key twice would read as rotation while resolving to one
        // account, which is the failure this feature exists to avoid.
        let ring = KeyRing::parse(Some("a:sk-same,b:sk-same,c:sk-other"), "x", 0.5);
        assert_eq!(ring.labels(), vec!["a", "c"]);
    }

    #[test]
    fn the_active_key_is_stable_while_it_is_available() {
        let ring = KeyRing::for_tests(&[("a", "sk-a"), ("b", "sk-b")], 0.5);
        assert_eq!(ring.label(), "a");
        assert_eq!(ring.label(), "a");
    }

    #[test]
    fn a_drained_key_is_stepped_over() {
        let ring = KeyRing::for_tests(&[("a", "sk-a"), ("b", "sk-b")], 0.5);
        let switched = ring.on_status(401, None, None);
        assert_eq!(
            switched,
            Some(Switch {
                from: "a".to_string(),
                to: "b".to_string(),
                reason: SwitchReason::Refused,
            })
        );
        assert_eq!(ring.label(), "b");
    }

    #[test]
    fn a_refused_key_is_not_reused_after_the_other_one_fails() {
        let ring = KeyRing::for_tests(&[("a", "sk-a"), ("b", "sk-b")], 0.5);
        ring.on_status(402, None, None);
        assert_eq!(ring.label(), "b");
        // Both are now retired, so there is nowhere left to go.
        assert_eq!(ring.on_status(402, None, None), None);
    }

    #[test]
    fn selection_wraps_back_to_an_earlier_key() {
        let ring = KeyRing::for_tests(&[("a", "sk-a"), ("b", "sk-b")], 0.5);
        // a refused: move to b.
        ring.on_status(401, None, None);
        assert_eq!(ring.label(), "b");
        // a is topped up, so it is a candidate again.
        ring.note_balance("a", &reading(2.0), 0);
        // b refused: the only way forward is back to a.
        ring.on_status(401, None, None);
        assert_eq!(ring.label(), "a");
    }

    #[test]
    fn a_fully_drained_ring_serves_on_the_richest_key() {
        let ring = KeyRing::for_tests(&[("a", "sk-a"), ("b", "sk-b")], 0.5);
        ring.note_balance("a", &reading(0.10), 0);
        ring.note_balance("b", &reading(0.40), 0);
        assert!(ring.summary(0).keys.iter().all(|k| k.drained));
        // Serving beats hard-failing a live proxy.
        assert_eq!(ring.label(), "b");
    }

    #[test]
    fn a_key_at_the_floor_is_retired_and_one_above_it_is_not() {
        let ring = KeyRing::for_tests(&[("a", "sk-a")], 0.5);
        ring.note_balance("a", &reading(0.50), 0);
        assert!(ring.summary(0).keys[0].drained, "at the floor retires");
        ring.note_balance("a", &reading(0.51), 0);
        assert!(!ring.summary(0).keys[0].drained, "above the floor returns");
    }

    #[test]
    fn capacity_saturation_never_rotates_the_key() {
        // Global capacity trouble: rotating would retire healthy accounts for
        // a blip no key can fix.
        let ring = KeyRing::for_tests(&[("a", "sk-a"), ("b", "sk-b")], 0.5);
        let body = r#"{"error":{"code":"capacity_saturated","message":"no capacity"}}"#;
        assert_eq!(
            ring.on_status(429, Some(body), Some(Duration::from_secs(2))),
            None
        );
        assert_eq!(ring.label(), "a", "the key must not have moved");
        assert_eq!(ring.on_status(503, Some(body), None), None);
    }

    #[test]
    fn a_per_key_rate_limit_cools_down_and_moves_on() {
        let ring = KeyRing::for_tests(&[("a", "sk-a"), ("b", "sk-b")], 0.5);
        let switched = ring.on_status(429, Some(r#"{"error":"rate limit exceeded"}"#), None);
        assert_eq!(
            switched.map(|s| (s.from, s.to, s.reason)),
            Some(("a".to_string(), "b".to_string(), SwitchReason::RateLimited))
        );
        assert_eq!(ring.label(), "b");
        // Rate limiting is temporary, so the key is not retired outright.
        assert!(!ring.summary(0).keys[0].drained);
    }

    #[test]
    fn a_rate_limited_key_with_nowhere_to_go_keeps_serving() {
        let ring = KeyRing::for_tests(&[("a", "sk-a")], 0.5);
        assert_eq!(ring.on_status(429, Some("slow down"), None), None);
        assert_eq!(ring.label(), "a");
        // The cooldown is dropped, or the only key would be skipped forever.
        assert!(!ring.summary(0).keys[0].rate_limited);
    }

    #[test]
    fn served_counts_track_the_active_key_only() {
        let ring = KeyRing::for_tests(&[("a", "sk-a"), ("b", "sk-b")], 0.5);
        ring.current();
        ring.current();
        ring.on_status(401, None, None);
        ring.current();
        let summary = ring.summary(0);
        assert_eq!(summary.keys[0].served, 2);
        assert_eq!(summary.keys[1].served, 1);
        assert_eq!(summary.active, "b");
        assert!(summary.keys[1].active);
    }

    #[test]
    fn parse_me_reads_the_string_balance_and_the_account() {
        let body = r#"{"id":"35d7aed4","email":"operator@example.com",
            "balances":{"consumer_balance":"2.329980","fiat_pendings":"2.330000"}}"#;
        let reading = parse_me(body).expect("parses");
        assert!((reading.balance - 2.32998).abs() < 1e-9);
        assert_eq!(reading.account.as_deref(), Some("35d7aed4"));
        assert_eq!(reading.email.as_deref(), Some("operator@example.com"));
    }

    #[test]
    fn parse_me_accepts_a_numeric_balance_and_rejects_the_rest() {
        assert_eq!(
            parse_me(r#"{"balances":{"consumer_balance":1.25}}"#)
                .expect("parses")
                .balance,
            1.25
        );
        assert!(parse_me(r#"{"balances":{}}"#).is_none());
        assert!(parse_me(r#"{"balances":{"consumer_balance":"nope"}}"#).is_none());
        assert!(parse_me("not json").is_none());
    }

    #[test]
    fn the_poller_interval_and_floor_fall_back_rather_than_disabling() {
        assert_eq!(
            interval_from(Some("0")),
            Duration::from_secs(DEFAULT_BALANCE_INTERVAL_SECS)
        );
        assert_eq!(
            interval_from(Some("junk")),
            Duration::from_secs(DEFAULT_BALANCE_INTERVAL_SECS)
        );
        assert_eq!(interval_from(Some("60")), Duration::from_secs(60));
        assert_eq!(min_balance_from(Some("1.5")), 1.5);
        assert_eq!(min_balance_from(Some("-1")), DEFAULT_MIN_BALANCE);
        assert_eq!(min_balance_from(None), DEFAULT_MIN_BALANCE);
    }
}
