//! Model rotation on upstream failure.
//!
//! When the model a request landed on dies (transport error, no response
//! headers within the per-attempt timeout, or an upstream 5xx), the proxy can
//! retry the same request on the next candidate model instead of feeding the
//! caller the upstream's error body. Candidates come from `ROTATION_MODELS`
//! (comma list) and default to the `ALLOWED_MODELS` list, so a proxy with one
//! enabled model gets a single-attempt plan and rotation is a no-op.
//!
//! The requested model always tries first, even when the configured list
//! starts elsewhere. A plan holds at most [`MAX_ATTEMPTS`] candidates, and
//! each attempt bounds only the wait for response headers with
//! `ROTATION_ATTEMPT_TIMEOUT_SECS`, so a hung upstream cannot pin a request
//! for the client's whole timeout and a streaming body is never cut
//! mid-generation.

use serde::Serialize;
use std::time::Duration;

/// Hard cap on upstream attempts for one request, requested model included.
pub const MAX_ATTEMPTS: usize = 3;

/// Default per-attempt wait for response headers, in seconds.
pub const DEFAULT_ATTEMPT_TIMEOUT_SECS: u64 = 15;

/// Bodies at or above this many bytes are compaction-scale (roughly 130k
/// tokens of JSON history at 4 bytes per token, the regime where the first
/// token legitimately takes minutes). They are exempt from the first-token
/// trigger: rotating would restart the same huge prompt on another model and
/// pay the prompt processing all over again.
pub const DEFAULT_TTFT_MAX_BODY_BYTES: usize = 524_288;

#[derive(Clone, Debug)]
pub struct Rotation {
    models: Vec<String>,
    attempt_timeout: Duration,
    ttft_max_body_bytes: usize,
}

/// The configured rotation, as reported by `/stats` and the startup log.
///
/// `plan_len` is what a request actually gets, not what is configured: the
/// candidate list is capped at [`MAX_ATTEMPTS`]. A `plan_len` of 1 means
/// rotation is a no-op no matter what the list says, which is the single
/// configuration mistake that turns every upstream hang into a full-timeout
/// wait with nothing to fall back to.
#[derive(Serialize, Clone, Debug, Default)]
pub struct RotationSummary {
    /// Comma-joined configured candidates, in order.
    pub names: String,
    /// Candidates one request may use, capped at [`MAX_ATTEMPTS`].
    pub plan_len: u64,
    /// Per-attempt wait for response headers.
    pub attempt_timeout_secs: u64,
    /// False when `plan_len` is 1, so nothing can be rotated to.
    pub engaging: bool,
}

fn split_ids(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Per-attempt header timeout from a raw env value. Junk values and zero fall
/// back to the default (a zero timeout would abort every attempt).
fn attempt_timeout(raw: Option<&str>) -> Duration {
    Duration::from_secs(
        raw.and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_ATTEMPT_TIMEOUT_SECS)
            .max(1),
    )
}

impl Rotation {
    /// `rotation_raw` is the `ROTATION_MODELS` value; `None` or a blank value
    /// means "the same ids as the allowlist". `allowlist` is the already
    /// resolved `ALLOWED_MODELS` string. Panics when the effective list is
    /// empty, matching `Allowlist::parse`: a proxy that can serve nothing is
    /// a config error and should refuse to start.
    pub fn parse(rotation_raw: Option<&str>, allowlist: &str) -> Self {
        let models = match rotation_raw.map(split_ids) {
            Some(ids) if !ids.is_empty() => ids,
            _ => split_ids(allowlist),
        };
        assert!(!models.is_empty(), "rotation candidate list is empty");
        let attempt_timeout = attempt_timeout(
            std::env::var("ROTATION_ATTEMPT_TIMEOUT_SECS")
                .ok()
                .as_deref(),
        );
        let ttft_max_body_bytes = std::env::var("ROTATION_TTFT_MAX_BODY_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_TTFT_MAX_BODY_BYTES);
        Self {
            models,
            attempt_timeout,
            ttft_max_body_bytes,
        }
    }

    /// Wait for response headers on one attempt.
    pub fn attempt_timeout(&self) -> Duration {
        self.attempt_timeout
    }

    /// Comma-joined candidates, for startup logs.
    pub fn names(&self) -> String {
        self.models.join(",")
    }

    /// The configured candidates, in order.
    pub fn candidates(&self) -> &[String] {
        &self.models
    }

    /// The configured rotation, for `/stats` and the startup log.
    pub fn summary(&self) -> RotationSummary {
        let plan_len = self.models.len().min(MAX_ATTEMPTS) as u64;
        RotationSummary {
            names: self.names(),
            plan_len,
            attempt_timeout_secs: self.attempt_timeout.as_secs(),
            engaging: plan_len > 1,
        }
    }

    /// Test constructor: no env reads, explicit timeout.
    #[cfg(test)]
    pub fn for_tests(models: &[&str], attempt_timeout: Duration) -> Self {
        Self {
            models: models.iter().map(|s| s.to_string()).collect(),
            attempt_timeout,
            ttft_max_body_bytes: DEFAULT_TTFT_MAX_BODY_BYTES,
        }
    }

    /// First-token budget for a request, when the request earns one.
    ///
    /// Only streaming requests get a first-token trigger: for a non-streaming
    /// request the whole body IS the completion, so no budget short of the
    /// client's own timeout can be safe. Bodies at or above
    /// `ROTATION_TTFT_MAX_BODY_BYTES` are compaction-scale (manual or auto
    /// compact) and are exempt: their first token legitimately takes minutes,
    /// and rotating on it would be a false positive that restarts the same
    /// huge prompt on another model.
    pub fn first_byte_budget(&self, request_body_bytes: usize, stream: bool) -> Option<Duration> {
        if stream && request_body_bytes < self.ttft_max_body_bytes {
            Some(self.attempt_timeout)
        } else {
            None
        }
    }

    /// Candidates for one request: the configured ids in order,
    /// deduplicated, capped at [`MAX_ATTEMPTS`]. The requested id takes the
    /// front only when it is already a member of the configured list, so an
    /// explicit `ROTATION_MODELS` list is used verbatim. That matters when
    /// the list carries provider-prefixed ids the client never sends: a bare
    /// client id must not be injected ahead of them. An empty requested id
    /// is skipped.
    pub fn plan_for(&self, requested: &str) -> Vec<String> {
        let mut plan: Vec<String> = Vec::with_capacity(MAX_ATTEMPTS);
        let mut push = |id: &str| {
            if !id.is_empty() && plan.len() < MAX_ATTEMPTS && plan.iter().all(|seen| seen != id) {
                plan.push(id.to_string());
            }
        };
        if self.models.iter().any(|m| m == requested) {
            push(requested);
        }
        for id in &self.models {
            push(id);
        }
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_the_allowlist_and_is_a_noop_for_one_model() {
        let r = Rotation::parse(None, "deepseek-v4.1-flash");
        assert_eq!(
            r.plan_for("deepseek-v4.1-flash"),
            vec!["deepseek-v4.1-flash"]
        );
    }

    #[test]
    fn explicit_list_is_used_verbatim_ignoring_the_requested_id() {
        let r = Rotation::parse(Some("b-model,c-model"), "a-model");
        assert_eq!(r.plan_for("a-model"), vec!["b-model", "c-model"]);
    }

    #[test]
    fn requested_inside_the_list_stays_first() {
        let r = Rotation::parse(Some("b-model,a-model,c-model"), "a-model");
        assert_eq!(r.plan_for("a-model"), vec!["a-model", "b-model", "c-model"]);
    }

    #[test]
    fn configured_candidates_follow_in_order_without_duplicates() {
        let r = Rotation::parse(Some("a-model,b-model,a-model,b-model"), "x");
        assert_eq!(r.plan_for("b-model"), vec!["b-model", "a-model"]);
    }

    #[test]
    fn plan_is_capped_at_max_attempts() {
        let r = Rotation::parse(Some("m1,m2,m3,m4,m5"), "x");
        assert_eq!(r.plan_for("req"), vec!["m1", "m2", "m3"]);
        assert_eq!(r.plan_for("").len(), MAX_ATTEMPTS);
    }

    #[test]
    fn empty_requested_id_is_skipped() {
        let r = Rotation::parse(Some("m1,m2,m3,m4"), "x");
        assert_eq!(r.plan_for(""), vec!["m1", "m2", "m3"]);
    }

    #[test]
    fn blank_rotation_value_falls_back_to_allowlist() {
        let r = Rotation::parse(Some(" , "), "fallback-model");
        assert_eq!(r.plan_for("fallback-model"), vec!["fallback-model"]);
    }

    #[test]
    #[should_panic(expected = "rotation candidate list is empty")]
    fn empty_effective_list_is_a_config_error() {
        Rotation::parse(Some(" , "), " , ");
    }

    #[test]
    fn attempt_timeout_parses_clamps_and_defaults() {
        assert_eq!(attempt_timeout(None), Duration::from_secs(15));
        assert_eq!(attempt_timeout(Some("junk")), Duration::from_secs(15));
        assert_eq!(attempt_timeout(Some("30")), Duration::from_secs(30));
        // Zero would abort every attempt, so it clamps to one second.
        assert_eq!(attempt_timeout(Some("0")), Duration::from_secs(1));
    }

    #[test]
    fn summary_reports_whether_rotation_can_engage() {
        // One candidate: rotation is a no-op and the summary must say so.
        let one = Rotation::parse(Some("combo/cheapest"), "x").summary();
        assert_eq!(one.plan_len, 1);
        assert!(!one.engaging);
        assert_eq!(one.names, "combo/cheapest");

        // Three candidates: rotation can engage.
        let three = Rotation::for_tests(&["a", "b", "c"], Duration::from_secs(40)).summary();
        assert_eq!(three.plan_len, 3);
        assert!(three.engaging);
        assert_eq!(three.attempt_timeout_secs, 40);

        // Five configured, three usable: the cap is what a request gets.
        let capped = Rotation::parse(Some("a,b,c,d,e"), "x").summary();
        assert_eq!(capped.plan_len, MAX_ATTEMPTS as u64);
        assert!(capped.engaging);
    }

    #[test]
    fn first_byte_budget_applies_to_small_streaming_requests_only() {
        let r = Rotation::parse(None, "m1");
        // Small streaming request: first-token trigger on.
        assert_eq!(
            r.first_byte_budget(4_096, true),
            Some(Duration::from_secs(15))
        );
        // Compaction-scale body: exempt, manual or auto compact alike.
        assert_eq!(r.first_byte_budget(524_288, true), None);
        assert_eq!(r.first_byte_budget(700_000, true), None);
        // Non-streaming: the whole body is the completion, never a trigger.
        assert_eq!(r.first_byte_budget(4_096, false), None);
    }
}
