//! Per-request input ceiling.
//!
//! The proxy forwards whatever a caller sends, so one caller can make the
//! upstream read an unbounded prompt and burn a key's balance on a single call.
//! `InputLimit` refuses a body whose estimated input reaches `max_tokens`,
//! before any upstream call and before the body is parsed for a model.
//!
//! `0` disables the ceiling, the same convention `Caps` uses. The default is
//! `DEFAULT_MAX_INPUT_TOKENS`, which matches the built-in context window of the
//! OpenCrabs client. That client compacts at 65% of its window, so a correctly
//! configured client stays well under the ceiling and never meets it. Traffic
//! that does meet it is either misconfigured or not ours.
//!
//! The estimate is per part, mirroring the client's own accounting:
//!
//! | Part | Charge |
//! |---|---|
//! | text | characters / 4 |
//! | `image_url` | 1,000 tokens, the client's per-image constant |
//! | `input_audio` | 1,000 tokens |
//! | `tool_calls`, `tools` | characters / 4 of the serialized JSON |
//! | no `messages` array, or not JSON | bytes / 4 |
//!
//! Bytes alone would be wrong. A 1 MB PNG is about 1.35 MB of base64, roughly
//! 340k by `bytes / 4`, while the client charges 1,000 tokens for it, so a
//! single screenshot would trip the ceiling on its own.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

/// Bytes per token for text, matching the compaction-scale math in `rotation`.
pub const BYTES_PER_TOKEN: u64 = 4;
/// Charged for one `image_url` part. The client charges the same per image.
pub const IMAGE_PART_TOKENS: u64 = 1_000;
/// Charged for one `input_audio` part, whose base64 length is not its cost.
pub const AUDIO_PART_TOKENS: u64 = 1_000;
/// The ceiling used when `MAX_INPUT_TOKENS` is unset or unparsable.
pub const DEFAULT_MAX_INPUT_TOKENS: u64 = 200_000;
/// The largest body accepted, matching `client_max_body_size 25m` on nginx.
pub const MAX_BODY_BYTES: usize = 25 * 1024 * 1024;

/// The per-request input ceiling. `max_tokens` of `0` means no ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputLimit {
    max_tokens: u64,
}

impl InputLimit {
    /// Read the ceiling from a raw value. Unset or unparsable falls back to the
    /// default, so a typo in the environment cannot remove the guardrail.
    pub fn parse(raw: Option<&str>) -> Self {
        Self {
            max_tokens: raw
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(DEFAULT_MAX_INPUT_TOKENS),
        }
    }

    /// Read the ceiling from `MAX_INPUT_TOKENS`.
    pub fn from_env() -> Self {
        Self::parse(std::env::var("MAX_INPUT_TOKENS").ok().as_deref())
    }

    /// The default ceiling, for tests that must not read the environment.
    #[cfg(test)]
    pub fn for_tests() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_INPUT_TOKENS,
        }
    }

    pub fn max_tokens(&self) -> u64 {
        self.max_tokens
    }

    /// False when the ceiling is disabled with `0`.
    pub fn enabled(&self) -> bool {
        self.max_tokens > 0
    }

    /// The test that gates a request: the estimate scaled by the caller's
    /// measured density, which corrects the flat `characters / 4` constant for
    /// how dense this caller's content really is. A factor of `1.0`, what a
    /// caller with no history gets, makes this a plain estimate against the
    /// ceiling.
    pub fn exceeded_scaled(&self, estimate: u64, factor: f64) -> bool {
        if !self.enabled() {
            return false;
        }
        let scaled = (estimate as f64 * factor).ceil();
        scaled >= self.max_tokens as f64
    }
}

/// The 413 answer, in the same shape `Allowlist::refusal` uses.
pub fn refusal(estimated: u64, max: u64) -> Response {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(json!({
            "error": {
                "message": format!(
                    "estimated input of {estimated} tokens is at or above the {max} token ceiling"
                ),
                "type": "input_too_large",
            }
        })),
    )
        .into_response()
}

/// Estimate the input tokens a request body will cost.
pub fn estimate_tokens(body: &[u8]) -> u64 {
    match serde_json::from_slice::<Value>(body) {
        Ok(value) if value.get("messages").and_then(Value::as_array).is_some() => {
            estimate_messages(&value)
        }
        _ => tokens_for(body.len() as u64),
    }
}

/// Every message, plus the tool schemas the request carries.
fn estimate_messages(body: &Value) -> u64 {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return 0;
    };
    let mut total: u64 = messages.iter().map(estimate_message).sum();
    if let Some(tools) = body.get("tools") {
        total += tokens_for_json(tools);
    }
    total
}

/// One message: its content, and any tool calls it carries.
fn estimate_message(message: &Value) -> u64 {
    let mut total = match message.get("content") {
        Some(Value::String(text)) => tokens_for_chars(text),
        Some(Value::Array(parts)) => parts.iter().map(estimate_part).sum(),
        Some(other) => tokens_for_json(other),
        None => 0,
    };
    if let Some(calls) = message.get("tool_calls") {
        total += tokens_for_json(calls);
    }
    total
}

/// One content part. Images and audio cost a flat constant, never their length.
fn estimate_part(part: &Value) -> u64 {
    if part.get("image_url").is_some() {
        return IMAGE_PART_TOKENS;
    }
    if part.get("input_audio").is_some() {
        return AUDIO_PART_TOKENS;
    }
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        return tokens_for_chars(text);
    }
    tokens_for_json(part)
}

fn tokens_for_chars(text: &str) -> u64 {
    tokens_for(text.chars().count() as u64)
}

fn tokens_for_json(value: &Value) -> u64 {
    tokens_for_chars(&value.to_string())
}

/// Rounded up, so a part that costs anything at all costs at least one token.
fn tokens_for(bytes: u64) -> u64 {
    bytes.div_ceil(BYTES_PER_TOKEN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    fn body_with_image(image_bytes: usize) -> Vec<u8> {
        let image = "A".repeat(image_bytes);
        json!({
            "model": "deepseek-v4.1-flash",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{image}")}}
                ]
            }]
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn unset_falls_back_to_the_default() {
        assert_eq!(InputLimit::parse(None).max_tokens(), 200_000);
        assert_eq!(InputLimit::parse(None), InputLimit::for_tests());
    }

    #[test]
    fn unparsable_falls_back_to_the_default() {
        assert_eq!(InputLimit::parse(Some("lots")).max_tokens(), 200_000);
        assert_eq!(InputLimit::parse(Some("")).max_tokens(), 200_000);
    }

    #[test]
    fn zero_is_unlimited() {
        let limit = InputLimit::parse(Some("0"));
        assert!(!limit.enabled());
        let huge = vec![b'x'; 8 * 1024 * 1024];
        assert!(!limit.exceeded_scaled(estimate_tokens(&huge), 1.0));
    }

    #[test]
    fn a_value_is_read_and_trimmed() {
        assert_eq!(InputLimit::parse(Some(" 4096 ")).max_tokens(), 4096);
    }

    #[test]
    fn at_the_ceiling_is_refused_and_one_under_passes() {
        let limit = InputLimit::parse(Some("100"));
        // A text body of 396 characters estimates to 99 tokens.
        let under = json!({"messages": [{"role": "user", "content": "x".repeat(396)}]})
            .to_string()
            .into_bytes();
        assert_eq!(estimate_tokens(&under), 99);
        assert!(!limit.exceeded_scaled(estimate_tokens(&under), 1.0));

        let at = json!({"messages": [{"role": "user", "content": "x".repeat(400)}]})
            .to_string()
            .into_bytes();
        assert_eq!(estimate_tokens(&at), 100);
        assert!(limit.exceeded_scaled(estimate_tokens(&at), 1.0));
    }

    #[test]
    fn a_base64_image_is_charged_the_image_constant_not_its_length() {
        // 200 KB of base64 is 50k tokens by `bytes / 4`, but 1,000 here.
        let body = body_with_image(200 * 1024);
        let estimate = estimate_tokens(&body);
        assert!(
            estimate < 1_100,
            "an image part must cost about 1,000 tokens, got {estimate}"
        );
        assert!(estimate >= IMAGE_PART_TOKENS);
        // The same body under a 200k ceiling passes, which is the whole point.
        assert!(!InputLimit::for_tests().exceeded_scaled(estimate_tokens(&body), 1.0));
    }

    #[test]
    fn text_costs_about_a_quarter_of_its_characters() {
        let body = json!({"messages": [{"role": "user", "content": "x".repeat(4000)}]})
            .to_string()
            .into_bytes();
        assert_eq!(estimate_tokens(&body), 1_000);
    }

    #[test]
    fn a_non_json_body_falls_back_to_bytes_per_token() {
        let body = vec![b'x'; 4000];
        assert_eq!(estimate_tokens(&body), 1_000);
    }

    #[test]
    fn a_json_body_without_messages_falls_back_to_bytes_per_token() {
        let body = br#"{"model":"deepseek-v4.1-flash"}"#.to_vec();
        assert_eq!(estimate_tokens(&body), (body.len() as u64).div_ceil(4));
    }

    #[test]
    fn tool_calls_and_tool_schemas_are_counted() {
        let without = json!({
            "messages": [{"role": "user", "content": "hi"}]
        })
        .to_string()
        .into_bytes();
        let with = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {
                "name": "read_file",
                "description": "x".repeat(4000),
                "parameters": {"type": "object"}
            }}]
        })
        .to_string()
        .into_bytes();
        assert!(estimate_tokens(&with) > estimate_tokens(&without) + 900);
    }

    #[test]
    fn an_audio_part_is_charged_a_flat_constant() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": "A".repeat(400_000), "format": "wav"}}
            ]}]
        })
        .to_string()
        .into_bytes();
        assert_eq!(estimate_tokens(&body), AUDIO_PART_TOKENS);
    }

    #[tokio::test]
    async fn the_refusal_is_a_413_with_the_input_too_large_type() {
        let response = refusal(334_842, 200_000);
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["error"]["type"], "input_too_large");
        let message = parsed["error"]["message"].as_str().unwrap();
        assert!(message.contains("334842"), "{message}");
        assert!(message.contains("200000"), "{message}");
    }
}
