/// Models that support image input on the upstream API.
pub const VISION_MODELS: &[&str] = &["mimo-v2-omni", "mimo-v2.5"];
pub const DEFAULT_VISION_MODEL: &str = "mimo-v2-omni";

/// Returns true if the request body contains image_url content in messages.
pub fn has_image_input(body: &serde_json::Value) -> bool {
    body.get("messages")
        .and_then(|m| m.as_array())
        .map(|msgs| {
            msgs.iter().any(|msg| {
                // content can be a string or an array of content parts
                match msg.get("content") {
                    Some(serde_json::Value::Array(parts)) => parts.iter().any(|p| {
                        p.get("type")
                            .and_then(|t| t.as_str())
                            .map(|t| t == "image_url")
                            .unwrap_or(false)
                    }),
                    _ => false,
                }
            })
        })
        .unwrap_or(false)
}

/// If the request has image input but the requested model doesn't support
/// vision, swap to the default vision model. Returns (possibly modified) body
/// and the model name actually sent upstream.
pub fn maybe_swap_vision_model(body: &mut serde_json::Value) -> String {
    let requested = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();

    if has_image_input(body) && !VISION_MODELS.iter().any(|vm| requested == *vm) {
        tracing::info!(
            "vision swap: {} -> {} (image input detected)",
            requested,
            DEFAULT_VISION_MODEL
        );
        body["model"] = serde_json::Value::String(DEFAULT_VISION_MODEL.to_string());
        DEFAULT_VISION_MODEL.to_string()
    } else {
        requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_body(model: &str, content: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 50
        })
    }

    #[test]
    fn text_only_no_swap() {
        let mut body = make_body("mimo-v2.5-pro", serde_json::json!("hello"));
        let used = maybe_swap_vision_model(&mut body);
        assert_eq!(used, "mimo-v2.5-pro");
        assert_eq!(body["model"], "mimo-v2.5-pro");
    }

    #[test]
    fn image_triggers_swap() {
        let content = serde_json::json!([
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        ]);
        let mut body = make_body("mimo-v2.5-pro", content);
        let used = maybe_swap_vision_model(&mut body);
        assert_eq!(used, "mimo-v2-omni");
        assert_eq!(body["model"], "mimo-v2-omni");
    }

    #[test]
    fn vision_model_no_swap() {
        let content = serde_json::json!([
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        ]);
        let mut body = make_body("mimo-v2-omni", content);
        let used = maybe_swap_vision_model(&mut body);
        assert_eq!(used, "mimo-v2-omni");
        assert_eq!(body["model"], "mimo-v2-omni");
    }

    #[test]
    fn v25_model_with_image_no_swap() {
        let content = serde_json::json!([
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        ]);
        let mut body = make_body("mimo-v2.5", content);
        let used = maybe_swap_vision_model(&mut body);
        assert_eq!(used, "mimo-v2.5");
        assert_eq!(body["model"], "mimo-v2.5");
    }

    #[test]
    fn string_content_no_swap() {
        let mut body = make_body("mimo-v2.5-pro", serde_json::json!("just text"));
        let used = maybe_swap_vision_model(&mut body);
        assert_eq!(used, "mimo-v2.5-pro");
    }

    #[test]
    fn has_image_detects_nested() {
        let body = serde_json::json!({
            "model": "mimo-v2.5-pro",
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]}
            ]
        });
        assert!(has_image_input(&body));
    }

    #[test]
    fn has_image_false_for_text() {
        let body = serde_json::json!({
            "model": "mimo-v2.5-pro",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        });
        assert!(!has_image_input(&body));
    }
}
