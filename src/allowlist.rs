//! Model allowlist. Only the listed upstream model ids are served: a request
//! naming any other model is refused before it reaches the upstream, and
//! `/v1/models` lists only these ids. Set with `ALLOWED_MODELS` (comma list).
//!
//! Image generation has its own list, `IMAGE_MODELS`. Its ids are served only
//! on `/v1/images/*`, and chat or `/v1/models` refuse them, so an image model
//! can never be used as a chat model. The two lists must not overlap.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

pub const DEFAULT_MODELS: &str = "deepseek-v4.1-flash";

/// Image generation ids served on `/v1/images/*` only. Callers send the bare id;
/// the proxy adds `IMAGE_UPSTREAM_PREFIX` on the upstream request.
pub const DEFAULT_IMAGE_MODELS: &str = "gemini-3.1-flash-image";

/// Prefix the upstream expects on image model ids. Added here, never by callers,
/// so the upstream route name is not part of anything a client sees.
pub const IMAGE_UPSTREAM_PREFIX: &str = "cb/";

/// The id the upstream receives for an image model the caller named.
pub fn upstream_image_id(id: &str) -> String {
    format!("{IMAGE_UPSTREAM_PREFIX}{id}")
}

/// Path prefix that selects the image list instead of the chat list.
const IMAGE_PATH_PREFIX: &str = "/v1/images/";

#[derive(Clone, Debug)]
pub struct Allowlist {
    models: Vec<String>,
    images: Vec<String>,
}

fn split_ids(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

impl Allowlist {
    /// Panics on an empty list. An allowlist that allows nothing is a config
    /// error, and the proxy should refuse to start rather than refuse everything.
    pub fn parse(raw: &str) -> Self {
        let models = split_ids(raw);
        assert!(!models.is_empty(), "ALLOWED_MODELS lists no model");
        Self {
            models,
            images: Vec::new(),
        }
    }

    /// Attach the image generation list. Empty means image generation is off.
    /// Panics if an id is on both lists, since that id would reach chat.
    pub fn with_images(mut self, raw: &str) -> Self {
        let images = split_ids(raw);
        for id in &images {
            assert!(
                !self.models.contains(id),
                "IMAGE_MODELS id `{id}` is also a chat model; the lists must not overlap"
            );
        }
        self.images = images;
        self
    }

    pub fn names(&self) -> String {
        self.models.join(",")
    }

    pub fn image_names(&self) -> String {
        self.images.join(",")
    }

    pub fn allows(&self, model: &str) -> bool {
        self.models.iter().any(|m| m == model)
    }

    pub fn allows_image(&self, model: &str) -> bool {
        self.images.iter().any(|m| m == model)
    }

    /// Route-aware check: image routes accept only image ids, every other
    /// route accepts only chat ids.
    pub fn allows_for_path(&self, path: &str, model: &str) -> bool {
        if path.starts_with(IMAGE_PATH_PREFIX) {
            self.allows_image(model)
        } else {
            self.allows(model)
        }
    }

    /// Keep only allowed ids in an OpenAI-style `{"data": [{"id": ...}]}` list.
    /// A payload of another shape is left as it is.
    pub fn filter_models(&self, v: &mut Value) {
        if let Some(arr) = v.get_mut("data").and_then(Value::as_array_mut) {
            arr.retain(|m| {
                m.get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| self.allows(id))
            });
        }
    }

    pub fn refusal(model: &str) -> Response {
        (
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": {
                    "message": format!("model `{model}` is not available on this proxy"),
                    "type": "model_not_allowed",
                }
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comma_list_and_trims() {
        let a = Allowlist::parse(" deepseek-v4.1-flash , other-model ,");
        assert!(a.allows("deepseek-v4.1-flash"));
        assert!(a.allows("other-model"));
        assert!(!a.allows("ali/deepseek-v4.1-flash"));
    }

    #[test]
    fn default_allows_only_flash() {
        let a = Allowlist::parse(DEFAULT_MODELS);
        assert!(a.allows("deepseek-v4.1-flash"));
        assert!(!a.allows("deepseek-v4.1-pro"));
        assert!(!a.allows(""));
    }

    #[test]
    #[should_panic(expected = "lists no model")]
    fn empty_list_is_a_config_error() {
        Allowlist::parse(" , ");
    }

    #[test]
    fn filter_keeps_only_allowed_ids() {
        let a = Allowlist::parse(DEFAULT_MODELS);
        let mut v = json!({"object": "list", "data": [
            {"id": "deepseek-v4.1-flash"},
            {"id": "ali/deepseek-v4.1-flash"},
            {"id": "gpt-x"},
        ]});
        a.filter_models(&mut v);
        let ids: Vec<&str> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["deepseek-v4.1-flash"]);
    }

    #[test]
    fn filter_leaves_other_shapes_alone() {
        let a = Allowlist::parse(DEFAULT_MODELS);
        let mut v = json!({"error": "nope"});
        a.filter_models(&mut v);
        assert_eq!(v, json!({"error": "nope"}));
    }

    #[test]
    fn image_id_is_never_a_chat_model() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_images(DEFAULT_IMAGE_MODELS);
        assert!(!a.allows(DEFAULT_IMAGE_MODELS));
        assert!(a.allows_for_path("/v1/chat/completions", "deepseek-v4.1-flash"));
        assert!(!a.allows_for_path("/v1/chat/completions", DEFAULT_IMAGE_MODELS));
        assert!(!a.allows_for_path("/v1/models/gemini-3.1-flash-image", DEFAULT_IMAGE_MODELS));
    }

    #[test]
    fn upstream_id_adds_the_prefix_only_for_the_upstream() {
        assert_eq!(
            upstream_image_id("gemini-3.1-flash-image"),
            "cb/gemini-3.1-flash-image"
        );
        assert_eq!(
            upstream_image_id(DEFAULT_IMAGE_MODELS),
            "cb/gemini-3.1-flash-image"
        );
    }

    #[test]
    fn image_route_accepts_only_image_ids() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_images(DEFAULT_IMAGE_MODELS);
        assert!(a.allows_for_path("/v1/images/generations", DEFAULT_IMAGE_MODELS));
        assert!(!a.allows_for_path("/v1/images/generations", "deepseek-v4.1-flash"));
    }

    #[test]
    fn image_list_is_not_listed_as_chat_models() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_images(DEFAULT_IMAGE_MODELS);
        let mut v = json!({"object": "list", "data": [
            {"id": "deepseek-v4.1-flash"},
            {"id": DEFAULT_IMAGE_MODELS},
        ]});
        a.filter_models(&mut v);
        assert_eq!(v["data"].as_array().unwrap().len(), 1);
        assert_eq!(a.image_names(), DEFAULT_IMAGE_MODELS);
    }

    #[test]
    fn images_default_to_off() {
        let a = Allowlist::parse(DEFAULT_MODELS);
        assert!(!a.allows_for_path("/v1/images/generations", DEFAULT_IMAGE_MODELS));
        assert_eq!(a.image_names(), "");
    }

    #[test]
    #[should_panic(expected = "must not overlap")]
    fn overlapping_image_and_chat_ids_are_a_config_error() {
        Allowlist::parse(DEFAULT_MODELS).with_images("deepseek-v4.1-flash");
    }
}
