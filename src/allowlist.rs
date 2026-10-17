//! Model allowlist. Only the listed upstream model ids are served: a request
//! naming any other model is refused before it reaches the upstream, and
//! `/v1/models` lists only these ids. Set with `ALLOWED_MODELS` (comma list).
//!
//! Image generation has its own list, `IMAGE_MODELS`. Its ids are served only
//! on `/v1/images/*`, and chat or `/v1/models` refuse them, so an image model
//! can never be used as a chat model.
//!
//! Audio input has its own list, `AUDIO_MODELS`, and it is narrower still: an
//! id on it is served only when a chat request carries an `input_audio` part,
//! and only on `/v1/audio/*` passthrough. A text request naming an audio model
//! is refused, and an audio request naming a text model is refused. The three
//! lists must not overlap.
//!
//! A capability shows the client one id. The chat list carries one id and the
//! image list carries one id, so a picker built from `/v1/models`, or a
//! `generation_model` set once on the client, can never offer a second
//! spelling of the same model. Publisher-prefixed routing ids (`ali/...`,
//! `cb/...`) belong in `ROTATION_MODELS`, where the proxy walks them and the
//! client never sees them. Voice is the exception: its list is a chain, tried
//! in order, and the caller names only the first id.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

/// Default chat ids served on `/v1/chat/completions`. One id, in the exact
/// spelling the upstream lists it: the proxy rotates between its own upstream
/// keys behind that id, so a caller never picks a publisher and never sees a
/// second spelling of the same model. Set with `ALLOWED_MODELS`.
pub const DEFAULT_MODELS: &str = "deepseek-v4.1-flash";

/// Image generation ids served on `/v1/images/*` only. One id, in the exact
/// spelling the upstream lists it. Set with `IMAGE_MODELS`; empty turns image
/// generation off.
pub const DEFAULT_IMAGE_MODELS: &str = "gemini-2.5-flash-image";

/// Audio input ids. An id here is served only when a chat request carries an
/// `input_audio` part, and only on `/v1/audio/*` passthrough. It is never a
/// chat model, so a text prompt cannot be answered by a voice model. The
/// first id is tried first, the rest are fallbacks in order. Set with
/// `AUDIO_MODELS`.
pub const DEFAULT_AUDIO_MODELS: &str = "mimo/mimo-v2.6-flash";

/// Path prefix that selects the image list instead of the chat list.
const IMAGE_PATH_PREFIX: &str = "/v1/images/";

/// Path prefix that selects the audio list instead of the chat list.
const AUDIO_PATH_PREFIX: &str = "/v1/audio/";

#[derive(Clone, Debug)]
pub struct Allowlist {
    models: Vec<String>,
    images: Vec<String>,
    audio: Vec<String>,
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
            audio: Vec::new(),
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

    /// Attach the audio input list. Empty means audio input is off. Panics if
    /// an id is also a chat or image model, since an audio id must never be
    /// reachable from a text request.
    pub fn with_audio(mut self, raw: &str) -> Self {
        let audio = split_ids(raw);
        for id in &audio {
            assert!(
                !self.models.contains(id),
                "AUDIO_MODELS id `{id}` is also a chat model; the lists must not overlap"
            );
            assert!(
                !self.images.contains(id),
                "AUDIO_MODELS id `{id}` is also an image model; the lists must not overlap"
            );
        }
        self.audio = audio;
        self
    }

    pub fn names(&self) -> String {
        self.models.join(",")
    }

    /// The chat ids, in configured order. Used where a slice is needed rather
    /// than the joined string, such as the health probe's fallback targets.
    pub fn chat_candidates(&self) -> &[String] {
        &self.models
    }

    pub fn image_names(&self) -> String {
        self.images.join(",")
    }

    pub fn audio_names(&self) -> String {
        self.audio.join(",")
    }

    pub fn allows(&self, model: &str) -> bool {
        self.models.iter().any(|m| m == model)
    }

    pub fn allows_image(&self, model: &str) -> bool {
        self.images.iter().any(|m| m == model)
    }

    pub fn allows_audio(&self, model: &str) -> bool {
        self.audio.iter().any(|m| m == model)
    }

    /// The models an audio request may be tried on, the requested id first and
    /// the remaining audio ids after it as fallbacks. `None` when the id is not
    /// on the audio list, so an audio request can never be served by a chat
    /// model and a text request can never be served by an audio one.
    pub fn audio_plan_for(&self, model: &str) -> Option<Vec<String>> {
        if !self.allows_audio(model) {
            return None;
        }
        let mut plan = vec![model.to_string()];
        plan.extend(self.audio.iter().filter(|m| m.as_str() != model).cloned());
        Some(plan)
    }

    /// Route-aware check: image routes accept only image ids, audio routes
    /// only audio ids, every other route only chat ids.
    pub fn allows_for_path(&self, path: &str, model: &str) -> bool {
        if path.starts_with(IMAGE_PATH_PREFIX) {
            self.allows_image(model)
        } else if path.starts_with(AUDIO_PATH_PREFIX) {
            self.allows_audio(model)
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
        assert!(!a.allows("cb/deepseek-v4.1-flash"));
        assert!(!a.allows("ali/deepseek-v4.1-flash"));
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
            {"id": "cb/deepseek-v4.1-flash"},
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

    /// The image id the shipped defaults carry.
    const IMG: &str = "gemini-2.5-flash-image";
    /// One audio id, since the list may carry fallbacks.
    const AUD: &str = "mimo/mimo-v2.6-flash";

    #[test]
    fn image_id_is_never_a_chat_model() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_images(DEFAULT_IMAGE_MODELS);
        assert!(!a.allows(IMG));
        assert!(a.allows_for_path("/v1/chat/completions", DEFAULT_MODELS));
        assert!(!a.allows_for_path("/v1/chat/completions", IMG));
        assert!(!a.allows_for_path(&format!("/v1/models/{IMG}"), IMG));
    }

    #[test]
    fn image_route_accepts_only_image_ids() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_images(DEFAULT_IMAGE_MODELS);
        assert!(a.allows_for_path("/v1/images/generations", IMG));
        assert!(a.allows_for_path("/v1/images/generations", DEFAULT_IMAGE_MODELS));
        // A prefixed spelling of the same image model is not a second id.
        assert!(!a.allows_for_path("/v1/images/generations", "cb/gemini-2.5-flash-image"));
        assert!(!a.allows_for_path("/v1/images/generations", DEFAULT_MODELS));
    }

    #[test]
    fn image_list_is_not_listed_as_chat_models() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_images(DEFAULT_IMAGE_MODELS);
        let mut v = json!({"object": "list", "data": [
            {"id": DEFAULT_MODELS},
            {"id": IMG},
            {"id": AUD},
        ]});
        a.filter_models(&mut v);
        assert_eq!(v["data"].as_array().unwrap().len(), 1);
        assert_eq!(a.image_names(), DEFAULT_IMAGE_MODELS);
    }

    #[test]
    fn images_default_to_off() {
        let a = Allowlist::parse(DEFAULT_MODELS);
        assert!(!a.allows_for_path("/v1/images/generations", IMG));
        assert_eq!(a.image_names(), "");
    }

    #[test]
    fn audio_id_is_never_a_chat_model() {
        let a = Allowlist::parse(DEFAULT_MODELS)
            .with_images(DEFAULT_IMAGE_MODELS)
            .with_audio(DEFAULT_AUDIO_MODELS);
        assert!(a.allows_audio(AUD));
        assert!(!a.allows(AUD));
        assert!(!a.allows_for_path("/v1/chat/completions", AUD));
        assert!(a.audio_plan_for(AUD).is_some());
        assert!(a.audio_plan_for(DEFAULT_MODELS).is_none());
    }

    #[test]
    fn audio_plan_puts_the_requested_id_first() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_audio("a/one,a/two,a/three");
        assert_eq!(
            a.audio_plan_for("a/two").unwrap(),
            vec!["a/two", "a/one", "a/three"]
        );
        assert_eq!(
            a.audio_plan_for("a/one").unwrap(),
            vec!["a/one", "a/two", "a/three"]
        );
    }

    #[test]
    fn audio_route_accepts_only_audio_ids() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_audio(DEFAULT_AUDIO_MODELS);
        assert!(a.allows_for_path("/v1/audio/transcriptions", AUD));
        assert!(!a.allows_for_path("/v1/audio/transcriptions", DEFAULT_MODELS));
    }

    /// What a client picks from is one id per capability: one chat id and one
    /// image id. A second spelling of a model belongs in `ROTATION_MODELS`,
    /// out of the client's sight, which is where the proxy walks it.
    #[test]
    fn shipped_defaults_show_one_id_per_capability() {
        let a = Allowlist::parse(DEFAULT_MODELS).with_images(DEFAULT_IMAGE_MODELS);
        assert_eq!(a.names(), "deepseek-v4.1-flash");
        assert_eq!(a.image_names(), "gemini-2.5-flash-image");
        let mut v = json!({"object": "list", "data": [
            {"id": "deepseek-v4.1-flash"},
            {"id": "ali/deepseek-v4.1-flash"},
            {"id": "cb/deepseek-v4.1-flash"},
        ]});
        a.filter_models(&mut v);
        assert_eq!(v["data"].as_array().unwrap().len(), 1);
    }

    #[test]
    #[should_panic(expected = "must not overlap")]
    fn overlapping_audio_and_chat_ids_are_a_config_error() {
        Allowlist::parse(DEFAULT_MODELS).with_audio(DEFAULT_MODELS);
    }

    #[test]
    #[should_panic(expected = "must not overlap")]
    fn overlapping_image_and_chat_ids_are_a_config_error() {
        Allowlist::parse(DEFAULT_MODELS).with_images(DEFAULT_MODELS);
    }
}
