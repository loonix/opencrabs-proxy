//! Ids a client may still name that this proxy does not serve under that
//! spelling, and what such an id resolves to.
//!
//! The published OpenCrabs config and docs name `gemini-3.1-flash-image` for
//! image generation and `qwen3.8-omni-flash` for voice. Neither is an id the
//! upstream behind this proxy serves, so a client built against those docs asks
//! for a model the proxy could serve and gets a refusal instead. This table
//! closes that gap: the request is served on the id the proxy carries for the
//! capability the id names.
//!
//! Two rules keep the table safe:
//!
//! 1. An alias is scoped to one capability and resolves only inside it. A voice
//!    id can never resolve onto an image id, and an alias resolves to nothing
//!    when the proxy serves nothing for that capability, so `IMAGE_MODELS`
//!    empty keeps image generation off whatever the caller names.
//! 2. An alias is accepted input, never advertised output. `/v1/models` lists
//!    the served id, so a picker fed by it still shows one id per capability.
//!
//! Note that the table holds ids that shipped in client-facing docs. A
//! publisher-prefixed routing id (`cb/...`, `ali/...`) is not an alias: those
//! stay in `ROTATION_MODELS`, where the proxy walks them and the client never
//! names them.

/// The capability an id belongs to, and the scope an alias resolves within.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// Text and vision: `/v1/chat/completions` without an audio part.
    Chat,
    /// Image generation: `/v1/images/*`.
    Image,
    /// Voice: an audio-carrying chat request, and `/v1/audio/*`.
    Audio,
}

/// An id published to clients, and the capability it names. The served id comes
/// from the allowlist at resolution time, so this table never restates what the
/// proxy serves: it only says which capability the named id belongs to.
pub const ALIASES: &[(&str, Capability)] = &[
    ("gemini-3.1-flash-image", Capability::Image),
    ("qwen3.8-omni-flash", Capability::Audio),
];

/// True when `model` is an id published to clients for `capability`.
pub fn is_alias(model: &str, capability: Capability) -> bool {
    ALIASES
        .iter()
        .any(|(alias, cap)| *alias == model && *cap == capability)
}

/// The id to serve when `model` is an alias for `capability`. `served` is the
/// id this proxy carries for that capability, or `None` when it serves nothing
/// for it, which leaves an alias unresolved and the request refused.
pub fn resolve(model: &str, capability: Capability, served: Option<&str>) -> Option<String> {
    match served {
        Some(served) if is_alias(model, capability) => Some(served.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_alias_resolves_to_the_served_id() {
        assert_eq!(
            resolve(
                "gemini-3.1-flash-image",
                Capability::Image,
                Some("gemini-2.5-flash-image")
            ),
            Some("gemini-2.5-flash-image".to_string())
        );
        assert_eq!(
            resolve(
                "qwen3.8-omni-flash",
                Capability::Audio,
                Some("mimo/mimo-v2.6-flash")
            ),
            Some("mimo/mimo-v2.6-flash".to_string())
        );
    }

    #[test]
    fn an_alias_does_not_cross_capabilities() {
        assert_eq!(
            resolve("qwen3.8-omni-flash", Capability::Image, Some("img")),
            None
        );
        assert_eq!(
            resolve("gemini-3.1-flash-image", Capability::Audio, Some("voice")),
            None
        );
        assert_eq!(
            resolve("qwen3.8-omni-flash", Capability::Chat, Some("chat")),
            None
        );
    }

    #[test]
    fn an_alias_resolves_to_nothing_when_the_capability_is_off() {
        assert_eq!(
            resolve("gemini-3.1-flash-image", Capability::Image, None),
            None
        );
        assert_eq!(resolve("qwen3.8-omni-flash", Capability::Audio, None), None);
    }

    #[test]
    fn an_unknown_id_is_not_an_alias() {
        assert!(!is_alias("gpt-4", Capability::Chat));
        assert_eq!(
            resolve("gpt-4", Capability::Chat, Some("deepseek-v4.1-flash")),
            None
        );
        // A publisher-prefixed routing id is never an alias.
        assert_eq!(
            resolve(
                "cb/gemini-2.5-flash-image",
                Capability::Image,
                Some("gemini-2.5-flash-image")
            ),
            None
        );
    }
}
