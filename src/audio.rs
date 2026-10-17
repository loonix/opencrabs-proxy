//! Voice-only fallback upstream.
//!
//! A chat request that carries an `input_audio` part may be answered by a
//! second upstream when every audio model on the primary upstream fails. The
//! fallback is never consulted for a text request, so a chat prompt cannot
//! reach a provider that was configured for voice.
//!
//! Configured with `AUDIO_FALLBACK_BASE_URL`, `AUDIO_FALLBACK_API_KEY` and
//! `AUDIO_FALLBACK_MODELS` (comma list, tried in order). All three are
//! required together: a partial configuration stops the process at startup
//! rather than silently leaving voice without the fallback it was promised.

/// The second upstream a voice request may use, and the models it serves.
#[derive(Clone, Debug)]
pub struct AudioFallback {
    base_url: String,
    key: String,
    /// Tried in order, after every audio id on the primary upstream failed.
    pub models: Vec<String>,
}

impl AudioFallback {
    /// Reads the three variables. All unset means no fallback, which is a
    /// valid deployment. One or two set is a config error: the operator meant
    /// to configure it and would otherwise believe voice has a second leg.
    pub fn from_env() -> Option<Self> {
        Self::from_values(
            std::env::var("AUDIO_FALLBACK_BASE_URL").ok().as_deref(),
            std::env::var("AUDIO_FALLBACK_API_KEY").ok().as_deref(),
            std::env::var("AUDIO_FALLBACK_MODELS").ok().as_deref(),
        )
    }

    pub fn from_values(
        base_url: Option<&str>,
        key: Option<&str>,
        models: Option<&str>,
    ) -> Option<Self> {
        fn clean(v: Option<&str>) -> Option<&str> {
            v.map(str::trim).filter(|s| !s.is_empty())
        }
        let (base_url, key, models) = (clean(base_url), clean(key), clean(models));
        let present = [base_url.is_some(), key.is_some(), models.is_some()]
            .iter()
            .filter(|set| **set)
            .count();
        if present == 0 {
            return None;
        }
        assert_eq!(
            present, 3,
            "AUDIO_FALLBACK_BASE_URL, AUDIO_FALLBACK_API_KEY and AUDIO_FALLBACK_MODELS \
             must be set together"
        );
        let models: Vec<String> = models
            .unwrap()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        assert!(!models.is_empty(), "AUDIO_FALLBACK_MODELS lists no model");
        Some(Self {
            // A base URL may be given with or without the `/v1` suffix, since
            // providers document it both ways. The suffix is stripped here and
            // added back by `chat_url`, so neither form produces `/v1/v1/...`.
            base_url: base_url
                .unwrap()
                .trim_end_matches('/')
                .trim_end_matches("/v1")
                .trim_end_matches('/')
                .to_string(),
            key: key.unwrap().to_string(),
            models,
        })
    }

    /// The chat endpoint on the fallback upstream. Audio rides the chat route
    /// as an `input_audio` part, so there is no separate audio path.
    pub fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    /// For the startup log: which models this leg would serve.
    pub fn describe(&self) -> String {
        self.models.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_is_no_fallback() {
        assert!(AudioFallback::from_values(None, None, None).is_none());
        assert!(AudioFallback::from_values(Some("  "), Some(""), None).is_none());
    }

    #[test]
    fn reads_all_three_and_trims() {
        let f = AudioFallback::from_values(
            Some("https://api-inference.modelscope.ai/v1/"),
            Some(" secret "),
            Some(" a/one , b/two ,"),
        )
        .unwrap();
        assert_eq!(
            f.chat_url(),
            "https://api-inference.modelscope.ai/v1/chat/completions"
        );
        assert_eq!(f.key(), "secret");
        assert_eq!(f.models, vec!["a/one", "b/two"]);
        assert_eq!(f.describe(), "a/one,b/two");
    }

    /// Providers document the base URL both with and without `/v1`. Both
    /// forms must reach the same endpoint, or a correct config 404s.
    #[test]
    fn base_url_is_accepted_with_or_without_the_api_suffix() {
        for base in [
            "https://x.example",
            "https://x.example/",
            "https://x.example/v1",
        ] {
            let f = AudioFallback::from_values(Some(base), Some("k"), Some("a/one")).unwrap();
            assert_eq!(
                f.chat_url(),
                "https://x.example/v1/chat/completions",
                "base `{base}`"
            );
        }
    }

    #[test]
    #[should_panic(expected = "must be set together")]
    fn partial_configuration_is_an_error() {
        AudioFallback::from_values(Some("https://x.example"), None, Some("a/one"));
    }

    #[test]
    #[should_panic(expected = "lists no model")]
    fn empty_model_list_is_an_error() {
        AudioFallback::from_values(Some("https://x.example"), Some("k"), Some(" , "));
    }
}
