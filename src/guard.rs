use axum::{
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::net::IpAddr;

use crate::AppState;

/// Client IP, used for logs only. nginx appends the real client address to
/// `X-Forwarded-For`, so the rightmost entry is the one to trust. A client can
/// put anything to the left of it. Falls back to the socket peer.
pub fn client_ip(headers: &HeaderMap, peer: IpAddr) -> IpAddr {
    headers
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.rsplit(',').next())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(peer)
}

/// The bearer token from `Authorization: Bearer <token>`, if present.
pub fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|t| !t.is_empty())
}

/// Runs every gate. `Ok` carries the caller's label. `Err` is the response to
/// send. Auth is mandatory: a missing or unknown token is refused.
pub fn gate(state: &AppState, headers: &HeaderMap) -> Result<String, Box<Response>> {
    let Some(token) = bearer(headers) else {
        return Err(Box::new(
            (StatusCode::UNAUTHORIZED, "missing bearer token").into_response(),
        ));
    };
    // The lock is held only for the lookup, and the label is copied out of it.
    let found = state
        .tokens
        .read()
        .expect("tokens lock poisoned")
        .lookup(token)
        .map(str::to_string);
    let Some(label) = found else {
        return Err(Box::new(
            (StatusCode::UNAUTHORIZED, "invalid token").into_response(),
        ));
    };
    if !state.rate.allow(label.clone()) {
        return Err(Box::new(
            (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded").into_response(),
        ));
    }
    if let Some(reason) = over_cap(state, &label) {
        return Err(Box::new(
            (StatusCode::TOO_MANY_REQUESTS, reason).into_response(),
        ));
    }
    Ok(label)
}

/// The cap message when `label` is over its daily or monthly token cap.
fn over_cap(state: &AppState, label: &str) -> Option<&'static str> {
    if state.caps.daily == 0 && state.caps.monthly == 0 {
        return None;
    }
    let today = state.tracker.tokens_used_since(label, "start of day");
    let month = state.tracker.tokens_used_since(label, "start of month");
    state.caps.exceeded(today, month)
}

/// True when the request carries the admin token.
pub fn admin_ok(headers: &HeaderMap, expected: &str) -> bool {
    bearer(headers) == Some(expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers_with(name: &'static str, value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(name, HeaderValue::from_str(value).unwrap());
        h
    }

    #[test]
    fn client_ip_uses_rightmost_forwarded_entry() {
        // A client-supplied address on the left must not win.
        let h = headers_with("x-forwarded-for", "10.0.0.1, 203.0.113.9");
        let peer: IpAddr = "127.0.0.1".parse().unwrap();
        assert_eq!(client_ip(&h, peer).to_string(), "203.0.113.9");
    }

    #[test]
    fn client_ip_falls_back_to_peer() {
        let peer: IpAddr = "127.0.0.1".parse().unwrap();
        assert_eq!(client_ip(&HeaderMap::new(), peer), peer);
    }

    #[test]
    fn bearer_parses_token() {
        let h = headers_with("authorization", "Bearer abc123");
        assert_eq!(bearer(&h), Some("abc123"));
    }

    #[test]
    fn bearer_rejects_other_schemes_and_empty() {
        assert_eq!(bearer(&headers_with("authorization", "Basic abc")), None);
        assert_eq!(bearer(&headers_with("authorization", "Bearer ")), None);
        assert_eq!(bearer(&HeaderMap::new()), None);
    }

    #[test]
    fn opencrabs_token_header_no_longer_authenticates() {
        // The old x-opencrabs-token header was a shared, spoofable gate.
        let h = headers_with("x-opencrabs-token", "abc123");
        assert_eq!(bearer(&h), None);
    }

    #[test]
    fn admin_ok_needs_exact_bearer() {
        assert!(admin_ok(
            &headers_with("authorization", "Bearer adm"),
            "adm"
        ));
        assert!(!admin_ok(
            &headers_with("authorization", "Bearer nope"),
            "adm"
        ));
        assert!(!admin_ok(&HeaderMap::new(), "adm"));
    }
}
