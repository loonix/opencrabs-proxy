//! Pass-through for every other `/v1/*` route (embeddings, responses, and so
//! on). The caller gate and the usage row apply as they do for chat. The
//! request body and response body are not inspected, so no token usage is read.

use axum::{
    body::{Body, Bytes},
    extract::{ConnectInfo, State},
    http::{header, HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use std::{net::SocketAddr, sync::Arc, time::Instant};

use crate::{
    allowlist::upstream_image_id, guard, handlers, limit, tracker::RequestRecord, AppState,
};

const ENDPOINT: &str = "/v1/* passthrough";

/// Rewrites the `model` of an image request to the upstream id. Everything else
/// in the body is kept as the caller sent it.
fn upstream_image_body(body: &Bytes) -> Result<Bytes, serde_json::Error> {
    let mut v: serde_json::Value = serde_json::from_slice(body)?;
    let id = v
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or_default()
        .to_string();
    v["model"] = serde_json::Value::String(upstream_image_id(&id));
    serde_json::to_vec(&v).map(Bytes::from)
}

/// The `model` the caller named, if the body is JSON and names one.
fn requested_model(body: &Bytes) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()?
        .get("model")?
        .as_str()
        .map(str::to_string)
}

/// Replaces every upstream image id in an image response with the bare id the
/// caller sent. The upstream route prefix never reaches the caller.
fn unprefix_image_body(body: &[u8], bare_id: &str) -> Vec<u8> {
    let from = upstream_image_id(bare_id);
    let from = from.as_bytes();
    let to = bare_id.as_bytes();
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        if body[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
        } else {
            out.push(body[i]);
            i += 1;
        }
    }
    out
}

/// The caller's path and query, with the path checked for a dot segment.
///
/// WHATWG URL parsing, which reqwest uses, resolves `.` and `..` segments. A
/// path such as `/v1/../../admin` would therefore leave `/v1/` and reach an
/// arbitrary upstream route with our upstream key attached. `%2e` spells a dot
/// too, so each segment is percent-decoded before it is judged. Returns `None`
/// when any decoded segment is `.` or `..`, or carries a byte that would change
/// how the URL parses (`/`, `\`, `#`, `?`, NUL, or a control byte).
fn safe_path_and_query(uri: &Uri) -> Option<String> {
    let raw = uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or_else(|| uri.path());
    let (path, query) = match raw.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (raw, None),
    };
    for segment in path.split('/') {
        if segment.is_empty() {
            continue;
        }
        let decoded = percent_decode(segment);
        if decoded == b"." || decoded == b".." {
            return None;
        }
        if decoded
            .iter()
            .any(|&b| matches!(b, b'/' | b'\\' | b'#' | b'?' | 0) || b < 0x20 || b == 0x7f)
        {
            return None;
        }
    }
    Some(match query {
        Some(query) => format!("{path}?{query}"),
        None => path.to_string(),
    })
}

/// Percent-decode one path segment. A malformed escape is kept verbatim, so a
/// broken `%` sequence cannot smuggle a byte past the segment check.
fn percent_decode(segment: &str) -> Vec<u8> {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(high), Some(low)) = (hex_digit(bytes[i + 1]), hex_digit(bytes[i + 2])) {
                out.push(high << 4 | low);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub async fn passthrough(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(path_and_query) = safe_path_and_query(&uri) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if !path_and_query.starts_with("/v1/") {
        return StatusCode::NOT_FOUND.into_response();
    }
    let ip = guard::client_ip(&headers, peer.ip()).to_string();

    let caller = match guard::gate(&state, &headers) {
        Ok(label) => label,
        Err(rejection) => {
            handlers::rejected(&state, ip, ENDPOINT, rejection.status());
            return *rejection;
        }
    };

    // Computed once, so the same number both gates the request and lands in
    // its stats row.
    let estimate = limit::estimate_tokens(&body);
    let factor = state.calibration.factor(&caller);
    if state.limit.exceeded_scaled(estimate, factor) {
        let scaled = (estimate as f64 * factor).ceil() as u64;
        return handlers::refuse_input(&state, ip, caller, ENDPOINT, scaled, body.len());
    }

    // Only allowlisted model ids may be named, in the path or in the JSON body.
    // Image routes take the image list, every other route the chat list.
    if let Some(id) = uri.path().strip_prefix("/v1/models/") {
        if !state.allow.allows_for_path(uri.path(), id) {
            return handlers::refuse_model(&state, ip, caller, ENDPOINT, id);
        }
    }
    // An image route must name an image id: a body with no model is refused,
    // so it cannot reach upstream on a default model.
    if uri.path().starts_with("/v1/images/") {
        let named = serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("model").and_then(|m| m.as_str().map(str::to_string)));
        if named
            .as_deref()
            .is_none_or(|m| !state.allow.allows_image(m))
        {
            let refused = named.unwrap_or_default();
            return handlers::refuse_model(&state, ip, caller, ENDPOINT, &refused);
        }
    }
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body) {
        if let Some(model) = v.get("model") {
            let model = model.as_str().unwrap_or("");
            if !state.allow.allows_for_path(uri.path(), model) {
                return handlers::refuse_model(&state, ip, caller, ENDPOINT, model);
            }
        }
    }

    // Image routes: the bare id the caller sent was checked above. It becomes the
    // upstream id only here, so the gates never see the upstream prefix.
    let image_bare = if uri.path().starts_with("/v1/images/") {
        requested_model(&body)
    } else {
        None
    };
    let body = if image_bare.is_some() {
        match upstream_image_body(&body) {
            Ok(rewritten) => rewritten,
            Err(e) => {
                tracing::error!("image body rewrite failed after the gate passed: {e}");
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
        }
    } else {
        body
    };

    let Ok(upstream_method) = reqwest::Method::from_bytes(method.as_str().as_bytes()) else {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    };
    let url = format!("{}{}", state.upstream, path_and_query);

    let (key_label, upstream_key) = state.keys.active();
    let mut req = state
        .client
        .request(upstream_method, &url)
        .header(header::AUTHORIZATION, format!("Bearer {upstream_key}"))
        .body(body);
    for name in [header::CONTENT_TYPE, header::ACCEPT] {
        if let Some(value) = headers.get(&name) {
            req = req.header(name, value.clone());
        }
    }

    let start = Instant::now();
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("passthrough upstream error: {e}");
            state.tracker.record(RequestRecord {
                ip,
                token: caller,
                endpoint: ENDPOINT,
                status: 502,
                latency_ms: start.elapsed().as_millis() as u64,
                failure_reason: Some("transport"),
                key_label: key_label.clone(),
                ..Default::default()
            });
            return (StatusCode::BAD_GATEWAY, "upstream error").into_response();
        }
    };

    let status = resp.status();
    let latency_ms = start.elapsed().as_millis() as u64;
    state.tracker.record(RequestRecord {
        ip,
        token: caller,
        endpoint: ENDPOINT,
        status: status.as_u16(),
        latency_ms,
        estimated_tokens: Some(estimate),
        key_label,
        ..Default::default()
    });

    // Headers are recorded above; the body streams back as the upstream sends it.
    let mut out = Response::builder().status(status);
    if let Some(ct) = resp.headers().get(header::CONTENT_TYPE) {
        out = out.header(header::CONTENT_TYPE, ct.clone());
    }
    // Image responses are buffered so the upstream id can be replaced before the
    // body leaves the proxy. Every other route streams as the upstream sends it.
    if let Some(bare) = image_bare {
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("image response read error: {e}");
                return (StatusCode::BAD_GATEWAY, "upstream error").into_response();
            }
        };
        return match out.body(Body::from(unprefix_image_body(&bytes, &bare))) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("image response build error: {e}");
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        };
    }
    match out.body(Body::from_stream(resp.bytes_stream())) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("passthrough response build error: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_body_gets_the_upstream_id_and_keeps_the_rest() {
        let body = Bytes::from_static(br#"{"model":"gemini-3.1-flash-image","prompt":"a crab"}"#);
        let out: serde_json::Value =
            serde_json::from_slice(&upstream_image_body(&body).unwrap()).unwrap();
        assert_eq!(out["model"], "cb/gemini-3.1-flash-image");
        assert_eq!(out["prompt"], "a crab");
    }

    #[test]
    fn image_response_drops_the_upstream_prefix() {
        let body = br#"{"model":"cb/gemini-3.1-flash-image","data":[{"b64_json":"QUJD"}]}"#;
        let out = unprefix_image_body(body, "gemini-3.1-flash-image");
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains(r#""model":"gemini-3.1-flash-image""#));
        assert!(!out.contains("cb/"));
        assert!(out.contains("QUJD"));
    }

    #[test]
    fn image_response_without_the_prefix_is_unchanged() {
        let body = br#"{"model":"gemini-3.1-flash-image","data":[]}"#;
        let out = unprefix_image_body(body, "gemini-3.1-flash-image");
        assert_eq!(out, body.to_vec());
    }

    fn path(raw: &str) -> Option<String> {
        safe_path_and_query(&raw.parse::<Uri>().unwrap())
    }

    #[test]
    fn dot_segments_are_refused() {
        assert_eq!(path("/v1/../../admin"), None);
        assert_eq!(path("/v1/../admin"), None);
        assert_eq!(path("/v1/./admin"), None);
        assert_eq!(path("/v1/%2e%2e/%2e%2e/admin"), None);
        assert_eq!(path("/v1/%2E%2E/admin"), None);
        assert_eq!(path("/v1/%2e/admin"), None);
    }

    #[test]
    fn encoded_separators_and_controls_are_refused() {
        assert_eq!(path("/v1/%2fadmin"), None);
        assert_eq!(path("/v1/%5c.."), None);
        assert_eq!(path("/v1/a%00b"), None);
    }

    #[test]
    fn normal_paths_pass_through_unchanged() {
        assert_eq!(
            path("/v1/chat/completions").as_deref(),
            Some("/v1/chat/completions")
        );
        assert_eq!(
            path("/v1/images/generations").as_deref(),
            Some("/v1/images/generations")
        );
        assert_eq!(
            path("/v1/models/gemini-3.1-flash-image").as_deref(),
            Some("/v1/models/gemini-3.1-flash-image")
        );
    }

    #[test]
    fn query_is_preserved() {
        assert_eq!(
            path("/v1/embeddings?x=1&y=..%2f").as_deref(),
            Some("/v1/embeddings?x=1&y=..%2f")
        );
    }
}
