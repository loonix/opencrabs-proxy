use std::{
    collections::HashMap,
    hash::Hash,
    sync::Mutex,
    time::{Duration, Instant},
};

/// Fixed-window rate limiter keyed by any caller identity (the token label).
pub struct RateLimiter<K> {
    inner: Mutex<HashMap<K, (Instant, u32)>>,
    max: u32,
    window: Duration,
}

impl<K: Hash + Eq + Clone> RateLimiter<K> {
    pub fn new(max: u32, window: Duration) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            max,
            window,
        }
    }

    pub fn allow(&self, key: K) -> bool {
        let now = Instant::now();
        let mut map = self.inner.lock().unwrap();
        // Keep the map from growing unbounded over the window's lifetime.
        if map.len() > 4096 {
            map.retain(|_, (start, _)| now.duration_since(*start) <= self.window);
        }
        let entry = map.entry(key).or_insert((now, 0));
        if now.duration_since(entry.0) > self.window {
            *entry = (now, 0);
        }
        entry.1 += 1;
        entry.1 <= self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_up_to_max_then_refuses() {
        let rl = RateLimiter::new(2, Duration::from_secs(60));
        assert!(rl.allow("alice".to_string()));
        assert!(rl.allow("alice".to_string()));
        assert!(!rl.allow("alice".to_string()));
    }

    #[test]
    fn daily_install_cap_refuses_past_max_per_ip_and_isolates_ips() {
        let rl = RateLimiter::new(5000, Duration::from_secs(86_400));
        for _ in 0..5000 {
            assert!(rl.allow("203.0.113.7".to_string()));
        }
        assert!(!rl.allow("203.0.113.7".to_string()));
        assert!(rl.allow("198.51.100.9".to_string()));
    }

    #[test]
    fn keys_are_independent() {
        let rl = RateLimiter::new(1, Duration::from_secs(60));
        assert!(rl.allow("alice".to_string()));
        assert!(!rl.allow("alice".to_string()));
        assert!(rl.allow("bob".to_string()));
    }
}
