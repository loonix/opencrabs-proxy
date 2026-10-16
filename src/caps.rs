//! Per-caller token caps, counted per UTC day and per UTC month.
//!
//! A cap of `0` means no limit for that window. The caps are read from
//! `DAILY_TOKEN_CAP` and `MONTHLY_TOKEN_CAP`. They apply to every caller alike.
//! The check runs before a request is forwarded, so one request can take a
//! caller past its cap. The next request is then refused.

/// Token caps for one caller. `0` means unlimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Caps {
    pub daily: u64,
    pub monthly: u64,
}

impl Caps {
    /// Read the caps from the environment. Unset or unparsable values are `0`.
    pub fn from_env() -> Self {
        Self {
            daily: read_u64("DAILY_TOKEN_CAP"),
            monthly: read_u64("MONTHLY_TOKEN_CAP"),
        }
    }

    /// The window that is over its cap, if any. Daily is checked first.
    pub fn exceeded(&self, used_today: u64, used_this_month: u64) -> Option<&'static str> {
        if self.daily > 0 && used_today >= self.daily {
            return Some("daily token cap reached");
        }
        if self.monthly > 0 && used_this_month >= self.monthly {
            return Some("monthly token cap reached");
        }
        None
    }
}

fn read_u64(name: &str) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_means_unlimited() {
        let caps = Caps {
            daily: 0,
            monthly: 0,
        };
        assert_eq!(caps.exceeded(u64::MAX, u64::MAX), None);
    }

    #[test]
    fn daily_cap_refuses_at_the_limit() {
        let caps = Caps {
            daily: 100,
            monthly: 0,
        };
        assert_eq!(caps.exceeded(99, 0), None);
        assert_eq!(caps.exceeded(100, 0), Some("daily token cap reached"));
    }

    #[test]
    fn monthly_cap_refuses_at_the_limit() {
        let caps = Caps {
            daily: 0,
            monthly: 500,
        };
        assert_eq!(caps.exceeded(0, 499), None);
        assert_eq!(caps.exceeded(0, 500), Some("monthly token cap reached"));
    }

    #[test]
    fn daily_is_reported_before_monthly() {
        let caps = Caps {
            daily: 10,
            monthly: 10,
        };
        assert_eq!(caps.exceeded(10, 10), Some("daily token cap reached"));
    }
}
