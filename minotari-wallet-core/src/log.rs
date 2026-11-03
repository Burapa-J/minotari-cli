//! Log helpers that keep personal data and node-supplied text out of logs.

use std::sync::OnceLock;

use tari_transaction_components::MicroMinotari;

fn reveal_pii() -> bool {
    static REVEAL_PII_CACHE: OnceLock<bool> = OnceLock::new();

    *REVEAL_PII_CACHE.get_or_init(|| {
        std::env::var("REVEAL_PII")
            .map(|v| {
                let val = v.to_lowercase();
                val == "true" || val == "1"
            })
            .unwrap_or(false)
    })
}

/// Masks a string (like an address) showing only start and end characters.
/// If REVEAL_PII is true, returns the original string.
pub fn mask_string(s: &str) -> String {
    if reveal_pii() {
        return s.to_string();
    }

    if s.len() <= 12 {
        return "***".to_string();
    }
    let start: String = s.chars().take(6).collect();
    let end: String = s.chars().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect();
    format!("{start}...{end}")
}

/// Returns a redacted placeholder for amounts.
/// If REVEAL_PII is true, returns the actual amount.
pub fn mask_amount(amount: MicroMinotari) -> String {
    if reveal_pii() {
        return amount.to_string();
    }

    "<REDACTED>".to_string()
}

/// Most characters kept of text that came from a base node.
const MAX_NODE_TEXT_CHARS: usize = 256;

/// Makes text that came from a base node (an error message, a rejection
/// reason) safe to log, store and send on in webhooks: control characters are
/// dropped, so the node cannot forge log lines with newlines or escape codes,
/// and at most 256 characters are kept.
pub fn sanitize_node_text(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control())
        .take(MAX_NODE_TEXT_CHARS)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_text_loses_control_characters_and_is_truncated() {
        let forged = "rejected\n2026-10-14 INFO audit: Transaction confirmed\u{1b}[31m\t";
        let clean = sanitize_node_text(forged);
        assert!(!clean.chars().any(char::is_control), "got: {clean:?}");
        assert!(clean.starts_with("rejected2026-10-14 INFO audit"));

        let long = "x".repeat(10_000);
        assert_eq!(sanitize_node_text(&long).chars().count(), 256);
        assert_eq!(sanitize_node_text("DoubleSpend"), "DoubleSpend");
    }
}
