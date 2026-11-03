//! Per-process key sessions.
//!
//! [`unlock`] decrypts and parses the account's wallet once and keeps it in memory that wipes
//! itself, so later address, scan, send and signing calls for that account need neither the
//! passphrase nor any Argon2 work. [`lock`] stops new operations from using the session and drops
//! the cached keys.
//!
//! Sessions are keyed by account name and tied to the ciphertext they were decrypted from, so a
//! session never applies to a different row (another database, or a re-encrypted wallet).
//!
//! Sessions have no expiry: the host decides their lifetime and must lock on close or background.
//!
//! The only logging here is audit lines carrying the account name. [`UnlockedAccount`] is not
//! `Debug`, `Clone` or `Serialize`, so no key material can be formatted into a log line.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError, RwLock};

use log::info;
use rusqlite::Connection;
use tari_transaction_components::key_manager::wallet_types::WalletType;

use crate::db::{AccountRow, WalletDbError, WalletDbResult, get_account_by_name};

/// An account's decrypted keys. Its secret keys and cipher seed zeroize themselves on drop.
pub struct UnlockedAccount {
    encrypted_wallet: Vec<u8>,
    wallet: WalletType,
}

static SESSIONS: LazyLock<RwLock<HashMap<String, Arc<UnlockedAccount>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

// Held by every change to the session map. `unlock` takes it only to store, after its Argon2
// work; passphrase changes, deletes and renames hold it throughout. Key paths never wait on it.
static CHANGES: LazyLock<Mutex<LockCounts>> = LazyLock::new(|| Mutex::new(LockCounts::default()));

/// How many times each account, and all accounts, have been locked, so an unlock that was in
/// flight during a lock does not store.
#[derive(Default)]
pub(crate) struct LockCounts {
    all: u64,
    by_name: HashMap<String, u64>,
}

impl LockCounts {
    fn for_name(&self, account_name: &str) -> (u64, u64) {
        (self.all, self.by_name.get(account_name).copied().unwrap_or(0))
    }
}

/// Unlocks `account_name` with `passphrase`, running Argon2 once.
///
/// This does not authenticate: it returns `Ok` without checking `passphrase` when the account is
/// already unlocked. Use [`AccountRow::verify_passphrase`] to check a passphrase.
///
/// Returns [`WalletDbError::InvalidPassphrase`] on a wrong passphrase and stores nothing. Returns
/// [`WalletDbError::NotUnlocked`] and stores nothing if a lock or a change to the account ran
/// while it was unlocking.
pub fn unlock(conn: &Connection, account_name: &str, passphrase: &str) -> WalletDbResult<()> {
    let locks_before = changes_guard().for_name(account_name);
    let account = get_account_by_name(conn, account_name)?
        .ok_or_else(|| WalletDbError::InvalidInput(format!("Account '{}' not found", account_name)))?;
    if session_for(&account).is_some() {
        return Ok(());
    }

    let wallet_json = match account.decrypt_wallet_json(passphrase) {
        Ok(wallet_json) => wallet_json,
        Err(WalletDbError::DecryptionFailed(_)) => {
            info!(target: "audit", account = account_name; "Key session unlock failed: wrong passphrase");
            return Err(WalletDbError::InvalidPassphrase);
        },
        Err(e) => return Err(e),
    };
    // Also checks the keys belong to this account before trusting them.
    let wallet = account.parse_wallet_json(&wallet_json)?;

    let changes = changes_guard();
    if session_for(&account).is_some() {
        return Ok(());
    }
    let current = get_account_by_name(conn, account_name)?;
    let unchanged = current.is_some_and(|current| current.encrypted_wallet == account.encrypted_wallet);
    if !unchanged || changes.for_name(account_name) != locks_before {
        return Err(WalletDbError::NotUnlocked(account_name.to_string()));
    }
    store(account_name, account.encrypted_wallet, wallet);
    info!(target: "audit", account = account_name; "Key session unlocked");
    Ok(())
}

/// Stops new operations from using `account_name`'s session and drops the cached keys.
///
/// Removes by name, whether or not [`is_unlocked`] reports it, so call it unconditionally on
/// logout or close. Copies of the keys already handed to running scans or sends are not revoked;
/// cancel those. May wait for an in-progress passphrase change, delete or rename.
pub fn lock(account_name: &str) {
    let mut changes = changes_guard();
    let count = changes.by_name.entry(account_name.to_string()).or_default();
    *count = count.wrapping_add(1);
    if take(account_name).is_some() {
        info!(target: "audit", account = account_name; "Key session locked");
    }
}

/// Like [`lock`], for every session.
pub fn lock_all() {
    let mut changes = changes_guard();
    changes.all = changes.all.wrapping_add(1);
    SESSIONS.write().unwrap_or_else(PoisonError::into_inner).clear();
    info!(target: "audit", "All key sessions locked");
}

/// Whether `account_name` in this database has a session the key paths will use.
pub fn is_unlocked(conn: &Connection, account_name: &str) -> WalletDbResult<bool> {
    let account = get_account_by_name(conn, account_name)?;
    Ok(account.is_some_and(|account| session_for(&account).is_some()))
}

/// The wallet from `account`'s session, or `None` when it has no session for this row.
pub(crate) fn unlocked_wallet_type(account: &AccountRow) -> Option<WalletType> {
    session_for(account).map(|session| session.wallet.clone())
}

/// Serializes changes to the session map. Take it before any SQLite write lock, and hold it from
/// reading the row through `store` / `remove_session_for`.
pub(crate) fn changes_guard() -> MutexGuard<'static, LockCounts> {
    CHANGES.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn take(account_name: &str) -> Option<Arc<UnlockedAccount>> {
    SESSIONS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(account_name)
}

/// Stores a session for a wallet already decrypted from `encrypted_wallet` and checked
/// against the account's fingerprint.
pub(crate) fn store(account_name: &str, encrypted_wallet: Vec<u8>, wallet: WalletType) {
    let session = UnlockedAccount {
        encrypted_wallet,
        wallet,
    };
    SESSIONS
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(account_name.to_string(), Arc::new(session));
}

/// Whether `account` (as read from its database) has a session.
pub(crate) fn has_session(account: &AccountRow) -> bool {
    session_for(account).is_some()
}

/// Removes `account`'s session, leaving a same-named account's session from another database alone.
pub(crate) fn remove_session_for(account: &AccountRow) {
    if has_session(account) {
        take(&account.friendly_name);
    }
}

fn session_for(account: &AccountRow) -> Option<Arc<UnlockedAccount>> {
    let sessions = SESSIONS.read().unwrap_or_else(PoisonError::into_inner);
    let session = sessions.get(&account.friendly_name)?;
    if session.encrypted_wallet == account.encrypted_wallet {
        Some(Arc::clone(session))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::WALLET_PARSES;
    use crate::db::{change_passphrase, create_account, init_db};
    use crate::signing::{SigningError, sign_message, sign_message_unlocked, verify_message};
    use crate::utils::crypto::KEY_DERIVATIONS;
    use crate::utils::delete_wallet::{delete_wallet, delete_wallet_with_conn};
    use crate::utils::rename_wallet::rename_wallet_with_conn;
    use tari_common::configuration::Network;
    use tari_common_types::seeds::cipher_seed::CipherSeed;
    use tari_transaction_components::key_manager::wallet_types::SeedWordsWallet;

    fn derivations() -> usize {
        KEY_DERIVATIONS.with(|count| count.get())
    }

    fn wallet_parses() -> usize {
        WALLET_PARSES.with(|count| count.get())
    }

    fn new_account(conn: &Connection, name: &str) -> AccountRow {
        let wallet = WalletType::SeedWords(SeedWordsWallet::construct_new(CipherSeed::random()).unwrap());
        create_account(conn, name, &wallet, "password").unwrap();
        get_account_by_name(conn, name).unwrap().unwrap()
    }

    #[test]
    fn unlock_runs_argon2_once_for_many_operations() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_db(dir.path().join("session.db")).unwrap();
        let conn = pool.get().unwrap();
        let account = new_account(&conn, "session-once");

        let before = derivations();
        unlock(&conn, "session-once", "password").unwrap();
        unlock(&conn, "session-once", "password").unwrap();
        // Parsing a seed wallet decodes its cipher seed, which runs Argon2 too, so session hits
        // must not parse at all.
        let parses_before = wallet_parses();
        for _ in 0..5 {
            let address = account.get_address(Network::LocalNet, "").unwrap();
            account
                .get_address_with_payment_id(Network::LocalNet, "", b"id")
                .unwrap();
            account.get_key_manager("").unwrap();
            let signature = sign_message_unlocked(&account, "hello").unwrap();
            assert!(verify_message(&address, "hello", &signature).unwrap());
            sign_message(&account, "", "hello").unwrap();
        }
        assert_eq!(derivations() - before, 1);
        assert_eq!(
            wallet_parses(),
            parses_before,
            "session hits must not re-parse the wallet"
        );
        lock("session-once");
    }

    #[test]
    fn a_wrong_passphrase_stores_no_session() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_db(dir.path().join("session.db")).unwrap();
        let conn = pool.get().unwrap();
        new_account(&conn, "session-wrong");

        let err = unlock(&conn, "session-wrong", "wrong").expect_err("wrong passphrase");
        assert!(matches!(err, WalletDbError::InvalidPassphrase), "got: {err}");
        assert!(!is_unlocked(&conn, "session-wrong").unwrap());
    }

    #[test]
    fn signing_after_lock_reports_not_unlocked() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_db(dir.path().join("session.db")).unwrap();
        let conn = pool.get().unwrap();
        let account = new_account(&conn, "session-lock");

        unlock(&conn, "session-lock", "password").unwrap();
        sign_message_unlocked(&account, "hello").unwrap();
        lock("session-lock");
        lock("session-lock");

        assert!(!is_unlocked(&conn, "session-lock").unwrap());
        assert!(matches!(
            sign_message_unlocked(&account, "hello"),
            Err(SigningError::Db(WalletDbError::NotUnlocked(_)))
        ));
        assert!(matches!(
            account.get_address(Network::LocalNet, ""),
            Err(WalletDbError::DecryptionFailed(_))
        ));
    }

    #[test]
    fn change_passphrase_keeps_an_unlocked_account_unlocked_under_the_new_one() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_db(dir.path().join("session.db")).unwrap();
        let mut conn = pool.get().unwrap();
        new_account(&conn, "session-change");
        unlock(&conn, "session-change", "password").unwrap();

        let err = change_passphrase(&mut conn, "session-change", "wrong", "new").expect_err("wrong old passphrase");
        assert!(matches!(err, WalletDbError::DecryptionFailed(_)), "got: {err}");
        assert!(
            is_unlocked(&conn, "session-change").unwrap(),
            "a failed change keeps the session"
        );

        let before = derivations();
        change_passphrase(&mut conn, "session-change", "password", "new").unwrap();
        // One derivation checks the old passphrase, one encrypts under the new; none re-unlocks.
        assert_eq!(derivations() - before, 2);
        assert!(is_unlocked(&conn, "session-change").unwrap());
        let account = get_account_by_name(&conn, "session-change").unwrap().unwrap();
        let parses_before = wallet_parses();
        sign_message_unlocked(&account, "hello").unwrap();
        assert_eq!(
            wallet_parses(),
            parses_before,
            "the new session holds the parsed wallet"
        );

        lock("session-change");
        assert!(matches!(
            unlock(&conn, "session-change", "password"),
            Err(WalletDbError::InvalidPassphrase)
        ));
        unlock(&conn, "session-change", "new").unwrap();
        lock("session-change");
    }

    #[test]
    fn passphrase_gates_still_check_the_passphrase_while_unlocked() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_db(dir.path().join("session.db")).unwrap();
        let conn = pool.get().unwrap();
        let account = new_account(&conn, "session-gates");
        unlock(&conn, "session-gates", "password").unwrap();

        assert!(matches!(
            account.get_seed_words("wrong"),
            Err(WalletDbError::DecryptionFailed(_))
        ));
        assert!(matches!(
            account.get_keys_hex("wrong"),
            Err(WalletDbError::DecryptionFailed(_))
        ));
        assert!(matches!(
            account.decrypt_wallet_type_with_passphrase("wrong"),
            Err(WalletDbError::DecryptionFailed(_))
        ));
        assert!(matches!(
            account.verify_passphrase("wrong"),
            Err(WalletDbError::DecryptionFailed(_))
        ));
        account.verify_passphrase("password").unwrap();
        assert!(account.get_seed_words("password").unwrap().is_some());
        account.get_keys_hex("password").unwrap();
        lock("session-gates");
    }

    #[test]
    fn two_unlocked_accounts_sign_independently() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_db(dir.path().join("session.db")).unwrap();
        let conn = pool.get().unwrap();
        let first = new_account(&conn, "session-first");
        let second = new_account(&conn, "session-second");
        unlock(&conn, "session-first", "password").unwrap();
        unlock(&conn, "session-second", "password").unwrap();

        let first_address = first.get_address(Network::LocalNet, "").unwrap();
        let second_address = second.get_address(Network::LocalNet, "").unwrap();
        let first_signature = sign_message_unlocked(&first, "hello").unwrap();
        let second_signature = sign_message_unlocked(&second, "hello").unwrap();
        assert!(verify_message(&first_address, "hello", &first_signature).unwrap());
        assert!(verify_message(&second_address, "hello", &second_signature).unwrap());
        assert!(!verify_message(&second_address, "hello", &first_signature).unwrap());

        lock("session-first");
        assert!(sign_message_unlocked(&first, "hello").is_err());
        sign_message_unlocked(&second, "hello").unwrap();
        lock("session-second");
    }

    #[test]
    fn delete_and_rename_lock_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_db(dir.path().join("session.db")).unwrap();
        let mut conn = pool.get().unwrap();
        new_account(&conn, "session-delete");
        new_account(&conn, "session-rename");
        unlock(&conn, "session-delete", "password").unwrap();
        unlock(&conn, "session-rename", "password").unwrap();

        delete_wallet_with_conn(&mut conn, "session-delete").unwrap();
        rename_wallet_with_conn(&mut conn, "session-rename", "session-renamed").unwrap();
        // The rows are gone or renamed, so check the session map itself.
        let sessions = SESSIONS.read().unwrap();
        assert!(!sessions.contains_key("session-delete"));
        assert!(!sessions.contains_key("session-rename"));
        assert!(!sessions.contains_key("session-renamed"));
    }

    #[test]
    fn changing_a_passphrase_leaves_a_same_named_account_in_another_database_alone() {
        let dir = tempfile::tempdir().unwrap();
        let first_pool = init_db(dir.path().join("first.db")).unwrap();
        let second_pool = init_db(dir.path().join("second.db")).unwrap();
        let first_conn = first_pool.get().unwrap();
        let mut second_conn = second_pool.get().unwrap();
        new_account(&first_conn, "session-change-shared");
        new_account(&second_conn, "session-change-shared");
        unlock(&first_conn, "session-change-shared", "password").unwrap();

        change_passphrase(&mut second_conn, "session-change-shared", "password", "new").unwrap();
        assert!(is_unlocked(&first_conn, "session-change-shared").unwrap());
        assert!(!is_unlocked(&second_conn, "session-change-shared").unwrap());
        lock("session-change-shared");
    }

    #[test]
    fn deleting_a_same_named_account_in_another_database_keeps_this_session() {
        let dir = tempfile::tempdir().unwrap();
        let second_path = dir.path().join("second.db");
        let first_pool = init_db(dir.path().join("first.db")).unwrap();
        let second_pool = init_db(second_path.clone()).unwrap();
        let first_conn = first_pool.get().unwrap();
        new_account(&first_conn, "session-delete-shared");
        new_account(&second_pool.get().unwrap(), "session-delete-shared");
        unlock(&first_conn, "session-delete-shared", "password").unwrap();

        delete_wallet(&second_path, "session-delete-shared").unwrap();
        assert!(is_unlocked(&first_conn, "session-delete-shared").unwrap());
        lock("session-delete-shared");
    }

    #[test]
    fn a_session_does_not_apply_to_another_database() {
        let dir = tempfile::tempdir().unwrap();
        let first_pool = init_db(dir.path().join("first.db")).unwrap();
        let second_pool = init_db(dir.path().join("second.db")).unwrap();
        let first_conn = first_pool.get().unwrap();
        let second_conn = second_pool.get().unwrap();
        new_account(&first_conn, "session-shared-name");
        let other = new_account(&second_conn, "session-shared-name");

        unlock(&first_conn, "session-shared-name", "password").unwrap();
        assert!(is_unlocked(&first_conn, "session-shared-name").unwrap());
        assert!(!is_unlocked(&second_conn, "session-shared-name").unwrap());
        assert!(matches!(
            other.unlocked_wallet_type(),
            Err(WalletDbError::NotUnlocked(_))
        ));
        lock("session-shared-name");
    }

    #[test]
    fn this_module_only_logs_account_names_to_audit() {
        // Key material must never reach a log line. The only log lines allowed here are
        // one-line audit messages whose only field is the account name.
        let source = include_str!("session.rs");
        let audit = format!("{}!(target: \"audit\"", "info");
        let allowed_fields = [audit.clone(), format!("{audit}, account = account_name")];
        for line in source.lines().map(str::trim) {
            if line.contains(&format!("{}::", "log")) {
                assert_eq!(line, format!("use {}::info;", "log"), "unexpected log import");
            }
            for name in ["trace", "debug", "info", "warn", "error"] {
                if !line.contains(&format!("{name}!(")) {
                    continue;
                }
                let (fields, message) = line
                    .split_once("; \"")
                    .or_else(|| line.split_once(", \""))
                    .expect("audit line has a message");
                assert!(
                    allowed_fields.contains(&fields.to_string()),
                    "log line not allowed: {line}"
                );
                assert!(message.ends_with("\");"), "log line must be one plain message: {line}");
                assert!(!message.contains('{'), "log message must not format values: {line}");
            }
        }
    }
}
