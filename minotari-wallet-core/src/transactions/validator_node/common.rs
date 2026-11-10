//! Shared helpers for validator node pay-to-self transaction construction.
//!
//! Both VN operations (registration and exit) lock the consensus-required
//! minimum deposit and build a single pay-to-self output with operation-specific
//! [`OutputFeatures`]. This module extracts that common pattern.

use anyhow::anyhow;
use log::info;
use rusqlite::Connection;
use tari_common::configuration::Network;
use tari_common_types::{transaction::TxId, types::FixedHash};
use tari_script::script;
use tari_transaction_components::{
    TransactionBuilder,
    consensus::ConsensusConstantsBuilder,
    offline_signing::{
        models::{PaymentRecipient, PrepareOneSidedTransactionForSigningResult},
        prepare_one_sided_transaction_for_signing,
    },
    tari_amount::MicroMinotari,
    transaction_components::{MemoField, OutputFeatures, memo_field::TxType},
};

use crate::{
    db::AccountRow,
    transactions::{
        fee_estimator::{estimated_output_size_for_send, measure_output_size},
        fund_locker::{FundLocker, release_reservation},
        idempotency::{IdempotencyBinding, IdempotencyOperation, RequestFingerprint},
        input_selector::map_change_required_error,
    },
};

/// Locks the VN registration deposit and prepares a pay-to-self transaction for signing.
///
/// Used by both VN operations (registration and exit) which share the same
/// transaction shape: one output sent back to the sender, carrying operation-specific
/// `output_features`.
///
/// `operation` both names the transaction in the audit log and scopes the
/// idempotency key, so a registration key cannot be redeemed for an exit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_vn_pay_to_self_tx(
    account: &AccountRow,
    conn: &mut Connection,
    network: Network,
    password: &str,
    output_features: OutputFeatures,
    fee_per_gram: MicroMinotari,
    payment_id: Option<&str>,
    idempotency_key: Option<String>,
    seconds_to_lock: u64,
    confirmation_window: u64,
    operation: IdempotencyOperation,
) -> Result<PrepareOneSidedTransactionForSigningResult, anyhow::Error> {
    let consensus_constants = ConsensusConstantsBuilder::new(network).build();
    let deposit_amount = consensus_constants.validator_node_registration_min_deposit_amount();

    info!(
        target: "audit",
        deposit_amount = deposit_amount.as_u64();
        "Creating {} transaction", operation.description()
    );

    // `output_features` carries everything that distinguishes one VN operation
    // from another — the node public key, its signature, the claim key, the max
    // epoch — so hashing its canonical encoding binds the
    // key to the specific node action being requested.
    let encoded_features = serde_json::to_vec(&output_features)
        .map_err(|e| anyhow!("Failed to encode validator node output features: {}", e))?;
    let idempotency = IdempotencyBinding::new(
        idempotency_key,
        operation,
        RequestFingerprint::new(operation)
            .field("account_id", account.id.to_le_bytes())
            .field("deposit_amount", deposit_amount.as_u64().to_le_bytes())
            .field("output_features", encoded_features)
            .field("fee_per_gram", fee_per_gram.as_u64().to_le_bytes())
            .optional_field("payment_id", payment_id)
            .field("seconds_to_lock", seconds_to_lock.to_le_bytes())
            .field("confirmation_window", confirmation_window.to_le_bytes()),
    );

    let sender_address = account.get_address(network, password)?;
    let memo = match payment_id {
        Some(s) => MemoField::new_open_from_string(s, TxType::PaymentToOther)
            .map_err(|e| anyhow!("Invalid payment_id for validator node transaction: {e}"))?,
        None => MemoField::default(),
    };
    MemoField::new_transaction_info(
        sender_address.clone(),
        MicroMinotari::zero(),
        MicroMinotari::zero(),
        true,
        TxType::PaymentToOther,
        vec![FixedHash::zero()],
        memo.payment_id_as_bytes(),
    )
    .map_err(|e| anyhow!("Payment id does not fit in the validator node change memo: {e}"))?;

    let script = script!(PushPubKey(Box::default())).map_err(|e| anyhow!("Failed to build the default script: {e}"))?;
    let vn_output_size = measure_output_size(&output_features, &script, &memo)?;
    let change_output_size = estimated_output_size_for_send(memo.payment_id_as_bytes().len(), 1, &sender_address)?;
    let estimated_output_size = vn_output_size.max(change_output_size);

    let fund_locker = FundLocker::new();
    let locked_funds = fund_locker.lock(
        conn,
        account.id,
        deposit_amount,
        1,
        fee_per_gram,
        Some(estimated_output_size),
        idempotency,
        seconds_to_lock,
        confirmation_window,
    )?;

    let result = (|| -> Result<PrepareOneSidedTransactionForSigningResult, anyhow::Error> {
        let key_manager = account.get_key_manager(password)?;
        let mut tx_builder = TransactionBuilder::new(consensus_constants, key_manager.clone(), network)?;
        tx_builder.with_fee_per_gram(fee_per_gram);
        // Every send carries a change output, so the wallet's scan can see it land.
        tx_builder.with_change_required();
        for utxo in &locked_funds.utxos {
            tx_builder.with_input(utxo.clone())?;
        }

        let tx_id = TxId::new_random();
        let payment_recipient = PaymentRecipient {
            amount: deposit_amount,
            output_features,
            address: sender_address.clone(),
            payment_id: memo.clone(),
        };

        prepare_one_sided_transaction_for_signing(
            &key_manager,
            tx_id,
            tx_builder,
            &[payment_recipient],
            memo,
            sender_address,
        )
        .map_err(map_change_required_error)
    })();

    if result.is_err() && locked_funds.created_reservation {
        release_reservation(conn, &locked_funds.reservation_id);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transactions::burn::tests::setup_funded_account;
    use rusqlite::named_params;
    use tari_common_types::{
        epoch::VnEpoch,
        types::{CompressedPublicKey, PrivateKey},
    };
    use tari_transaction_components::transaction_components::ValidatorNodeSignature;

    fn sample_vn_features() -> OutputFeatures {
        let vn_sk = PrivateKey::from(42u64);
        let claim_pk = CompressedPublicKey::from_secret_key(&PrivateKey::from(99u64));
        let sidechain_sk = PrivateKey::from(77u64);
        let vn_sig = ValidatorNodeSignature::sign_for_registration(
            &vn_sk,
            Network::LocalNet.as_byte(),
            Some(&CompressedPublicKey::from_secret_key(&sidechain_sk)),
            &claim_pk,
            VnEpoch(10),
        );
        OutputFeatures::for_validator_node_registration(
            vn_sig,
            claim_pk,
            Some(&sidechain_sk),
            VnEpoch(10),
        )
    }

    #[test]
    fn build_vn_pay_to_self_tx_refuses_oversized_payment_id_before_locking_funds() {
        let (pool, account, _temp) = setup_funded_account(100_000_000_000);
        let mut conn = pool.get().expect("conn");

        // 200-byte payment_id fits an open memo, but overflows the TransactionInfo change memo.
        let oversized = "x".repeat(200);
        let err = build_vn_pay_to_self_tx(
            &account,
            &mut conn,
            Network::LocalNet,
            "pass",
            sample_vn_features(),
            MicroMinotari(5),
            Some(&oversized),
            Some("vn-oversized-memo".to_string()),
            3600,
            100,
            IdempotencyOperation::ValidatorNodeRegistration,
        )
        .expect_err("oversized payment_id must be rejected before locking");
        assert!(
            err.to_string().contains("Payment id") || err.to_string().contains("payment_id"),
            "unexpected error: {err}"
        );

        let locked_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM outputs WHERE account_id = :id AND status = 'LOCKED'",
                named_params! { ":id": account.id },
                |r| r.get(0),
            )
            .expect("query locked outputs");
        assert_eq!(locked_count, 0, "no UTXOs should be locked when payment_id is invalid");
    }

    #[test]
    fn vn_registration_features_exceed_default_output_size_estimate() {
        let features = sample_vn_features();
        let script = script!(PushPubKey(Box::default())).expect("script");
        let memo = MemoField::new_open_from_string("vn-reg-memo", TxType::PaymentToOther).expect("memo");
        let vn_size = measure_output_size(&features, &script, &memo).expect("measure vn output");
        let default_size = crate::transactions::fee_estimator::get_default_features_and_scripts_size().expect("default");
        assert!(
            vn_size > default_size,
            "VN output with sidechain proof ({vn_size}) exceeds generic estimate ({default_size})"
        );
    }
}
