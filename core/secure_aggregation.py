from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch


SECURE_AGG_SIM_DISCLAIMER = (
    "secure_agg_sim is a deterministic masking simulation for academic demos. "
    "It is not cryptographically secure and does not replace real secure aggregation."
)

HE_DEMO_DISCLAIMER = (
    "he_demo only demonstrates additive homomorphic aggregation on a tiny vector. "
    "Full neural-network weight encryption is not implemented."
)


@dataclass(frozen=True)
class SecureAggregationMetadata:
    security_mode: str
    client_id: str
    cohort_ids: list[str]
    round_number: int
    sample_count: int
    mask_std: float
    disclaimer: str = SECURE_AGG_SIM_DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def mask_weighted_state_dict_for_upload(
    state_dict: Mapping[str, torch.Tensor],
    client_id: str,
    cohort_ids: Sequence[str],
    round_number: int,
    sample_count: int,
    mask_std: float = 0.01,
) -> tuple[dict[str, torch.Tensor], SecureAggregationMetadata]:
    """Apply deterministic pairwise masks to a PyTorch state_dict.

    The mask is divided by the local sample count so the sample-weighted server
    sum cancels pairwise masks. This is a simulation; there is no key exchange.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive for secure aggregation masking.")

    cohort = _normalized_cohort(cohort_ids, client_id)
    masked: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        value = tensor.detach().cpu().clone()
        if torch.is_tensor(value) and torch.is_floating_point(value) and mask_std > 0:
            mask = _client_mask_sum(
                key=key,
                reference_tensor=value,
                client_id=client_id,
                cohort_ids=cohort,
                round_number=round_number,
                mask_std=mask_std,
            )
            masked[key] = (value.float() + mask / float(sample_count)).to(value.dtype)
        else:
            masked[key] = value

    metadata = SecureAggregationMetadata(
        security_mode="secure_agg_sim",
        client_id=str(client_id),
        cohort_ids=cohort,
        round_number=int(round_number),
        sample_count=int(sample_count),
        mask_std=float(mask_std),
    )
    return masked, metadata


def aggregate_masked_weighted_state_dicts(
    masked_updates: Mapping[str, Mapping[str, torch.Tensor]],
    sample_counts: Mapping[str, int],
    cohort_ids: Sequence[str],
    round_number: int,
    mask_std: float = 0.01,
    client_cohorts: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Aggregate masked updates with sample-weighted FedAvg.

    The aggregate logic removes any residual deterministic masks. This makes the
    dropout behavior usable for a prototype, but it is also why the feature is
    labelled as a simulation instead of real secure aggregation.
    """
    if not masked_updates:
        raise ValueError("Cannot aggregate zero masked updates.")

    completed_clients = sorted(str(client_id) for client_id in masked_updates)
    counts = {str(k): int(v) for k, v in sample_counts.items()}
    total_samples = sum(counts.get(client_id, 0) for client_id in completed_clients)
    if total_samples <= 0:
        raise ValueError("Cannot aggregate masked updates with zero total samples.")

    first_id = completed_clients[0]
    expected_keys = set(masked_updates[first_id].keys())
    for client_id in completed_clients:
        update_keys = set(masked_updates[client_id].keys())
        if update_keys != expected_keys:
            raise ValueError(f"Masked update keys for {client_id} do not match the first client.")

    aggregate: dict[str, torch.Tensor] = {}
    residual_masks: dict[str, torch.Tensor] = {}

    for key in masked_updates[first_id].keys():
        first_tensor = masked_updates[first_id][key].detach().cpu()
        if torch.is_floating_point(first_tensor):
            value = torch.zeros_like(first_tensor.float())
            residual = torch.zeros_like(first_tensor.float())
            for client_id in completed_clients:
                tensor = masked_updates[client_id][key].detach().cpu()
                if tuple(tensor.shape) != tuple(first_tensor.shape):
                    raise ValueError(f"Shape mismatch for parameter {key} from client {client_id}.")
                weight = counts[client_id] / total_samples
                value = value + tensor.float() * weight

                cohort = (
                    client_cohorts.get(client_id)
                    if client_cohorts and client_id in client_cohorts
                    else cohort_ids
                )
                client_mask = _client_mask_sum(
                    key=key,
                    reference_tensor=first_tensor,
                    client_id=client_id,
                    cohort_ids=_normalized_cohort(cohort, client_id),
                    round_number=round_number,
                    mask_std=mask_std,
                )
                residual = residual + client_mask / float(total_samples)

            aggregate[key] = (value - residual).to(first_tensor.dtype)
            residual_masks[key] = residual
        else:
            aggregate[key] = first_tensor.clone()

    metadata = {
        "security_mode": "secure_agg_sim",
        "round_number": int(round_number),
        "completed_clients": completed_clients,
        "total_samples": int(total_samples),
        "mask_std": float(mask_std),
        "cohort_ids": _normalized_cohort(cohort_ids, completed_clients[0]),
        "dropout_residual_unmasked": True,
        "disclaimer": SECURE_AGG_SIM_DISCLAIMER,
    }
    return aggregate, metadata


def communication_size_bytes(state_dict: Mapping[str, torch.Tensor]) -> int:
    """Estimate tensor payload size for a state_dict."""
    total = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total += int(value.numel() * value.element_size())
    return total


def run_paillier_he_demo(values: Sequence[float]) -> dict:
    """Run a tiny additive homomorphic encryption demonstration if phe exists."""
    plain_values = [float(v) for v in values]
    try:
        from phe import paillier
    except Exception:
        return {
            "security_mode": "he_demo",
            "available": False,
            "input_values": plain_values,
            "note": "Install the optional 'phe' package to run this toy Paillier demo.",
            "disclaimer": HE_DEMO_DISCLAIMER,
        }

    public_key, private_key = paillier.generate_paillier_keypair(n_length=512)
    encrypted_values = [public_key.encrypt(value) for value in plain_values]
    encrypted_sum = encrypted_values[0] if encrypted_values else public_key.encrypt(0.0)
    for value in encrypted_values[1:]:
        encrypted_sum = encrypted_sum + value

    return {
        "security_mode": "he_demo",
        "available": True,
        "library": "phe",
        "input_values": plain_values,
        "decrypted_sum": float(private_key.decrypt(encrypted_sum)),
        "note": "Toy vector only; DenseNet model weights are not encrypted.",
        "disclaimer": HE_DEMO_DISCLAIMER,
    }


def _client_mask_sum(
    key: str,
    reference_tensor: torch.Tensor,
    client_id: str,
    cohort_ids: Sequence[str],
    round_number: int,
    mask_std: float,
) -> torch.Tensor:
    mask_sum = torch.zeros_like(reference_tensor.detach().cpu().float())
    if mask_std <= 0:
        return mask_sum

    client_id = str(client_id)
    for other_id in cohort_ids:
        other_id = str(other_id)
        if other_id == client_id:
            continue
        left, right = sorted([client_id, other_id])
        sign = 1.0 if client_id == left else -1.0
        mask_sum = mask_sum + sign * _pairwise_mask(
            key=key,
            reference_tensor=reference_tensor,
            left_id=left,
            right_id=right,
            round_number=round_number,
            mask_std=mask_std,
        )
    return mask_sum


def _pairwise_mask(
    key: str,
    reference_tensor: torch.Tensor,
    left_id: str,
    right_id: str,
    round_number: int,
    mask_std: float,
) -> torch.Tensor:
    seed = _stable_seed(f"{round_number}|{left_id}|{right_id}|{key}|{tuple(reference_tensor.shape)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        tuple(reference_tensor.shape),
        generator=generator,
        dtype=torch.float32,
    ) * float(mask_std)


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


def _normalized_cohort(cohort_ids: Sequence[str] | None, client_id: str) -> list[str]:
    values = [str(item) for item in (cohort_ids or []) if str(item)]
    if str(client_id) not in values:
        values.append(str(client_id))
    return sorted(set(values))
