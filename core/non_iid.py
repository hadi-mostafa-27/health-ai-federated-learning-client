from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FederatedSplitConfig:
    strategy: str = "balanced_iid"
    num_hospitals: int = 3
    seed: int = 42
    imbalance_severity: float = 0.5
    hospital_ids: Sequence[str] | None = field(default=None)


def split_federated_rows(rows: Sequence[Mapping], config: FederatedSplitConfig) -> dict[str, list[dict]]:
    """Split image rows into simulated hospital datasets.

    Supported strategies:
    - balanced_iid: class-balanced round-robin assignment where possible.
    - label_skew: hospitals receive different class proportions.
    - quantity_skew: hospitals receive different dataset sizes.
    """
    hospital_ids = _hospital_ids(config)
    output = {hospital_id: [] for hospital_id in hospital_ids}
    rows_list = [dict(row) for row in rows]
    if not rows_list:
        return output

    strategy = (config.strategy or "balanced_iid").lower()
    rng = random.Random(config.seed)
    severity = max(0.0, min(0.99, float(config.imbalance_severity)))

    if strategy in {"balanced_iid", "iid", "balanced"}:
        return _balanced_iid(rows_list, hospital_ids, rng)
    if strategy in {"label_skew", "label-skew", "non_iid_label"}:
        return _label_skew(rows_list, hospital_ids, rng, severity)
    if strategy in {"quantity_skew", "quantity-skew", "non_iid_quantity"}:
        return _quantity_skew(rows_list, hospital_ids, rng, severity)

    raise ValueError(f"Unknown federated split strategy: {config.strategy}")


def summarize_federated_split(split: Mapping[str, Sequence[Mapping]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for hospital_id, rows in split.items():
        normal = 0
        pneumonia = 0
        other = 0
        for row in rows:
            label = _label_key(row)
            if label == "NORMAL":
                normal += 1
            elif label == "PNEUMONIA":
                pneumonia += 1
            else:
                other += 1
        total = normal + pneumonia + other
        minority = min(normal, pneumonia) if normal and pneumonia else 0
        majority = max(normal, pneumonia)
        imbalance_ratio = float(majority / minority) if minority else (float(majority) if majority else 0.0)
        summary[hospital_id] = {
            "total": total,
            "normal": normal,
            "pneumonia": pneumonia,
            "other": other,
            "imbalance_ratio": imbalance_ratio,
        }
    return summary


def _balanced_iid(rows: list[dict], hospital_ids: list[str], rng: random.Random) -> dict[str, list[dict]]:
    output = {hospital_id: [] for hospital_id in hospital_ids}
    by_label = _group_by_label(rows)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        for idx, row in enumerate(label_rows):
            output[hospital_ids[idx % len(hospital_ids)]].append(row)
    for hospital_rows in output.values():
        rng.shuffle(hospital_rows)
    return output


def _label_skew(
    rows: list[dict],
    hospital_ids: list[str],
    rng: random.Random,
    severity: float,
) -> dict[str, list[dict]]:
    output = {hospital_id: [] for hospital_id in hospital_ids}
    if len(hospital_ids) == 1:
        output[hospital_ids[0]] = list(rows)
        return output

    midpoint = max(1, len(hospital_ids) // 2)
    for row in rows:
        label = _label_key(row)
        if label == "PNEUMONIA":
            preferred = set(hospital_ids[midpoint:])
        elif label == "NORMAL":
            preferred = set(hospital_ids[:midpoint])
        else:
            preferred = set(hospital_ids)

        weights = []
        for hospital_id in hospital_ids:
            if hospital_id in preferred:
                weights.append(1.0 + severity * len(hospital_ids))
            else:
                weights.append(max(0.01, 1.0 - severity))
        chosen = rng.choices(hospital_ids, weights=weights, k=1)[0]
        output[chosen].append(row)

    _rebalance_empty_clients(output, rng)
    return output


def _quantity_skew(
    rows: list[dict],
    hospital_ids: list[str],
    rng: random.Random,
    severity: float,
) -> dict[str, list[dict]]:
    output = {hospital_id: [] for hospital_id in hospital_ids}
    shuffled = list(rows)
    rng.shuffle(shuffled)

    if len(hospital_ids) == 1:
        output[hospital_ids[0]] = shuffled
        return output

    order = list(hospital_ids)
    rng.shuffle(order)
    power = 1.0 + severity * 5.0
    weights = [(idx + 1) ** power for idx in range(len(order))]
    total_weight = sum(weights)

    raw_counts = [int(round(len(shuffled) * weight / total_weight)) for weight in weights]
    while sum(raw_counts) < len(shuffled):
        raw_counts[raw_counts.index(min(raw_counts))] += 1
    while sum(raw_counts) > len(shuffled):
        idx = raw_counts.index(max(raw_counts))
        raw_counts[idx] -= 1

    cursor = 0
    for hospital_id, count in zip(order, raw_counts):
        output[hospital_id] = shuffled[cursor: cursor + count]
        cursor += count

    _rebalance_empty_clients(output, rng)
    return output


def _rebalance_empty_clients(split: dict[str, list[dict]], rng: random.Random) -> None:
    empty = [hospital_id for hospital_id, rows in split.items() if not rows]
    for hospital_id in empty:
        donors = [key for key, rows in split.items() if len(rows) > 1]
        if not donors:
            return
        donor = rng.choice(donors)
        split[hospital_id].append(split[donor].pop())


def _hospital_ids(config: FederatedSplitConfig) -> list[str]:
    if config.hospital_ids:
        hospital_ids = [str(hid) for hid in config.hospital_ids if str(hid)]
        if hospital_ids:
            return hospital_ids
    count = max(1, int(config.num_hospitals))
    return [f"hospital_{idx + 1}" for idx in range(count)]


def _group_by_label(rows: Sequence[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_label_key(row), []).append(row)
    return grouped


def _label_key(row: Mapping) -> str:
    return str(row.get("label", row.get("class", "UNKNOWN"))).upper()
