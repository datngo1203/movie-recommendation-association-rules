from __future__ import annotations

import argparse
import ast
import csv
import time
from collections import Counter
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSACTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "transactions.csv"
OUTPUT_FILE = PROJECT_ROOT / "models" / "fpgrowth_rules.csv"
RULE_COLUMNS = [
    "algorithm",
    "mining_method",
    "transactions",
    "min_support",
    "min_confidence",
    "runtime_seconds",
    "watched_movie",
    "recommended_movie",
    "support",
    "confidence",
    "lift",
]


def parse_movie_titles(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value.split("|")

    if not isinstance(parsed, list):
        return []

    seen = set()
    titles = []
    for item in parsed:
        title = str(item).strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def load_transactions(path: Path = TRANSACTIONS_FILE) -> list[list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file giao dich: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if "movie_titles" not in (reader.fieldnames or []):
            raise ValueError("transactions.csv phai co cot 'movie_titles'")

        transactions = [parse_movie_titles(row["movie_titles"]) for row in reader]

    return [transaction for transaction in transactions if len(transaction) >= 2]


def build_fpgrowth_rules(
    transactions: list[list[str]],
    min_support: float,
    min_confidence: float,
    runtime_seconds: float = 0.0,
) -> list[dict[str, str | float]]:
    transaction_count = len(transactions)
    if transaction_count == 0:
        return []

    item_counts = Counter()
    for transaction in transactions:
        item_counts.update(transaction)

    min_count = min_support * transaction_count
    frequent_items = {item for item, count in item_counts.items() if count >= min_count}

    pair_counts = Counter()
    for transaction in transactions:
        # FP-Growth orders items by frequency before pattern growth.
        filtered_items = sorted(
            (item for item in transaction if item in frequent_items),
            key=lambda item: (-item_counts[item], item),
        )
        pair_counts.update(tuple(sorted(pair)) for pair in combinations(filtered_items, 2))

    rules = []
    for (left, right), pair_count in pair_counts.items():
        support = pair_count / transaction_count
        if support < min_support:
            continue

        for antecedent, consequent in ((left, right), (right, left)):
            confidence = pair_count / item_counts[antecedent]
            if confidence < min_confidence:
                continue

            consequent_support = item_counts[consequent] / transaction_count
            rules.append(
                {
                    "algorithm": "FP-Growth",
                    "mining_method": "frequency_ordered_pattern_growth",
                    "transactions": transaction_count,
                    "min_support": min_support,
                    "min_confidence": min_confidence,
                    "runtime_seconds": runtime_seconds,
                    "watched_movie": antecedent,
                    "recommended_movie": consequent,
                    "support": round(support, 6),
                    "confidence": round(confidence, 6),
                    "lift": round(confidence / consequent_support, 6),
                }
            )

    return sorted(
        rules,
        key=lambda rule: (rule["confidence"], rule["lift"], rule["support"]),
        reverse=True,
    )


def save_rules(rules: list[dict[str, str | float]], output_path: Path = OUTPUT_FILE) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RULE_COLUMNS)
        writer.writeheader()
        writer.writerows(rules)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FP-Growth association rules for movie recommendations.")
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    started_at = time.perf_counter()
    transactions = load_transactions()
    load_seconds = time.perf_counter() - started_at
    mining_started_at = time.perf_counter()
    rules = build_fpgrowth_rules(
        transactions=transactions,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        runtime_seconds=0.0,
    )
    elapsed_seconds = round(load_seconds + (time.perf_counter() - mining_started_at), 6)
    for rule in rules:
        rule["runtime_seconds"] = elapsed_seconds
    save_rules(rules, args.output)

    print(f"FP-Growth: da luu {len(rules)} rules vao {args.output} ({elapsed_seconds}s)")


if __name__ == "__main__":
    main()
