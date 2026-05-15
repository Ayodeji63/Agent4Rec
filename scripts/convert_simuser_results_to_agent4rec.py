import argparse
import json
import re
from pathlib import Path

import pandas as pd


FIELD_PATTERN = re.compile(r"^([A-Z ]+):\s*(.*)$")


def parse_persona(text: str) -> dict:
    fields = {}
    current_key = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^CANDIDATE\s+\d+", line, flags=re.I):
            fields["CANDIDATE"] = line
            current_key = "CANDIDATE"
            continue
        match = FIELD_PATTERN.match(line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            fields[key] = value
            current_key = key
        elif current_key:
            fields[current_key] = (fields.get(current_key, "") + " " + line).strip()
    return fields


def split_items(value: str) -> list[str]:
    if not value:
        return []
    value = re.sub(r"\([^)]*\)", "", value)
    parts = re.split(r";|,", value)
    cleaned = []
    for part in parts:
        item = part.strip().strip('"').strip("'")
        if item and item.lower() not in {"none", "none mentioned", "n/a"}:
            cleaned.append(item)
    return list(dict.fromkeys(cleaned))


def sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def compact_join(items: list[str], limit: int = 8) -> str:
    return "; ".join(items[:limit])


def infer_price_sensitivity(fields: dict) -> str:
    text = " ".join(
        str(fields.get(key, ""))
        for key in ["PREFERENCE SUMMARY", "LIKES", "DISLIKES", "RATING TENDENCY"]
    ).lower()
    high_markers = [
        "budget",
        "value",
        "cheap",
        "affordable",
        "fair price",
        "fair prices",
        "overpriced",
        "pricey",
        "expensive",
        "money's worth",
    ]
    low_markers = ["higher-end", "fine dining", "across all price tiers", "willing to spend"]
    if any(marker in text for marker in high_markers):
        return "high"
    if any(marker in text for marker in low_markers):
        return "low"
    return "medium"


def infer_review_tone(fields: dict) -> str:
    tendency = str(fields.get("RATING TENDENCY", "")).strip()
    if tendency:
        return tendency
    text = " ".join(str(v) for v in fields.values()).lower()
    if any(word in text for word in ["harsh", "scathing", "punish", "1-2"]):
        return "polarized and critical when disappointed"
    if any(word in text for word in ["generous", "enthusiastic", "loyal"]):
        return "warm and generous toward favorites"
    return "balanced and evidence-driven"


def build_knowledge_edges(fields: dict, meta: dict) -> list[tuple[str, str, str]]:
    likes = split_items(fields.get("LIKES", ""))
    dislikes = split_items(fields.get("DISLIKES", ""))
    evidence = split_items(fields.get("EVIDENCE RESTAURANTS", ""))
    rating_pattern = fields.get("RATING TENDENCY", "")
    price_sensitivity = infer_price_sensitivity(fields)
    ethnic_group = meta.get("ethnic_group") or "nigerian"
    edges = [
        ("user", "ethnic_group", str(ethnic_group)),
        ("user", "price_sensitivity", str(price_sensitivity)),
    ]
    edges.extend(("user", "likes", item) for item in likes[:6])
    edges.extend(("user", "dislikes", item) for item in dislikes[:4])
    if rating_pattern:
        edges.append(("user", "rating_pattern", str(rating_pattern)))
    edges.extend(("user", "evidence_restaurant", item) for item in evidence[:6])
    return edges


def build_knowledge_graph(fields: dict, meta: dict) -> str:
    return " | ".join(f"{subj} -> {rel} -> {obj}" for subj, rel, obj in build_knowledge_edges(fields, meta))


def build_taste(fields: dict) -> str:
    likes = split_items(fields.get("LIKES", ""))
    dislikes = split_items(fields.get("DISLIKES", ""))
    summary = fields.get("PREFERENCE SUMMARY", "")
    taste_lines = []

    for like in likes[:6]:
        taste_lines.append(f"I enjoy restaurants or cuisines related to {like}.")

    if summary:
        taste_lines.append(f"I tend to prefer {summary[0].lower() + summary[1:] if summary else summary}")

    for dislike in dislikes[:3]:
        taste_lines.append(f"I avoid restaurants or dining experiences related to {dislike}.")

    if not taste_lines:
        taste_lines.append("I prefer restaurants that match my historical dining tastes and rating patterns.")

    return "| ".join(sentence(line) for line in taste_lines)


def build_high_rating(fields: dict, meta: dict) -> str:
    tendency = fields.get("RATING TENDENCY", "")
    pickiness = fields.get("PICKINESS") or meta.get("pickiness", "")
    likes = fields.get("LIKES", "")
    pieces = []
    if tendency:
        pieces.append(sentence(tendency))
    if pickiness:
        pieces.append(f"You are {pickiness} when evaluating restaurants.")
    if likes:
        pieces.append(f"You usually rate restaurants higher when they match these tastes: {likes}.")
    return " ".join(pieces) or "You give high ratings when restaurants align with your preferences."


def build_low_rating(fields: dict) -> str:
    dislikes = fields.get("DISLIKES", "")
    if dislikes and dislikes.lower() not in {"none", "none mentioned", "n/a"}:
        return f"You give low ratings when restaurants show these disliked patterns: {dislikes}."
    return "You give low ratings when restaurants fail to match your preferences, value expectations, or service standards."


def tier_value(label: str, *, reverse: bool = False) -> int:
    mapping = {"low": 1, "medium": 2, "high": 3}
    reverse_mapping = {"low": 3, "medium": 2, "high": 1}
    label = str(label or "medium").lower()
    return (reverse_mapping if reverse else mapping).get(label, 2)


def convert(input_dir: Path, output_dir: Path) -> None:
    users_path = input_dir / "users.json"
    if not users_path.exists():
        raise FileNotFoundError(f"Missing {users_path}")

    users = json.loads(users_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    persona_rows = []
    statistic_rows = []
    manifest_rows = []
    kg_rows = []

    for key in sorted(users, key=lambda x: int(x)):
        meta = users[key]
        persona_path = input_dir / f"persona_{key}.txt"
        if not persona_path.exists():
            continue

        fields = parse_persona(persona_path.read_text(encoding="utf-8", errors="ignore"))
        avatar_id = int(key)
        for subject, relation, obj in build_knowledge_edges(fields, meta):
            kg_rows.append(
                {
                    "avatar_id": avatar_id,
                    "subject": f"user_{avatar_id}" if subject == "user" else subject,
                    "relation": relation,
                    "object": obj,
                }
            )
        persona_rows.append(
            {
                "taste": build_taste(fields),
                "reason": sentence(fields.get("PREFERENCE SUMMARY", "")),
                "high_rating": build_high_rating(fields, meta),
                "low_rating": build_low_rating(fields),
                "liked_items": compact_join(split_items(fields.get("LIKES", ""))),
                "disliked_items": compact_join(split_items(fields.get("DISLIKES", ""))),
                "past_rating_patterns": sentence(fields.get("RATING TENDENCY", "")),
                "past_review_tone": infer_review_tone(fields),
                "frequent_categories": compact_join(split_items(fields.get("LIKES", "")), limit=10),
                "price_sensitivity": infer_price_sensitivity(fields),
                "evidence_items": compact_join(split_items(fields.get("EVIDENCE RESTAURANTS", "")), limit=10),
                "knowledge_graph": build_knowledge_graph(fields, meta),
            }
        )
        statistic_rows.append(
            {
                "user_id": avatar_id,
                "activity": tier_value(meta.get("engagement")),
                "diversity": tier_value(meta.get("variety")),
                # Agent4Rec conformity=1 is follower/high-conformity,
                # conformity=3 is maverick/low-conformity.
                "conformity": tier_value(meta.get("conformity"), reverse=True),
            }
        )
        manifest_rows.append(
            {
                "avatar_id": avatar_id,
                "yelp_user_id": meta.get("yelp_user_id"),
                "ethnic_group": meta.get("ethnic_group"),
                "selected_candidate_id": meta.get("selected_candidate_id"),
                "selected_candidate_score": meta.get("selected_candidate_score"),
                "n_reviews": meta.get("n_reviews"),
                "avg_rating": meta.get("avg_rating"),
                "pickiness": meta.get("pickiness"),
                "engagement": meta.get("engagement"),
                "conformity": meta.get("conformity"),
                "variety": meta.get("variety"),
            }
        )

    persona_df = pd.DataFrame(persona_rows)
    statistic_df = pd.DataFrame(statistic_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    kg_df = pd.DataFrame(kg_rows)

    persona_df.to_csv(output_dir / "all_personas_like_modify.csv", index=False)
    statistic_df.to_csv(output_dir / "user_statistic.csv", index=False)
    manifest_df.to_csv(output_dir / "avatar_manifest.csv", index=False)
    kg_df.to_csv(output_dir / "knowledge_graph_triples.csv", index=False)

    print(f"Converted {len(persona_df)} personas")
    print(f"Wrote {output_dir / 'all_personas_like_modify.csv'}")
    print(f"Wrote {output_dir / 'user_statistic.csv'}")
    print(f"Wrote {output_dir / 'avatar_manifest.csv'}")
    print(f"Wrote {output_dir / 'knowledge_graph_triples.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("results/simulation_simusers_kimi"),
        help="Directory containing persona_*.txt and users.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/agent4rec_kimi_adapter"),
        help="Directory for Agent4Rec-compatible CSVs",
    )
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
