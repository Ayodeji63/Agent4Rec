import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def write_cf_file(path: Path, user_items: dict[int, list[int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for user_id in sorted(user_items):
            items = " ".join(str(item_id) for item_id in user_items[user_id])
            f.write(f"{user_id} {items}\n" if items else f"{user_id}\n")


def split_items(item_ids: list[int]) -> tuple[list[int], list[int], list[int]]:
    # Keep train non-empty and reserve one validation/test item when possible.
    if len(item_ids) >= 5:
        return item_ids[:-2], [item_ids[-2]], [item_ids[-1]]
    if len(item_ids) >= 3:
        return item_ids[:-2], [item_ids[-2]], [item_ids[-1]]
    if len(item_ids) == 2:
        return [item_ids[0]], [], [item_ids[1]]
    return item_ids, [], []


def clean_categories(categories: str) -> str:
    cats = [c.strip() for c in str(categories or "").split(",") if c.strip()]
    return "|".join(cats) if cats else "Restaurant"


def restaurant_summary(row: pd.Series) -> str:
    categories = ", ".join(clean_categories(row.get("categories", "")).split("|")[:4])
    price = row.get("price", "?")
    halal = " Halal-friendly metadata is present." if bool(row.get("halal", False)) else ""
    return (
        f"{row.get('name', 'This restaurant')} is a restaurant in {row.get('city', 'an unknown city')} "
        f"with categories including {categories}. It has a historical Yelp rating of "
        f"{float(row.get('stars', 0)):.2f}, {int(row.get('review_count', 0))} reviews, "
        f"and price range {price}.{halal}"
    )


def build_agg_row(user_id: int, reviews: list[dict], business_to_item: dict[str, int]) -> dict:
    top_reviews = sorted(reviews, key=lambda r: (float(r.get("stars_review", 0)), str(r.get("date", ""))), reverse=True)[:25]
    return {
        "user_id": user_id,
        "restaurant_id_list": "; ".join(str(business_to_item[r["business_id"]]) for r in top_reviews if r["business_id"] in business_to_item),
        "restaurant_title_list": "; ".join(str(r.get("name", "")) for r in top_reviews),
        "restaurant_categories_list": "; ".join(str(r.get("categories", "")) for r in top_reviews),
        "rating_list": "; ".join(str(r.get("stars_review", "")) for r in top_reviews),
        "source": "yelp",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_history", type=Path, default=Path("results/user_review_history.json"))
    parser.add_argument("--restaurant_detail", type=Path, default=Path("results/restaurant_detail.csv"))
    parser.add_argument("--persona_users", type=Path, default=Path("results/simulation_simusers_kimi/users.json"))
    parser.add_argument("--persona_adapter", type=Path, default=Path("results/agent4rec_kimi_adapter"))
    parser.add_argument("--dataset", default="yelp-kimi")
    args = parser.parse_args()

    user_history = json.loads(args.user_history.read_text())
    persona_users = json.loads(args.persona_users.read_text())
    restaurant_detail = pd.read_csv(args.restaurant_detail, index_col="business_id")

    dataset_dir = Path("datasets") / args.dataset
    cf_dir = dataset_dir / "cf_data"
    sim_dir = dataset_dir / "simulation"
    raw_dir = dataset_dir / "raw_data"
    cf_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    avatar_to_yelp_user = {
        int(avatar_id): meta["yelp_user_id"]
        for avatar_id, meta in persona_users.items()
    }

    used_business_ids = []
    avatar_reviews = {}
    for avatar_id, yelp_user_id in sorted(avatar_to_yelp_user.items()):
        reviews = sorted(
            user_history[yelp_user_id],
            key=lambda r: str(r.get("date", "")),
        )
        avatar_reviews[avatar_id] = reviews
        used_business_ids.extend(r["business_id"] for r in reviews)

    business_ids = sorted(set(used_business_ids))
    business_to_item = {business_id: idx for idx, business_id in enumerate(business_ids)}
    item_to_business = {idx: business_id for business_id, idx in business_to_item.items()}

    train, valid, test = {}, {}, {}
    agg_rows = []
    for avatar_id, reviews in avatar_reviews.items():
        item_ids = []
        seen = set()
        for review in reviews:
            business_id = review["business_id"]
            if business_id in business_to_item and business_id not in seen:
                item_ids.append(business_to_item[business_id])
                seen.add(business_id)
        train[avatar_id], valid[avatar_id], test[avatar_id] = split_items(item_ids)
        agg_rows.append(build_agg_row(avatar_id, reviews, business_to_item))

    write_cf_file(cf_dir / "train.txt", train)
    write_cf_file(cf_dir / "train_nodrop.txt", train)
    write_cf_file(cf_dir / "valid.txt", valid)
    write_cf_file(cf_dir / "test.txt", test)

    detail_rows = []
    for item_id in range(len(item_to_business)):
        business_id = item_to_business[item_id]
        row = restaurant_detail.loc[business_id]
        detail_rows.append(
            {
                "movie_id": item_id,
                "title": row.get("name", ""),
                "genres": clean_categories(row.get("categories", "")),
                "rating": float(row.get("stars", 0)),
                "summary": restaurant_summary(row),
                "business_id": business_id,
                "city": row.get("city", ""),
                "price": row.get("price", "?"),
                "halal": bool(row.get("halal", False)),
                "review_count": int(row.get("review_count", 0)),
            }
        )
    pd.DataFrame(detail_rows).to_csv(sim_dir / "movie_detail.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(raw_dir / "agg_top_25.csv", index=False)

    for name in ["all_personas_like_modify.csv", "user_statistic.csv", "avatar_manifest.csv", "knowledge_graph_triples.csv"]:
        src = args.persona_adapter / name
        if src.exists():
            shutil.copy2(src, sim_dir / name)

    mapping = {
        "dataset": args.dataset,
        "n_users": len(avatar_to_yelp_user),
        "n_items": len(business_to_item),
        "avatar_to_yelp_user": avatar_to_yelp_user,
        "business_to_item": business_to_item,
    }
    (dataset_dir / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    print(f"Built datasets/{args.dataset}")
    print(f"Users: {len(avatar_to_yelp_user)}")
    print(f"Items: {len(business_to_item)}")
    print(f"Train interactions: {sum(len(v) for v in train.values())}")
    print(f"Valid interactions: {sum(len(v) for v in valid.values())}")
    print(f"Test interactions: {sum(len(v) for v in test.values())}")


if __name__ == "__main__":
    main()
