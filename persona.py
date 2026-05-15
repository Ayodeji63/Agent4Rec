"""
generate_nigerian_diaspora_personas.py

Filters the Yelp dataset and generates Agent4Rec-style simulation personas
for Nigerian diaspora users — with distinct Yoruba, Igbo, and Hausa cultural
food preferences, language patterns, and dining behaviours.

Pipeline:
  1. Filter Yelp businesses to restaurants with relevant cuisine profiles
  2. Filter reviews to active users in target cities
  3. Assign each user to an ethnic group (Yoruba / Igbo / Hausa)
  4. Generate culturally-grounded personas via Gemini

Usage:
  python generate_nigerian_diaspora_personas.py \
      --review_path   data/yelp_academic_dataset_review.json \
      --business_path data/yelp_academic_dataset_business.json \
      --output_dir    datasets/yelp-nigerian/simulation \
      --n_users       1000 \
      --min_reviews   10 \
      --city          "Philadelphia"
"""

import os, json, random, time, argparse
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from tqdm import tqdm

load_dotenv()


# ─────────────────────────────────────────────────────────────
# 1.  ETHNIC GROUP DEFINITIONS
# ─────────────────────────────────────────────────────────────

ETHNIC_GROUPS = {
    "yoruba": {
        "weight": 0.40,          # ~40 % of Nigerian diaspora sample
        "spice_tolerance": "very high",
        "food_values": (
            "You are Yoruba Nigerian living in the diaspora. Food must be bold and peppery — "
            "you judge a restaurant first by how much heat it carries. You grew up on Amala, "
            "Ewedu, Gbegiri, Efo Riro, Ofada rice, and thick palm-oil soups. You actively seek "
            "West African and Nigerian restaurants abroad but will also eat at Caribbean, Ethiopian, "
            "or Ghanaian spots that come close to home. You are very vocal when spice is lacking — "
            "bland food is almost offensive. You love communal, family-style dining and loud, "
            "vibrant atmospheres. Price matters but flavour matters more."
        ),
        "language_hints": (
            "Sprinkle in natural Yoruba-Nigerian Pidgin expressions: "
            "'omo', 'e jo', 'shebi', 'wahala', 'abi', 'na wa o', 'ehn ehn', 'jare'. "
            "Write review snippets the way a Yoruba person in the diaspora would actually text — "
            "warm, expressive, slightly dramatic when food is good or bad."
        ),
        "review_style": "expressive and opinionated, praises pepper and bold flavour, complains loudly about blandness",
        "halal_sensitive": False,
    },

    "igbo": {
        "weight": 0.35,
        "spice_tolerance": "moderate",
        "food_values": (
            "You are Igbo Nigerian living in the diaspora. You appreciate richness and depth of "
            "flavour over raw heat — a well-made Ofe Onugbu, Oha soup, or Nkwobi is your gold "
            "standard. You are very value-conscious: portion size, quality of ingredients, and "
            "price-to-satisfaction ratio are always on your mind. You enjoy Igbo staples like "
            "Ofe Akwu, Abacha, and well-seasoned jollof rice. In the diaspora you patronise "
            "West African restaurants but also Nigerian-owned businesses specifically. "
            "You are loyal — if a place earns your trust you keep going back and bring family. "
            "You dislike overpriced food with small portions, and you will absolutely leave a review."
        ),
        "language_hints": (
            "Sprinkle in natural Igbo-Nigerian Pidgin expressions: "
            "'nna', 'biko', 'chai', 'nneka', 'tufiakwa', 'ndo', 'oga'. "
            "Write review snippets the way an Igbo person in the diaspora would — "
            "measured, practical, very specific about value for money and portion size."
        ),
        "review_style": "analytical and value-focused, mentions portion size and ingredient quality, loyal to good finds",
        "halal_sensitive": False,
    },

    "hausa": {
        "weight": 0.25,
        "spice_tolerance": "low to moderate",
        "food_values": (
            "You are Hausa Nigerian living in the diaspora. Halal compliance is non-negotiable — "
            "you always check whether meat is halal before ordering, and a non-halal restaurant "
            "gets a pass unless they have good vegetarian or seafood options. You love grilled "
            "meats deeply: suya culture runs through your dining preferences — smoky, well-seasoned "
            "meat skewers are always a win. You grew up on Tuwo Shinkafa, Miyan Kuka, Dan Wake, "
            "Kilishi, and the warmth of Northern Nigerian hospitality. You prefer savoury depth "
            "over extreme spice. The attitude and warmth of staff matter enormously to you — "
            "cold or rude service will cost a restaurant a star regardless of food quality. "
            "You appreciate Middle Eastern, Turkish, and African Muslim restaurants in the diaspora."
        ),
        "language_hints": (
            "Sprinkle in natural Hausa-Nigerian Pidgin expressions: "
            "'wallahi', 'kai', 'sannu', 'ba komai', 'to', 'ai', 'haba'. "
            "Write review snippets the way a Hausa person in the diaspora would — "
            "calm, hospitable in tone, very attentive to halal status and staff warmth."
        ),
        "review_style": "calm and service-focused, always notes halal status, praises grilled meats and warm hospitality",
        "halal_sensitive": True,
    },
}


# ─────────────────────────────────────────────────────────────
# 2.  RESTAURANT RELEVANCE SCORING
#     Gives each restaurant a relevance score for Nigerian
#     diaspora users based on cuisine category keywords.
# ─────────────────────────────────────────────────────────────

HIGH_RELEVANCE_KEYWORDS = [
    "nigerian", "west african", "african", "ghanaian", "senegalese",
    "ethiopian", "eritrean", "caribbean", "jamaican", "halal",
    "middle eastern", "turkish", "lebanese", "mediterranean",
    "soul food", "southern", "bbq", "barbecue", "grilled",
]

MODERATE_RELEVANCE_KEYWORDS = [
    "seafood", "american", "comfort food", "creole", "cajun",
    "indian", "pakistani", "bangladeshi", "rice", "chicken",
]


def score_restaurant(categories: str) -> int:
    cats_lower = (categories or "").lower()
    if not any(k in cats_lower for k in ["restaurant", "food", "bar", "grill", "kitchen"]):
        return 0
    for kw in HIGH_RELEVANCE_KEYWORDS:
        if kw in cats_lower:
            return 2
    for kw in MODERATE_RELEVANCE_KEYWORDS:
        if kw in cats_lower:
            return 1
    return 1   # generic restaurant still counts


# ─────────────────────────────────────────────────────────────
# 3.  DATA LOADING & FILTERING
# ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_restaurant_index(business_path, city=None):
    print("Loading businesses...")
    businesses = load_jsonl(business_path)
    restaurants = {}
    for b in businesses:
        cats = b.get("categories") or ""
        if city and b.get("city", "").lower() != city.lower():
            continue
        score = score_restaurant(cats)
        if score == 0:
            continue
        attrs = b.get("attributes") or {}
        restaurants[b["business_id"]] = {
            "name":       b["name"],
            "categories": cats,
            "city":       b.get("city", ""),
            "stars":      b.get("stars", 0),
            "price":      attrs.get("RestaurantsPriceRange2", "?"),
            "halal":      "halal" in cats.lower() or
                          str(attrs.get("RestaurantsAttire", "")).lower() == "halal",
            "relevance":  score,
        }
    print(f"  → {len(restaurants):,} restaurants kept")
    return restaurants


def build_user_reviews(review_path, restaurant_ids, min_reviews=10):
    print("Loading reviews...")
    all_reviews = load_jsonl(review_path)
    user_reviews = defaultdict(list)
    for r in all_reviews:
        if r["business_id"] not in restaurant_ids:
            continue
        user_reviews[r["user_id"]].append({
            "business_id": r["business_id"],
            "stars":       r["stars"],
            "text":        r["text"][:300],
            "date":        r.get("date", ""),
        })
    user_reviews = {
        uid: revs for uid, revs in user_reviews.items()
        if len(revs) >= min_reviews
    }
    print(f"  → {len(user_reviews):,} users with >= {min_reviews} reviews")
    return user_reviews


# ─────────────────────────────────────────────────────────────
# 4.  ETHNIC GROUP ASSIGNMENT
#     Weighted random assignment so the sample reflects
#     approximate Nigerian diaspora demographics.
# ─────────────────────────────────────────────────────────────

def assign_ethnic_groups(user_ids: list, seed: int = 42) -> dict:
    """Returns {user_id: ethnic_group_name}"""
    rng = random.Random(seed)
    groups      = list(ETHNIC_GROUPS.keys())
    weights     = [ETHNIC_GROUPS[g]["weight"] for g in groups]
    assignments = {}
    for uid in user_ids:
        assignments[uid] = rng.choices(groups, weights=weights, k=1)[0]
    return assignments


# ─────────────────────────────────────────────────────────────
# 5.  PERSONA GENERATION
# ─────────────────────────────────────────────────────────────

def build_system_prompt(ethnic_group: str) -> str:
    grp = ETHNIC_GROUPS[ethnic_group]
    return f"""You are an expert at building culturally authentic user personas.

CULTURAL IDENTITY:
{grp['food_values']}

LANGUAGE & TONE:
{grp['language_hints']}
Review style: {grp['review_style']}

OUTPUT FORMAT — repeat this block 6-10 times, one theme per block:

LIKE: [one sentence about a food or dining preference, written in first person, culturally grounded]
REASON: [one sentence explaining why, may include a Pidgin/ethnic phrase naturally]
RESTAURANT: [restaurant name]; [restaurant name]; ...

RULES:
- Only reference restaurants actually given in the input
- Focus LIKE themes on 4-5 star restaurants, mention 1-2 star ones only in a dislike context
- Each theme must be distinct (spice level, cuisine type, ambience, service, halal, portion size, etc.)
- DO NOT add any text outside the LIKE/REASON/RESTAURANT blocks
- Make it sound like a real Nigerian person in the diaspora, not a textbook description
{'- Always include at least one block specifically about halal food availability' if grp['halal_sensitive'] else ''}
"""


def build_user_context(user_reviews: list, restaurant_index: dict) -> str:
    lines = []
    for r in sorted(user_reviews, key=lambda x: x["stars"], reverse=True)[:30]:
        biz = restaurant_index.get(r["business_id"])
        if not biz:
            continue
        halal_tag = " [HALAL]" if biz["halal"] else ""
        lines.append(
            f'- "{biz["name"]}" | {biz["categories"][:60]}{halal_tag} '
            f'| price={biz["price"]} | {r["stars"]}★ | "{r["text"][:120]}"'
        )
    return "\n".join(lines)


def generate_persona(llm, system_prompt: str, user_context: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User's restaurant history:\n{user_context}\n\nGenerate the persona:"),
            ])
            return response.content.strip()
        except Exception as e:
            print(f"  Gemini error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return "LIKE: I enjoy dining out.\nREASON: Good food nourishes the soul.\nRESTAURANT: N/A"


# ─────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_path",   required=True)
    parser.add_argument("--business_path", required=True)
    parser.add_argument("--output_dir",    default="datasets/yelp-nigerian/simulation")
    parser.add_argument("--n_users",       type=int, default=1000)
    parser.add_argument("--min_reviews",   type=int, default=10)
    parser.add_argument("--city",          default=None,
                        help="Filter to one city e.g. 'Philadelphia'")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────
    restaurant_index = build_restaurant_index(args.business_path, city=args.city)
    user_reviews     = build_user_reviews(
        args.review_path, set(restaurant_index.keys()), min_reviews=args.min_reviews
    )

    all_uids = list(user_reviews.keys())
    if len(all_uids) > args.n_users:
        selected = random.sample(all_uids, args.n_users)
    else:
        selected = all_uids
        print(f"  Warning: only {len(all_uids)} qualifying users, using all.")

    # ── Assign ethnic groups ───────────────────────────────────
    ethnic_assignments = assign_ethnic_groups(selected, seed=args.seed)

    # Print distribution
    from collections import Counter
    dist = Counter(ethnic_assignments.values())
    print(f"\nEthnic group distribution:")
    for grp, count in dist.items():
        print(f"  {grp.capitalize():8s}: {count:4d} users  ({count/len(selected)*100:.1f}%)")

    # ── Init LLM ──────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.75,
        convert_system_message_to_human=True,
    )

    # ── Generate personas ──────────────────────────────────────
    print(f"\nGenerating {len(selected)} personas...\n")
    user_index_map = {}

    for idx, uid in enumerate(tqdm(selected)):
        ethnic_group  = ethnic_assignments[uid]
        system_prompt = build_system_prompt(ethnic_group)
        user_context  = build_user_context(user_reviews[uid], restaurant_index)

        if not user_context.strip():
            continue

        persona_text = generate_persona(llm, system_prompt, user_context)

        # Save persona
        persona_path = out / f"persona_{idx}.txt"
        with open(persona_path, "w", encoding="utf-8") as f:
            f.write(persona_text)

        user_index_map[idx] = {
            "yelp_user_id":  uid,
            "ethnic_group":  ethnic_group,
            "n_reviews":     len(user_reviews[uid]),
        }

        time.sleep(0.3)   # rate limit buffer

    # ── Save index ──────────────────────────────────────────────
    index_path = out / "users.json"
    with open(index_path, "w") as f:
        json.dump(user_index_map, f, indent=2)

    # Summary
    print(f"\n✓ {len(user_index_map)} personas saved to {out}/")
    print(f"✓ User index saved to {index_path}")
    print("\nSample persona (persona_0.txt):")
    print("-" * 50)
    with open(out / "persona_0.txt") as f:
        print(f.read()[:600])


if __name__ == "__main__":
    main()