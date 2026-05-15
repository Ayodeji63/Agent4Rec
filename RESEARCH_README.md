# Nigerian-Diaspora Food Recommendation Simulation

## Research Context

This project extends Agent4Rec from a movie-centered generative-agent simulator into a restaurant recommendation simulator for Nigerian users in diaspora. The system is designed to evaluate how a recommender behaves when users are not treated as generic clickers, but as socially and culturally situated diners with memory, contextual needs, price sensitivity, and food perception priors.

The work is motivated by a limitation in the original Agent4Rec-style setup: the simulated user mostly reacts to item metadata and a persona prompt, while real users are influenced by external and contextual factors such as visual appeal, social proof, price, location, mood, cultural taste, service expectation, and prior dining experience. The implementation therefore adds SimUSER-inspired modules around persona construction, episodic memory, knowledge graph grounding, perception, and brain-like decision making.

## Core Research Question

The central question is:

Can a recommender system for restaurants be evaluated more realistically by combining collaborative filtering with Nigerian-diaspora-aware user simulation, memory, contextual perception, and grounded behavioral metrics?

The system does not assume that Nigerian users in diaspora only want Nigerian restaurants. Instead, it models diaspora preference through proxy signals that are available in a US Yelp-style dataset:

- bold seasoning and pepper-forward cuisines
- grilled meat, barbecue, seafood, rice, stews, and filling meals
- Indian, Thai, Mexican, Vietnamese, Chinese/Szechuan, Middle Eastern, African, Caribbean, and Southern/Soul food as possible flavor proxies
- generous portions and price fairness
- halal or alcohol-sensitive context where relevant
- lively atmosphere, social dining, and practical value

This framing is important because the Yelp dataset is not an African or Nigerian dataset. A strict Nigerian-cuisine filter would be inappropriate and would reduce both realism and recommendation coverage.

## Relationship To Agent4Rec

Agent4Rec provides the base architecture:

- a recommender model such as MF or LightGCN
- simulated avatars initialized from historical user behavior
- page-by-page recommendation exposure
- avatar actions such as selection, rating, exit, and post-interaction interview
- behavioral metrics such as clicks, ratings, exit page, precision, and recall

In the original formulation, the agent primarily evaluates recommended items using the persona and item description. This project keeps the Agent4Rec interaction loop but adapts the domain from movies to restaurants and adds richer user grounding.

## Relationship To SimUSER

The SimUSER paper provides the conceptual basis for more realistic simulated users. The relevant SimUSER concepts used in this project are:

- Persona profile: the stable user identity, tastes, preferences, and behavioral tendencies.
- Episodic memory: records of past interactions, liked items, disliked items, review tone, and rating behavior.
- Knowledge graph memory: structured relations such as user -> likes -> cuisine, user -> dislikes -> overpriced spots, user -> price_sensitive -> high.
- Perception module: item interpretation using observable cues before final decision making.
- Brain module: reasoning over persona, memory, perception, and current context before producing an action.
- Context layer: time, location, mood, budget, and task goal.

The implementation adapts these ideas to the available Yelp metadata. Since real thumbnails and full review text are not always available, the perception module uses proxy cues from category, price, review count, historical rating, nightlife/alcohol signals, cuisine type, and summary text.

## System Architecture

The final system is a hybrid pipeline:

```text
Historical Yelp interactions
        |
        v
Persona and memory construction
        |
        v
LightGCN candidate generation
        |
        v
Nigerian-diaspora-aware hybrid reranker
        |
        v
SimUSER-style avatar interaction
        |
        v
Behavioral and grounding evaluation
```

## Persona Module

Each avatar receives a persona built from historical restaurant interactions. The persona includes:

- liked items
- disliked items
- past rating patterns
- past review tone
- frequent categories
- price sensitivity
- cultural identity field, such as Yoruba, Igbo, Hausa, or general Nigerian
- pickiness and dining activity level

The cultural identity is treated as a strong interpretive prior, not as a hard filter. A Yoruba avatar may value pepper level, social energy, and atmosphere; an Igbo avatar may emphasize portion size, hearty food, value, and practicality; a Hausa avatar may give greater attention to cleanliness, family suitability, and halal/alcohol-related cues. These priors can be overridden by the user's actual history.

## Episodic Memory Module

The episodic memory module stores historical preferences and newly observed interactions. It includes:

- restaurants previously liked
- restaurants previously disliked
- categories frequently selected
- ratings from historical reviews
- simulated reactions during the recommendation session

This memory is used to make the avatar less generic. The avatar is not only a prompt describing a cultural group; it is also a user with a personal dining history.

## Knowledge Graph Module

The knowledge graph stores lightweight triples such as:

```text
user -> likes -> spicy food
user -> dislikes -> overpriced restaurants
user -> historical_like_5 -> restaurant_name
user -> price_sensitivity -> high
restaurant -> has_category -> Mexican
restaurant -> has_price -> 2
```

The purpose of the knowledge graph is not to build a full ontology. It is used as a compact grounding layer that connects user preferences to item attributes. During recommendation, the avatar can reason over paths such as:

```text
user -> likes -> bold spice -> similar_to -> Szechuan/Indian/Mexican restaurant
user -> dislikes -> high price -> conflicts_with -> price range 4 restaurant
```

## Perception Module

The perception module estimates how the avatar may perceive each restaurant before choosing. Since the Yelp dataset does not include real thumbnails in the current pipeline, perception is approximated using structured metadata:

- cuisine/category
- price range
- Yelp rating
- number of reviews
- nightlife/alcohol-heavy signals
- halal risk signals
- bold flavor proxies
- premium or pretentiousness risk
- social proof from review count and rating

This is a proxy for thumbnail/social/context perception rather than a replacement for true multimodal modeling. If real images or menu text become available, this module should be extended with visual and textual encoders.

## Brain Module

The brain module is the avatar's decision process. It considers:

- persona preferences
- episodic memory
- knowledge graph evidence
- perception cues
- context layer
- current page number
- pickiness and activity group

The avatar then produces:

- selected restaurants
- ratings
- feelings
- alignment decisions
- exit or continue behavior

The system also enforces a minimum browsing depth when configured, because extremely selective simulated users may otherwise exit too early and make evaluation unstable.

## Context Layer

The context layer adds situational factors:

- location: Lagos Island, Victoria Island, Lekki, Ikeja, Ibadan, Abuja, campus, work
- time: morning, afternoon, evening, night
- mood: tired, excited, hungry, budget-conscious
- goal: buy food
- budget: low, medium, high

These factors help model the difference between, for example, a late-night quick meal, a campus lunch, and a family-suitable weekend restaurant.

## Hybrid Reranker

The original LightGCN recommender is retained as the candidate generator. A hybrid reranker is then applied to the top candidate pool before items are shown to avatars.

The reranker combines:

- collaborative filtering rank score
- semantic overlap with user history and frequent categories
- Nigerian-diaspora proxy score
- price-sensitivity match
- Yelp rating and review-count social proof
- penalties for disliked categories, alcohol-heavy mismatch, halal risk, and overpriced mismatches

This approach is more appropriate than injecting ground truth because it improves the natural ranking process without giving the model the answer.

## Ground Truth Exposure Problem

The observed precision and recall are low because the held-out ground truth is sparse. In the Yelp-kimi split, each user has only one validation item and one test item. With 20 displayed items, the theoretical maximum precision for a user with one relevant held-out item is:

```text
1 / 20 = 0.05
```

The trained LightGCN model achieves approximately:

```text
Recall@20 ~= 0.159
Precision@20 ~= 0.008
```

Therefore, low raw precision is expected under the sparse held-out protocol. The important question is whether the system improves exposure, recognition, satisfaction, and click behavior relative to the baseline.

## Why Ground Truth Injection Is Not Used For Natural Evaluation

The flag `--inject_ground_truth` forces a held-out positive item into the recommendation page when no ground-truth item appears naturally. This makes recall measurable, but it changes the exposure process.

It answers:

```text
If the correct item is shown, can the avatar recognize it?
```

It does not answer:

```text
Can the recommender naturally surface the correct item?
```

For this reason, ground-truth injection should only be reported as a controlled recognition probe. It should not be used for natural recommender precision, recall, or ranking-quality claims.

## Handling Invalid LLM Outputs

The project avoids silently replacing failed LLM ratings with `3.0` in final research reporting. Such imputation would preserve runtime stability but could bias satisfaction and rating distributions.

The research-clean protocol is:

```text
1. Request strict structured output.
2. Validate the structure.
3. Retry or repair only when fields are recoverable.
4. Mark unrecoverable outputs as parse_failed=True.
5. Exclude invalid pages from rating averages.
6. Report parse-valid rate separately.
```

This is more defensible than silent imputation. It is also different from ground-truth injection: invalid-output handling affects measurement reliability, while ground-truth injection changes what the recommender exposes.

## Evaluation Metrics

The system reports two classes of metrics.

### Classical Ranking Metrics

- Precision
- Recall
- Micro precision
- Micro recall over exposed items
- Ground-truth item exposure
- Exposed-user hit rate

These metrics are retained for comparability with recommender literature, but they are interpreted cautiously because the held-out set is sparse.

### Behavioral Simulation Metrics

- number of likes
- average rating
- click rate
- exposure-adjusted click rate
- true satisfaction score from interview
- exit page
- grounded recognition rate
- parse-valid rate, when structured-output validation is enabled

For this project, behavioral metrics are especially important because the goal is not only to rank held-out items, but to simulate whether culturally and contextually grounded users find the recommendations useful.

## Current Experimental Finding

In the 50-avatar natural evaluation, the hybrid reranker improved the system relative to the LightGCN-only baseline.

```text
Ground-truth exposed:        4  ->  7
Ground-truth hits:           2  ->  3
Micro precision:        0.0068 -> 0.0090
Overall click rate:      0.284 -> 0.311
Exposure-adjusted CTR:   0.411 -> 0.442
True satisfaction:        0.66 -> 0.738
Average exit page:        3.40 -> 3.46
Average likes/page:      1.696 -> 1.914
```

The precision remains numerically small, but the relative improvement is meaningful under the sparse held-out protocol. The result suggests that Nigerian-diaspora-aware reranking improves both natural exposure and simulated user satisfaction.

## Recommended Reporting Language

The result should not be reported as simply:

```text
Precision = 0.009
```

Instead, it should be reported as:

```text
The hybrid Nigerian-diaspora reranker improved natural ground-truth exposure from 4 to 7 items and increased micro precision from 0.0068 to 0.0090. It also improved behavioral quality, raising true satisfaction from 0.66 to 0.738 and exposure-adjusted click rate from 0.411 to 0.442.
```

This wording makes clear that the contribution is not only classical ranking accuracy, but also culturally grounded behavioral evaluation.

## Limitations

The current system has several limitations:

- The Yelp dataset is US-centered and contains few explicitly Nigerian or African restaurants.
- Cultural food preference is therefore modeled using diaspora proxy cues rather than direct Nigerian cuisine labels.
- Real thumbnails, menus, and image-based food perception are not yet included.
- LLM outputs can fail structured parsing and should be validated rather than silently imputed.
- Running many avatars is expensive and slow because each page requires LLM interaction.
- Offline ground truth is sparse and incomplete; many useful recommendations may not be present in the held-out set.

## Future Work

Future improvements should include:

- multimodal perception using restaurant thumbnails and menu images
- menu-text grounding for pepper, spice, meat, halal, portion, and price signals
- better structured-output enforcement using schema validation
- calibrated parse-failure reporting
- larger user subsets with stratified sampling
- Nigerian or African diaspora restaurant datasets
- learning-to-rank over the hybrid reranker features
- comparison against LLM-only reranking, LightGCN-only ranking, and popularity baselines

## Key Terms

Agent4Rec:
An agent-based recommender simulation framework where LLM-powered avatars interact with recommender outputs over multiple pages.

Avatar:
A simulated user agent with profile, memory, and behavioral traits.

Persona:
The avatar's stable identity, preferences, taste profile, and behavioral description.

Episodic Memory:
A memory store of past user-specific interactions and simulated recommendation experiences.

Knowledge Graph:
A structured set of triples connecting users, preferences, items, categories, prices, and dislikes.

Perception Module:
The component that interprets item cues before decision making. In this implementation, perception is based on metadata proxies.

Brain Module:
The reasoning component that combines persona, memory, perception, context, and recommendation evidence to decide actions.

Ground Truth Exposure:
Whether the recommender naturally displays a held-out validation or test item to the avatar.

Grounded Recognition:
Whether the avatar selects or aligns with a ground-truth item after it has been exposed.

Controlled Recognition Probe:
An evaluation condition where ground-truth items are intentionally injected to test avatar recognition. This is not natural ranking evaluation.

Hybrid Reranking:
A reranking method that combines collaborative filtering scores with semantic, cultural, price, and social-proof features.

Parse-Valid Rate:
The percentage of LLM responses that successfully match the required structured output format.

## References

- Agent4Rec: On Generative Agents in Recommendation. Project repository and base simulator.
- SimUSER: Simulated user modeling with persona, memory, perception, knowledge graph, and brain modules.
- He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. https://arxiv.org/abs/2002.02126
- Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. https://arxiv.org/abs/1205.2618
- SoEval: A Benchmark for Structured Outputs of LLMs. https://www.sciencedirect.com/science/article/abs/pii/S0306457324001687
- JSONSchemaBench: Evaluating Constrained Decoding for Structured Generation. https://openreview.net/pdf?id=87f0994dff5f854cb02110866e3c61a8e14c80f2
- Good-Enough Structured Generation: A Case Study. https://openreview.net/pdf?id=p84kZ3ZFux
- LLM-powered User Simulator for Recommender Systems. https://arxiv.org/abs/2412.16984
- A Survey on LLM-powered Agents for Recommender Systems. https://aclanthology.org/2025.findings-emnlp.620.pdf
- Consumer-side Fairness in Recommender Systems: a Survey of Methods and Evaluation. https://link.springer.com/article/10.1007/s10462-023-10663-5
