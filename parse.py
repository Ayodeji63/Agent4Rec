import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Overall settings
    parser.add_argument('--simulation_name', type=str, default= 'Test',
                        help='The name of one trial of simulation.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--items_per_page', type=int, default=4,
                        help='Number of items per page.')
    parser.add_argument('--num_avatars', type=int, default=20,
                        help='Number of avatars for sandbox simulation.')
    parser.add_argument('--execution_mode', type=str, default= 'parallel',
                        choices=['serial', 'parallel'],
                        help='Specify execution mode: serial or parallel.')

    # Only recommend ground truth
    parser.add_argument("--rec_gt", action="store_true",
                        help="whether to recommend ground truth")
    parser.add_argument('--ground_truth_split', type=str, default='valid',
                        choices=['valid', 'test'],
                        help='Held-out split used as ground truth during avatar simulation.')
    parser.add_argument("--inject_ground_truth", action="store_true",
                        help="Controlled probe: inject one held-out positive into pages where no ground-truth item is exposed.")
    
    # Using wandb
    parser.add_argument("--use_wandb", action="store_true",
                        help="whether to use wandb")
    
    # Only validate the effectiveness of agents
    parser.add_argument("--val_users", action="store_true",
                        help="whether to validate users")
    parser.add_argument('--val_ratio', type=int, default=1,
                        help='Ratio of unobserved items vs ground truth for validation.')
    
    # Advertisement settings
    parser.add_argument("--add_advert", action="store_true",
                        help="whether to add advertisement")
    parser.add_argument("--display_advert", action="store_true",
                        help="whether to display advertisement")
    parser.add_argument('--advert_type', type=str, default='pop_high',
                        choices=['all', 'pop_high', 'pop_low', 'unpop_high', 'unpop_low'],
                        help='Specify advertisement type.')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset to use.')

    # Avatar settings
    parser.add_argument('--n_avatars', type=int, default=3,
                        help='How many avatars to simulate.')
    parser.add_argument('--max_pages', type=int, default=1,
                        help='The maximum page number users would like to view')
    parser.add_argument('--min_browse_pages', type=int, default=1,
                        help='Minimum pages each avatar must browse before it may exit. Use 2+ for more stable simulation metrics.')
    parser.add_argument('--persona_dir', type=str, default=None,
                        help='Optional directory containing all_personas_like_modify.csv and user_statistic.csv.')
    parser.add_argument('--sim_llm_provider', type=str, default='groq',
                        choices=['kimi', 'groq', 'gemini', 'openai'],
                        help='LLM provider used by avatars during simulation.')
    parser.add_argument('--sim_llm_model', type=str, default=None,
                        help='Optional model name for the simulation LLM provider.')
    parser.add_argument('--sim_llm_max_tokens', type=int, default=400,
                        help='Maximum output tokens for each avatar LLM call.')
    parser.add_argument('--sim_llm_min_interval', type=float, default=16.0,
                        help='Minimum seconds between avatar LLM calls across threads. Helps avoid TPM rate limits.')
    parser.add_argument('--sim_embedding_provider', type=str, default='local',
                        choices=['local', 'huggingface', 'gemini'],
                        help='Embedding provider used by avatar memory retrieval.')


    # Recommender settings
    parser.add_argument('--model_path', type=str, default= 'Saved',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'LightGCN',
                        help='Specify model save path.')
    parser.add_argument("--hybrid_rerank", action="store_true",
                        help="Re-rank the recommender candidate pool with persona/history/context signals before simulation.")
    parser.add_argument('--rerank_pool_size', type=int, default=100,
                        help='How many top recommender candidates to re-rank per avatar when --hybrid_rerank is enabled.')
    parser.add_argument('--rerank_cf_weight', type=float, default=0.55,
                        help='Weight for the original collaborative-filtering rank score in hybrid re-ranking.')
    parser.add_argument('--rerank_semantic_weight', type=float, default=0.30,
                        help='Weight for user-history/category semantic match in hybrid re-ranking.')
    parser.add_argument('--rerank_diaspora_weight', type=float, default=0.25,
                        help='Weight for Nigerian-diaspora proxy food signals in hybrid re-ranking.')
    parser.add_argument('--rerank_price_weight', type=float, default=0.12,
                        help='Weight for price sensitivity match in hybrid re-ranking.')
    parser.add_argument('--rerank_social_weight', type=float, default=0.08,
                        help='Weight for historical Yelp rating/review-count social proof in hybrid re-ranking.')
    parser.add_argument('--rerank_penalty_weight', type=float, default=0.35,
                        help='Weight for disliked-category, alcohol/halal, and overpriced mismatch penalties in hybrid re-ranking.')

    # others
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument("--pred_norm", action="store_true",
                        help="pred_norm")

    args, _ = parser.parse_known_args()

    return args
