from simulation.base.abstract_avatar import abstract_avatar
from simulation.memory import AvatarMemory

from termcolor import colored, cprint
import openai
import os

import re
import hashlib
import threading
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from simulation.retriever import AvatarRetriver
import time
import datetime
import torch
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


import wandb

import simulation.vars as vars

_llm_rate_lock = threading.Lock()
_last_llm_call_at = 0.0


class HashEmbeddings:
    """Small local embedding fallback to avoid quota-bound embedding APIs."""

    def __init__(self, size=384):
        self.size = size

    def _embed(self, text):
        vector = np.zeros(self.size, dtype=np.float32)
        for token in re.findall(r"\w+", str(text).lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_query(self, text):
        return self._embed(text)

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]


def _env_first(*keys):
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _clean_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "n/a"} else text


def _split_semicolon(value):
    text = _clean_value(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r";|\|", text) if part.strip()]


def _clamp_rating(value):
    try:
        rating = float(str(value).strip().strip(";"))
    except (TypeError, ValueError):
        return 3.0
    return max(1.0, min(5.0, rating))


def _strip_thinking(text):
    """Remove reasoning blocks emitted by thinking models before parsing."""
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if "<think>" in text.lower():
        structured_markers = ["CONTEXT:", "MOVIE:", "NUM:", "[EXIT]", "[NEXT]", "POSITIVE:", "NEGATIVE:"]
        marker_positions = [
            pos for marker in structured_markers
            for pos in [text.upper().find(marker)]
            if pos >= 0
        ]
        text = text[min(marker_positions):] if marker_positions else ""
    return text.strip()


class Avatar(abstract_avatar):
    def __init__(self, args, avatar_id, init_property, init_statistic):
        super().__init__(args, avatar_id)

        dataset_name = str(getattr(args, "dataset", "")).lower()
        self.is_restaurant_domain = dataset_name.startswith("yelp") or "restaurant" in dataset_name
        self.item_kind = "restaurant" if self.is_restaurant_domain else "movie"
        self.item_plural = "restaurants" if self.is_restaurant_domain else "movies"
        self.action_verb = "try" if self.is_restaurant_domain else "watch"
        self.action_gerund = "trying" if self.is_restaurant_domain else "watching"
        self.recommender_domain = "restaurant recommendation system" if self.is_restaurant_domain else "movie recommendation system"
        self.taste_domain = "restaurant and cuisine tastes" if self.is_restaurant_domain else "movie tastes"

        self.parse_init_property(init_property)
        self.parse_init_statistic(init_statistic)
        self.log_file = f"storage/{args.dataset}/{args.modeltype}/{args.simulation_name}/running_logs/{avatar_id}.txt"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.init_memory()

    def _warn_invalid_profile(self, field, fallback):
        cprint(
            f"Avatar {self.avatar_id}: missing or invalid {field}; using fallback profile value.",
            color="yellow",
        )
        return fallback

    def parse_init_property(self, init_property):
        raw_taste = _clean_value(init_property.get("taste", ""))
        if not raw_taste:
            raw_taste = self._warn_invalid_profile(
                "taste",
                f"I prefer {self.item_plural} with good quality, fair value, reliable service, and tastes that match my history.",
            )
        self.taste = [taste.strip() for taste in raw_taste.split("| ") if taste.strip()]
        if not self.taste:
            self.taste = [self._warn_invalid_profile(
                "taste list",
                f"I prefer {self.item_plural} with good quality, fair value, reliable service, and tastes that match my history.",
            )]
        self.high_rating = _clean_value(init_property.get("high_rating", ""))
        if not self.high_rating:
            self.high_rating = self._warn_invalid_profile(
                "high_rating",
                f"You usually give higher ratings to {self.item_plural} that match your stated tastes, provide good value, and feel satisfying.",
            )
        self.reason = _clean_value(init_property.get("reason", ""))
        self.ethnic_group = str(init_property.get("ethnic_group", "") or "").strip().lower()
        if self.is_restaurant_domain and self.ethnic_group not in {"yoruba", "igbo", "hausa"}:
            self.ethnic_group = self._warn_invalid_profile("ethnic_group", "nigerian")
        self.pickiness_label = _clean_value(init_property.get("pickiness", ""))
        self.liked_items = _split_semicolon(init_property.get("liked_items", ""))
        self.disliked_items = _split_semicolon(init_property.get("disliked_items", ""))
        self.history_titles = _split_semicolon(init_property.get("history_titles", ""))
        self.history_categories = _split_semicolon(init_property.get("history_categories", ""))
        self.history_ratings = []
        for rating in _split_semicolon(init_property.get("history_ratings", "")):
            try:
                self.history_ratings.append(float(rating))
            except ValueError:
                continue
        self.past_rating_patterns = _clean_value(init_property.get("past_rating_patterns", ""))
        self.past_review_tone = _clean_value(init_property.get("past_review_tone", ""))
        self.frequent_categories = _split_semicolon(init_property.get("frequent_categories", ""))
        self.price_sensitivity = _clean_value(init_property.get("price_sensitivity", "medium")) or "medium"
        if self.price_sensitivity.lower() not in {"low", "medium", "high"}:
            self.price_sensitivity = self._warn_invalid_profile("price_sensitivity", "medium")
        self.evidence_items = _split_semicolon(init_property.get("evidence_items", ""))
        self.knowledge_graph = _clean_value(init_property.get("knowledge_graph", ""))
        self.kg_edges = self._init_kg_edges()
        self.cultural_context = self._build_cultural_context()
        self.episodic_memory_context = self._build_episodic_memory_context()
        self.knowledge_graph_context = self._build_knowledge_graph_context()

    def _build_cultural_context(self):
        if not self.is_restaurant_domain:
            return ""
        cultural_priors = {
            "yoruba": {
                "identity": "Yoruba Nigerian",
                "food_preferences": "pepper-forward soups and stews, amala-style comfort, suya, grilled or fried fish, jollof/rice dishes, and lively social dining",
                "dining_style": "communal, expressive, hospitable, attentive to portions, pepper level, service, and atmosphere",
                "price_view": "value-conscious, but willing to stretch for quality, celebration, or strong vibes",
                "rating_style": "warm and expressive; mentions vibes, pepper, portion, service, and whether the outing felt worth it",
                "discomfort": "bland food, cold service, small portions, weak pepper, inflated prices, and places that feel soulless",
            },
            "igbo": {
                "identity": "Igbo Nigerian",
                "food_preferences": "hearty soups, pepper soup, goat meat, grilled or fried fish, rice dishes, filling portions, and practical comfort food",
                "dining_style": "direct, practical, quality-focused, and attentive to whether the food is filling and worth the money",
                "price_view": "price-sensitive but generous when quality, quantity, and service justify the bill",
                "rating_style": "direct and honest; mentions specific dishes, value, portions, and whether they would recommend it",
                "discomfort": "overpriced small portions, weak seasoning, poor pepper level, careless service, and food that looks better than it tastes",
            },
            "hausa": {
                "identity": "Hausa Nigerian",
                "food_preferences": "halal-friendly meats, suya/kilishi-style grilled flavors, rice dishes, masa-like snacks, soups, clean preparation, and familiar savory depth",
                "dining_style": "measured, family-aware, cleanliness-conscious, and attentive to halal suitability where relevant",
                "price_view": "moderate and practical, with preference for affordable everyday spots that feel clean and reliable",
                "rating_style": "measured and clear; mentions cleanliness, halal confidence, service, portion, and whether it is family-appropriate",
                "discomfort": "pork-adjacent dishes, alcohol-heavy environments, unclear halal status, poor hygiene, and overly noisy spaces",
            },
        }
        profile = cultural_priors.get(
            self.ethnic_group,
            {
                "identity": "Nigerian",
                "food_preferences": "bold seasoning, pepper/heat, rice or stew-like comfort, grilled or fried meats/fish, generous portions, and hospitable dining",
                "dining_style": "social, practical, value-aware, and attentive to service, atmosphere, portion size, and authenticity",
                "price_view": "value-conscious urban diner, but willing to pay for quality and occasion",
                "rating_style": "specific, warm, and expressive; mentions food quality, pepper, value, service, and atmosphere",
                "discomfort": "bland food, inflated prices, poor portions, poor service, and places that feel inauthentic",
            },
        )
        return (
            f"You are role-playing a {profile['identity']} restaurant user. "
            f"Treat this cultural profile as a strong prior for interpreting restaurants, while still letting the observed Yelp history and explicit persona evidence override it when they conflict. "
            f"Food preferences lean toward: {profile['food_preferences']}. "
            f"Dining style: {profile['dining_style']}. "
            f"Price perspective: {profile['price_view']}. "
            f"Review/rating voice: {profile['rating_style']}. "
            f"Discomfort signals: {profile['discomfort']}."
        )

    def _build_episodic_memory_context(self):
        if not self.is_restaurant_domain:
            return ""
        lines = []
        if self.liked_items:
            lines.append(f"liked_items={', '.join(self.liked_items[:5])}")
        if self.disliked_items:
            lines.append(f"disliked_items={', '.join(self.disliked_items[:4])}")
        if self.past_rating_patterns:
            lines.append(f"past_rating_patterns={self.past_rating_patterns}")
        if self.frequent_categories:
            lines.append(f"frequent_categories={', '.join(self.frequent_categories[:6])}")
        if self.price_sensitivity:
            lines.append(f"price_sensitivity={self.price_sensitivity}")
        if self.evidence_items:
            lines.append(f"evidence_restaurants={', '.join(self.evidence_items[:4])}")
        if self.history_titles:
            lines.append(f"historical_interactions={', '.join(self.history_titles[:6])}")
        return "; ".join(lines)

    def _build_knowledge_graph_context(self):
        if not self.is_restaurant_domain:
            return ""
        return " | ".join(self.kg_edges[:14])

    def _init_kg_edges(self):
        edges = []
        if self.knowledge_graph:
            edges.extend(edge.strip() for edge in self.knowledge_graph.split("|") if edge.strip())
        edges.extend(f"user -> likes -> {item}" for item in self.liked_items[:6])
        edges.extend(f"user -> dislikes -> {item}" for item in self.disliked_items[:4])
        for title, rating in list(zip(self.history_titles, self.history_ratings))[:12]:
            if rating >= 4:
                edges.append(f"user -> historical_like_{rating:g} -> {title}")
            elif rating <= 2:
                edges.append(f"user -> historical_dislike_{rating:g} -> {title}")
            else:
                edges.append(f"user -> historical_neutral_{rating:g} -> {title}")
        if self.price_sensitivity:
            edges.append(f"user -> price_sensitivity -> {self.price_sensitivity}")
        if self.ethnic_group:
            edges.append(f"user -> ethnic_group -> {self.ethnic_group}")
        seen = set()
        deduped = []
        for edge in edges:
            key = edge.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(edge)
        return deduped

    def _append_kg_edge(self, subject, relation, obj):
        edge = f"{subject} -> {relation} -> {obj}"
        if edge.lower() not in {existing.lower() for existing in self.kg_edges}:
            self.kg_edges.insert(0, edge)
        self.kg_edges = self.kg_edges[:60]
        self.knowledge_graph_context = self._build_knowledge_graph_context()

    def _initialize_memory_from_history(self):
        if not self.is_restaurant_domain:
            return
        for item in self.liked_items[:8]:
            self.memory.add_memory(f"I liked {item} based on my historical restaurant interactions.", now=datetime.datetime.now())
        for item in self.disliked_items[:6]:
            self.memory.add_memory(f"I disliked {item} based on my historical restaurant interactions.", now=datetime.datetime.now())
        for title, rating in list(zip(self.history_titles, self.history_ratings))[:18]:
            if rating >= 4:
                sentiment = "liked"
            elif rating <= 2:
                sentiment = "disliked"
            else:
                sentiment = "felt neutral about"
            self.memory.add_memory(f"I {sentiment} {title} based on my review score of {rating:g}.", now=datetime.datetime.now())
        if self.past_rating_patterns:
            self.memory.add_memory(f"My past rating pattern is: {self.past_rating_patterns}", now=datetime.datetime.now())
        if self.frequent_categories:
            self.memory.add_memory(f"My frequent restaurant categories are: {', '.join(self.frequent_categories[:8])}.", now=datetime.datetime.now())

    def _context_layer(self, current_page=None):
        if not self.is_restaurant_domain:
            return ""
        locations = ["Lagos Island", "Victoria Island", "Lekki", "Ikeja", "Ibadan", "Abuja", "campus", "work"]
        times = ["morning", "afternoon", "evening", "night"]
        meal_by_time = {
            "morning": "breakfast or brunch",
            "afternoon": "lunch",
            "evening": "dinner or after-work meal",
            "night": "late dinner or hangout spot",
        }
        moods = ["tired", "excited", "hungry", "budget-conscious"]
        budget_by_sensitivity = {
            "high": ["low", "low", "medium"],
            "medium": ["medium", "low", "high"],
            "low": ["medium", "high", "high"],
        }
        key = f"{getattr(self.args, 'seed', 101)}:{self.avatar_id}:{current_page or 0}:{self.ethnic_group}:{self.price_sensitivity}"
        seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        time_context = times[(seed // 7) % len(times)]
        budget_options = budget_by_sensitivity.get(str(self.price_sensitivity).lower(), ["medium", "low", "high"])
        budget = budget_options[(seed // 31) % len(budget_options)]
        if current_page and current_page > 3 and budget == "low":
            budget = "low but willing to stretch slightly for a strong match"
        return (
            f"location={locations[seed % len(locations)]}; "
            f"time={time_context}; "
            f"meal_occasion={meal_by_time[time_context]}; "
            f"mood={moods[(seed // 17) % len(moods)]}; "
            f"goal=buy food; "
            f"budget={budget}"
        )

    def _profile_context_block(self, current_page=None):
        parts = []
        if self.cultural_context:
            parts.append(f"Cultural identity context: {self.cultural_context}")
        if self.episodic_memory_context:
            parts.append(f"Episodic memory: {self.episodic_memory_context}")
        if self.knowledge_graph_context:
            parts.append(f"Knowledge graph triples: {self.knowledge_graph_context}")
        context_layer = self._context_layer(current_page)
        if context_layer:
            parts.append(f"Current context layer: {context_layer}")
        return "\n".join(parts)

    def parse_init_statistic(self, init_statistic):
        """
        Parse the init statistic of the avatar
        """
# diversity_dict
        if self.is_restaurant_domain:
            activity_dict = {1:"An Incredibly Elusive Occasional Diner, so seldom attracted by restaurant recommendations that it is unusual when you decide to try one. Your dining-out habits from recommendations are extraordinarily infrequent. You will exit the recommender system quickly if you feel even slightly unsatisfied.",
                             2:"An Occasional Diner, seldom attracted by restaurant recommendations. You are only curious about trying restaurants that strictly align with your taste. Your restaurant-trying habits are not very frequent, and you tend to exit the recommender system after a few unsatisfied memories.",
                             3:"A Food Enthusiast with a strong appetite for dining discovery, willing to try many restaurants recommended to you. Restaurant discovery is important to you, and recommendations are useful when they match your tastes. You are tolerant of the recommender system, so you are not quick to exit after only a little dissatisfaction."}
        else:
            activity_dict = {1:"An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. Your movie-watching habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
                             2:"An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
                             3:"A Movie Enthusiast with an insatiable appetite for films, willing to watch nearly every movie recommended to you. Movies are a central part of your life, and movie recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory."}
# conformity_dict
        if self.is_restaurant_domain:
            conformity_dict = {1:"A Dedicated Follower whose restaurant ratings rely heavily on historical Yelp ratings, rarely expressing independent opinions. Usually gives ratings close to historical ratings.",
                               2:"A Balanced Evaluator who considers both historical Yelp ratings and personal preferences when giving ratings to restaurants. Sometimes gives ratings that differ from historical ratings.",
                               3:"A Maverick Critic who mostly ignores historical Yelp ratings and evaluates restaurants based on personal taste. Usually gives ratings that can be very different from historical ratings."}
        else:
            conformity_dict = {1:"A Dedicated Follower who gives ratings heavily relies on movie historical ratings, rarely expressing independent opinions. Usually give ratings that are same as historical ratings. ",
                               2:"A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.",
                               3:"A Maverick Critic who completely ignores historical ratings and evaluates movies solely based on own taste. Usually give ratings that are a lot different from historical ratings."}
# activity_dict
        if self.is_restaurant_domain:
            diversity_dict = {1:"An Exceedingly Discerning Selective Diner who tries restaurants with a very high level of selectivity. Dining choices are carefully curated to match personal taste, leaving little room for variety.",
                              2:"A Niche Dining Explorer who occasionally explores different cuisines or restaurant types but mostly sticks to preferred dining patterns.",
                              3:"A Culinary Trailblazer, a seeker of unique, unfamiliar, and varied restaurant experiences. Dining choices are diverse and adventurous."}
        else:
            diversity_dict = {1:"An Exceedingly Discerning Selective Viewer who watches movies with a level of selectivity that borders on exclusivity. The movie choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
                              2:"A Niche Explorer who occasionally explores different genres and mostly sticks to preferred movie types.",
                              3:"A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization."}
        
        self.conformity_group = init_statistic["conformity"]
        self.activity_group = init_statistic["activity"]
        self.diversity_group = init_statistic["diversity"]
        self.conformity_dsc = conformity_dict[self.conformity_group]
        self.activity_dsc = activity_dict[self.activity_group]
        self.diversity_dsc = diversity_dict[self.diversity_group]

    def init_memory(self):
        """
        Initialize the memory of the avatar
        """
        t1 = time.time()
        def score_normalizer(val: float) -> float:
            return 1 - 1 / (1 + np.exp(val))
        
        embedding_provider = getattr(self.args, "sim_embedding_provider", "local")
        if embedding_provider == "gemini":
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            embedding_size = 3072
        elif embedding_provider == "huggingface":
            embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            embedding_size = 384
        else:
            embeddings_model = HashEmbeddings(size=384)
            embedding_size = 384
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=score_normalizer)

        llm_provider = getattr(self.args, "sim_llm_provider", "groq")
        llm_model = getattr(self.args, "sim_llm_model", None)
        llm_max_tokens = getattr(self.args, "sim_llm_max_tokens", 400)
        if llm_provider == "gemini":
            LLM = ChatGoogleGenerativeAI(
                model=llm_model or "gemini-3.1-flash-lite",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7,
                convert_system_message_to_human=True
            )
        elif llm_provider == "groq":
            LLM = ChatOpenAI(
                model_name=llm_model or "llama-3.1-8b-instant",
                openai_api_key=_env_first("GROQ_API_KEY"),
                openai_api_base="https://api.groq.com/openai/v1",
                max_tokens=llm_max_tokens,
                temperature=0.3,
                request_timeout=30
            )
        elif llm_provider == "openai":
            LLM = ChatOpenAI(
                model_name=llm_model or "gpt-4o-mini",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=llm_max_tokens,
                temperature=0.3,
                request_timeout=30
            )
        else:
            LLM = ChatOpenAI(
                model_name=llm_model or "kimi-k2.6",
                openai_api_key=_env_first("KIMI_API_KEY", "MOONSHOT_API_KEY"),
                openai_api_base="https://api.moonshot.ai/v1",
                max_tokens=llm_max_tokens,
                temperature=0.3,
                request_timeout=30
            )
        avatar_retriever = AvatarRetriver(vectorstore=vectorstore, k=5)
        self.memory = AvatarMemory(
            memory_retriever=avatar_retriever,
            llm=LLM,
            reflection_threshold=None,
            use_wandb=self.use_wandb,
            ethnic_group=self.ethnic_group,
            is_restaurant_domain=self.is_restaurant_domain,
        )
        self._initialize_memory_from_history()
        t2 = time.time()

        
        cprint(f"Avatar {self.avatar_id} is initialized with memory", color='green', attrs=['bold'])
        cprint(f"Time cost: {t2-t1}s", color='green', attrs=['bold'])



    # def _reaction(self, messages=None, timeout=30):
    #     """
    #     Summarize the feelings of the avatar for recommended item list.
    #     """ 
    #     response = ''
    #     except_waiting_time = 1
    #     max_waiting_time = 16
    #     current_sleep_time = 0.5
    #     while response == '':
    #         try:
    #             start_time = time.time()
    #             time_local = time.localtime(start_time)
    #             l_start = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

    #             if(self.use_wandb): # whether to use wandb
    #                 if((start_time - vars.global_start_time)//vars.global_interval > vars.global_steps):
    #                     print("\nStart Identifier", start_time, vars.global_start_time, (start_time - vars.global_start_time), vars.global_steps)
    #                     if(vars.lock.acquire(False)):
    #                         print("\nStart Identifier", start_time, vars.global_start_time, (start_time - vars.global_start_time), vars.global_steps)
    #                         vars.global_steps += 1
    #                         wandb.log(
    #                             data = {"Real-time Traffic": vars.global_k_tokens - vars.global_last_tokens_record,
    #                                     "Total Traffic": vars.global_k_tokens,
    #                                     "Finished Users": vars.global_finished_users,
    #                                     "Finished Pages": vars.global_finished_pages,
    #                                     "Error Cast": vars.global_error_cast/1000,
    #                             },
    #                             step = vars.global_steps
    #                         )
    #                         vars.global_last_tokens_record = vars.global_k_tokens
    #                         vars.lock.release()
    #                         print("\nEnd Identifier", time.time(), vars.global_start_time, (time.time() - vars.global_start_time), vars.global_steps)
                            
    #             # completion = openai.ChatCompletion.create(
    #             #     model="gpt-3.5-turbo", 
    #             #     messages=messages,
    #             #     temperature=0.2,
    #             #     request_timeout = timeout,
    #             #     max_tokens=1000
    #             #     )

    #             # l_end = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    #             # k_tokens = completion["usage"]["total_tokens"]/1000
    #             # print(f"User {self.avatar_id} used {k_tokens} tokens from {l_start} to {l_end}")
    #             # self.memory.user_k_tokens += k_tokens
    #             # vars.global_k_tokens += k_tokens
    #             # response = completion["choices"][0]["message"]["content"]
    #             gemini_messages = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    #             completion = gemini_client.models.generate_content(
    #                     model="gemini-3.1-flash-lite",
    #                     contents=gemini_messages,
    #                     config= types.GenerationConfig(
    #                         temperature=0.2,
    #                         max_output_tokens=1000
    #                     )
    #                 )
    #             response = completion.text
    #         except Exception as e:
    #             print(e)
    #             vars.global_error_cast += 1
    #             time.sleep(current_sleep_time)
    #             if except_waiting_time < max_waiting_time:
    #                 except_waiting_time *= 2
    #             current_sleep_time = np.random.randint(0, except_waiting_time-1)
                
    #     return response
    
    def _reaction(self, messages=None, timeout=30):
            response = ''
            while response == '':
                try:
                    global _last_llm_call_at
                    min_interval = getattr(self.args, "sim_llm_min_interval", 16.0)
                    if min_interval > 0:
                        with _llm_rate_lock:
                            wait_time = min_interval - (time.time() - _last_llm_call_at)
                            if wait_time > 0:
                                time.sleep(wait_time)
                            _last_llm_call_at = time.time()
                    from langchain.schema import HumanMessage, SystemMessage
                    lc_messages = []
                    for m in messages:
                        if m['role'] == 'system':
                            lc_messages.append(SystemMessage(content=m['content']))
                        else:
                            lc_messages.append(HumanMessage(content=m['content']))

                    raw_response = self.memory.llm.invoke(lc_messages).content
                    response = _strip_thinking(raw_response) or str(raw_response or "").strip()

                except Exception as e:
                    print(e)  # keep this uncommented so you can see errors
                    vars.global_error_cast += 1
                    time.sleep(1)
            return response

    def _parse_recommended_items(self, recommended_items_str):
        items = []
        pattern = re.compile(
            r"<-\s*(.*?)\s*->\s*<-\s*History ratings:\s*([0-9.]+)\s*->\s*<-\s*Summary:\s*(.*?)\s*->",
            re.DOTALL,
        )
        for title, history_rating, summary in pattern.findall(recommended_items_str):
            try:
                rating = float(history_rating)
            except ValueError:
                rating = 3.0
            categories_match = re.search(r"categories including\s+(.*?)\.\s+It has", summary, flags=re.IGNORECASE)
            reviews_match = re.search(r"([0-9,]+)\s+reviews", summary, flags=re.IGNORECASE)
            price_match = re.search(r"price range\s+([0-9?]+)", summary, flags=re.IGNORECASE)
            categories = []
            if categories_match:
                categories = [part.strip() for part in categories_match.group(1).split(",") if part.strip()]
            items.append({
                "title": title.strip(),
                "history_rating": rating,
                "summary": re.sub(r"\s+", " ", summary).strip(),
                "categories": categories,
                "review_count": int(reviews_match.group(1).replace(",", "")) if reviews_match else 0,
                "price_range": price_match.group(1) if price_match else "?",
            })
        if items:
            return items
        for line in recommended_items_str.splitlines():
            if line.strip():
                title = line.strip().strip("<>- ")
                items.append({"title": title, "history_rating": 3.5, "summary": line.strip(), "categories": [], "review_count": 0, "price_range": "?"})
        return items

    def _perception_caption(self, item):
        if not self.is_restaurant_domain:
            return ""
        categories = ", ".join(item.get("categories", [])[:5]) or "general restaurant"
        price = item.get("price_range", "?")
        reviews = item.get("review_count", 0)
        rating = item.get("history_rating", 0)
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()

        social_proof = "strong local/social proof" if reviews >= 1000 or rating >= 4.5 else "moderate social proof"
        if reviews < 100:
            social_proof = "limited social proof"

        atmosphere = []
        if any(token in text for token in ["bars", "nightlife", "cocktail", "wine", "beer", "whiskey", "sports bar"]):
            atmosphere.append("nightlife/alcohol-forward")
        if any(token in text for token in ["breakfast", "brunch", "cafes", "coffee", "bakery"]):
            atmosphere.append("casual morning/brunch")
        if any(token in text for token in ["barbeque", "burgers", "sandwiches", "cheesesteaks", "buffets", "soup"]):
            atmosphere.append("hearty/filling")
        if any(token in text for token in ["vegan", "vegetarian", "health", "salad"]):
            atmosphere.append("health-conscious")
        if not atmosphere:
            atmosphere.append("standard dining")

        external_influence = []
        if price == "1":
            external_influence.append("budget-friendly signal")
        elif price in {"3", "4"}:
            external_influence.append("premium-price risk")
        if rating >= 4.5:
            external_influence.append("high crowd approval")
        if "french" in text or "wine" in text:
            external_influence.append("possible pretentious/upscale perception")
        if self.ethnic_group == "hausa" and any(token in text for token in ["pork", "wine", "cocktail", "beer", "whiskey", "bar "]):
            external_influence.append("possible halal or alcohol-environment concern")
        if self.ethnic_group in {"yoruba", "igbo", "nigerian"} and any(token in text for token in ["mexican", "szechuan", "indian", "malaysian", "thai", "barbeque"]):
            external_influence.append("bold/spicy flavor cue")

        return (
            f"{item['title']}: visual/social cue proxy from metadata -> categories={categories}; "
            f"atmosphere={', '.join(atmosphere)}; price={price}; rating={rating}; reviews={reviews}; "
            f"{social_proof}; external_influence={', '.join(external_influence) if external_influence else 'neutral'}"
        )

    def _page_perception_context(self, recommended_items_str):
        items = self._parse_recommended_items(recommended_items_str)
        captions = [self._perception_caption(item) for item in items]
        return "\n".join(caption for caption in captions if caption)

    def _kg_evidence_for_items(self, recommended_items_str, top_k=3):
        items = self._parse_recommended_items(recommended_items_str)
        evidence_lines = []
        profile_terms = self.liked_items + self.frequent_categories + self.evidence_items
        dislike_terms = self.disliked_items
        for item in items:
            text = f"{item['title']} {item['summary']}".lower()
            support = [term for term in profile_terms if str(term).lower() in text][:top_k]
            conflict = [term for term in dislike_terms if str(term).lower() in text][:top_k]
            paths = []
            paths.extend(f"user -> liked_history/category -> {term} -> similar_to -> {item['title']}" for term in support)
            paths.extend(f"user -> disliked_history -> {term} -> conflicts_with -> {item['title']}" for term in conflict)
            if item.get("price_range") in {"3", "4"} and self.price_sensitivity.lower() == "high":
                paths.append(f"user -> price_sensitivity -> high -> conflicts_with_price -> {item['title']}")
            if item.get("history_rating", 0) >= 4.5:
                paths.append(f"community -> high_rating/social_proof -> {item['title']}")
            if paths:
                evidence_lines.append(f"{item['title']}: " + " | ".join(paths[:4]))
        if not evidence_lines and self.knowledge_graph_context:
            evidence_lines.append("General user KG evidence: " + self.knowledge_graph_context)
        return "\n".join(evidence_lines[:8])

    def _keyword_score(self, text, keywords, weight):
        score = 0.0
        for keyword in keywords:
            key = str(keyword).lower().strip()
            if key and key in text:
                score += weight
        return score

    def _fallback_recommended_reaction(self, recommended_items_str, current_page):
        """Deterministic backup when the LLM output is not parseable."""
        items = self._parse_recommended_items(recommended_items_str)
        scored = []
        for item in items:
            text = f"{item['title']} {item['summary']}".lower()
            score = 0.0
            score += self._keyword_score(text, self.liked_items, 2.0)
            score += self._keyword_score(text, self.frequent_categories, 1.0)
            score += self._keyword_score(text, self.evidence_items, 1.5)
            score -= self._keyword_score(text, self.disliked_items, 3.0)
            if self.conformity_group == 1:
                score += max(-1.0, min(1.0, item["history_rating"] - 3.5))
            elif self.conformity_group == 2:
                score += 0.5 * max(-1.0, min(1.0, item["history_rating"] - 3.5))
            if self.price_sensitivity.lower() == "high" and "price range 4" in text:
                score -= 1.5
            if self.price_sensitivity.lower() in {"high", "medium"} and "price range 1" in text:
                score += 0.5
            scored.append((score, item))

        lines = [f"CONTEXT: {self._context_layer(current_page)}; BASIS: deterministic fallback after unparseable LLM output"]
        aligned = []
        for score, item in scored:
            align = score >= 1.0 or (score >= 0.0 and item["history_rating"] >= 4.4)
            if align:
                aligned.append((score, item))
            reason = "matches persona evidence, categories, value, or historical quality" if align else "does not strongly match the persona evidence or current context"
            lines.append(f"MOVIE: {item['title']}; ALIGN: {'yes' if align else 'no'}; REASON: {reason}")

        max_choices = 1 if self.activity_group in {1, 2} else 2
        chosen = [item for _, item in sorted(aligned, key=lambda pair: pair[0], reverse=True)[:max_choices]]
        lines.append(f"NUM: {len(chosen)}; WATCH: {', '.join(item['title'] for item in chosen)}; REASON: chose only restaurants with enough persona/context alignment;")
        for item in chosen:
            rating = _clamp_rating(4.0 + (0.5 if item["history_rating"] >= 4.4 else 0.0))
            lines.append(f"MOVIE: {item['title']}; RATING: {rating}; FEELING: This feels like a reasonable match for my restaurant taste and current context;")
        return "\n".join(lines)
    
    def make_next_decision(self, remember=False, current_page=None):
        observation = f"Are you satisfied with the current {self.recommender_domain}, and what is your interaction history?"
        relevant_memories = self.memory.fetch_memories(observation)
        formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
        page_count = current_page or 1
        fatigue_hint = "low" if page_count <= 2 else "medium" if page_count <= 4 else "high"
        sys_prompt = (f"You excel at role-playing. Picture yourself as a user exploring a {self.recommender_domain}. You have the following social traits: "
                    +f"\nYour activity trait is described as: {self.activity_dsc}"
                    +(f"\n{self._profile_context_block(current_page)}" if self._profile_context_block(current_page) else "")
                    +f"\nNow you are in Page {current_page}. You may get tired with the increase of the pages you have browsed. (above 2 pages is a little bit tired, above 4 pages is very tired)"
                    +f"\nRelevant context from your memory:"
                    +f"\n{formated_relevant_memories}"
                    )
        prompt = ("/no_think\n"
                +"Output only the final decision. Do not include analysis, reasoning blocks, markdown, or <think> tags."
                +f"\nBrain module step 1: estimate satisfaction from previous pages as low/medium/high."
                +f"\nBrain module step 2: estimate fatigue. Current fatigue prior is {fatigue_hint}."
                +"\nBrain module step 3: infer current emotion from the interaction history, such as excited, curious, frustrated, tired, or budget-conscious."
                +"\nBrain module step 4: perform a short causal action check: would exiting now increase satisfaction, or would one more page likely reveal a better match?"
                +"\nNow decide whether to continue browsing or exit based on satisfaction, fatigue, emotion, persona, and causal check."
                +"\nIf minimum browse pages have already been met, selective diners may exit after enough poor matches. Curious or high-engagement diners may continue after mixed results."
                +"\nTo leave, write: [EXIT]; Reason: [brief reason]"
                +"\nTo continue browsing, write: [NEXT]; Reason: [brief reason]"
            )
        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]
        
        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        response = self._reaction(messages)
        self.write_log("\n" + response, color="white")

        return response
    
    def response_to_question(self, question, remember=False):
        relevant_memories = self.memory.memory_retriever.memory_stream
        formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
        sys_prompt = (f"You excel at role-playing. Picture yourself as user {self.avatar_id} who has just finished exploring a {self.recommender_domain}. You have the following social traits:"
                +f"\nYour activity trait is described as: {self.activity_dsc}"
                +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                +(f"\n{self._profile_context_block()}" if self._profile_context_block() else "")
                +f"\nBeyond that, your {self.taste_domain} are: {'; '.join(self.taste).replace('I ','')}. "
                +f"\nThe activity characteristic pertains to the frequency of your {self.item_kind} recommendation habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of {self.action_gerund} {self.item_plural} that may not align with your usual taste."
                )
        prompt = f"""
        /no_think
        Output only the final answer. Do not include analysis, markdown, or <think> tags.
        Relevant context from user {self.avatar_id}'s memory:
        {formated_relevant_memories}
        Act as user {self.avatar_id}, assume you are having a interview, reponse the following question:
        {question}
        Base your answer only on the interaction history and restaurant names shown in the relevant context.
        Do not invent restaurants, pages, or future recommendations that are not in the context.
        If the experiment ended after one page, do not say you will evaluate page 2.
        Keep the response concise and complete.
        """


        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]
        
        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        response = self._reaction(messages)
        self.write_log("\n" + response, color="blue")
        # 
        if(remember):
            self.memory.add_memory(f"I was asked '{question}', and I responsed: '{response}'"
                                , now=datetime.datetime.now())
        return response
    
    def reaction_to_forced_items(self, recommended_items_str):
        """
        Summarize the feelings of the avatar for recommended item list.
        """

        sys_prompt = (f"Assume you are a user browsing a {self.recommender_domain} who has the following characteristics: "
                +(f"\n{self._profile_context_block()}" if self._profile_context_block() else "")
                +f"\nYour {self.taste_domain} are: {'; '.join(self.taste).replace('I ','')}. ")
        prompt = (
                "##recommended list## \n" 
                +recommended_items_str
                +f"\nPlease choose {self.item_plural} in the ##recommended list## that you want to {self.action_verb} and explain why. After {self.action_gerund} the {self.item_kind}, evaluate each {self.item_kind} based on your characteristics, taste and historical ratings to give a rating from 1 to 5."
                +f"\nYou only {self.action_verb} {self.item_plural} which align with your taste."
                +f"\nUse this format: MOVIE: [{self.item_kind} name]; WATCH: [yes or no]; REASON: [brief reason]"
                +f"\nThe literal field label MOVIE is required for parser compatibility; in this run it means {self.item_kind} name."
                +f"\nYou must judge all the {self.item_plural}. If you don't want to {self.action_verb} a {self.item_kind}, use WATCH: no; REASON: [brief reason]"
                +"\nEach response should be on one line. Do not include any additional information or explanations and stay grounded in reality."
        )
        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]

        reaction = self._reaction(messages, timeout=60)

        return reaction
    
    def reaction_to_recommended_items(self, recommended_items_str, current_page):
        """
        Summarize the feelings of the avatar for recommended item list.
        """ 
        try:
            high_rating = self.high_rating.replace('You are','')
        except:
            high_rating = ''

        sys_prompt = (f"You excel at role-playing. Picture yourself as a user exploring a {self.recommender_domain}. You have the following social traits:"
                +f"\nYour activity trait is described as: {self.activity_dsc}"
                +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                +(f"\n{self._profile_context_block(current_page)}" if self._profile_context_block(current_page) else "")
                +f"\nBeyond that, your {self.taste_domain} are: {'; '.join(self.taste).replace('I ','')}. "
                +f"\nAnd your rating tendency is {high_rating}"#+f"{low_rating}"
                +f"\nThe activity characteristic pertains to the frequency of your {self.item_kind} recommendation habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of {self.action_gerund} {self.item_plural} that may not align with your usual taste."
                )
        if self.memory.memory_retriever.memory_stream:
            observation = f"What {self.item_plural} have you interacted with on the previous pages of the current recommender system?"
            relevant_memories = self.memory.fetch_memories(observation)
            formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
            sys_prompt = sys_prompt +f"\nRelevant context from your memory:{formated_relevant_memories}"
        perception_context = self._page_perception_context(recommended_items_str)
        kg_evidence_context = self._kg_evidence_for_items(recommended_items_str)

        prompt = (
                "/no_think\n"
                +"Output only parseable lines in the requested format. Do not include analysis, markdown, numbered lists, explanations outside the fields, or <think> tags.\n"
                +"#### Recommended List #### \n"
                + f"PAGE {current_page}\n"
                +recommended_items_str
                +f"\n#### Perception Module: visual/social cue proxies ####\n{perception_context}\n"
                +f"\n#### KG Memory Retrieval: similar paths and external influence ####\n{kg_evidence_context}\n"
                +f"\nBrain module procedure: form an initial decision using persona, pickiness, episodic memory, KG evidence, perception cues, and current context. Then check for contradictions. Examples: do not choose an alcohol-heavy venue for a halal-conscious user without a strong reason; do not choose premium-price venues for a highly price-sensitive user unless evidence strongly supports it; do not reject a strong bold-flavor/value match only because it is unfamiliar."
                +f"\nWrite one short context line using this exact format:"
                +f"\nCONTEXT: {self._context_layer(current_page)}; BASIS: [brief persona/perception/KG evidence used]"
                +f"\nThen judge every {self.item_kind} using this exact format:"
                +f"\nMOVIE: [{self.item_kind} name]; ALIGN: [yes or no]; REASON: [brief reason]"
                +f"\nThe literal field label MOVIE is required for parser compatibility; in this Yelp run it means {self.item_kind} name. Do not talk about films or movies."
                +f"\nAfter the ALIGN lines, choose from only the aligned {self.item_plural}. Use this exact format:"
                +f"\nNUM: [number of {self.item_plural} you choose to {self.action_verb}]; WATCH: [all {self.item_kind} names you choose to {self.action_verb}]; REASON: [brief reason];"
                +f"\nFor each chosen {self.item_kind}, rate it from 1 to 5. Use this exact format:"
                +f"\nMOVIE: [{self.item_kind} you choose to {self.action_verb}]; RATING: [integer between 1-5]; FEELING: [aftermath sentence];"
                +"\nEvery ALIGN and RATING line must begin with MOVIE:"
        )

        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]
        
        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        reaction = self._reaction(messages, timeout=60) # reaction
        self.write_log("\n" + reaction, color="yellow")

        # @ 2 Add user satisfaction information for this page.

        # =========================
        pattern1 = re.compile(r'(?:MOVIE:\s*)?(.+?);\s*RATING:\s*(\d+(?:\.\d+)?);\s*FEELING:\s*(.*)')
        match1 = pattern1.findall(reaction)
        pattern2 = re.compile(r'(?:MOVIE:\s*)?(.+?);\s*ALIGN:\s*(.+?);\s*REASON:\s*(.*)')
        match2 = pattern2.findall(reaction)
        if not match2:
            fallback_reaction = self._fallback_recommended_reaction(recommended_items_str, current_page)
            self.write_log("\n[FALLBACK AFTER UNPARSEABLE LLM OUTPUT]\n" + fallback_reaction, color="red")
            reaction = fallback_reaction
            match1 = pattern1.findall(reaction)
            match2 = pattern2.findall(reaction)
        all_movies = ", ".join([movie_title.strip(';') for movie_title, align, reason in match2])
        watched_movies = [movie_title.strip(';') for movie_title, rating, feeling in match1]
        watched_movies_ratings = [rating.strip(';') for movie_title, rating, feeling in match1]
        like_movies = [movie_title.strip(';') for movie_title, rating, feeling in match1 if _clamp_rating(rating.strip(';')) == 5]
        dislike_movies = [movie_title.strip(';') for movie_title, rating, feeling in match1 if (_clamp_rating(rating.strip(';')) < 4)]
        dislike_movies.extend([movie_title.strip(';') for movie_title, align, reason in match2 if align.strip(';').lower() == 'no'])
        for movie_title, rating, feeling in match1:
            rating_value = _clamp_rating(rating.strip(';'))
            if rating_value >= 4:
                self._append_kg_edge("user", "sim_liked", movie_title.strip(';'))
            elif rating_value <= 2:
                self._append_kg_edge("user", "sim_disliked", movie_title.strip(';'))
            self.memory.add_memory(
                f"I rated {movie_title.strip(';')} {rating_value}/5 and felt: {feeling.strip(';')}",
                now=datetime.datetime.now(),
            )
        for movie_title, align, reason in match2:
            if align.strip(';').lower() == 'no':
                self._append_kg_edge("user", "sim_rejected", movie_title.strip(';'))
        self.memory.add_memory(f"The recommender recommended the following {self.item_plural} to me on page {current_page}: {all_movies}, among them, I chose {watched_movies} and rate them {watched_movies_ratings} respectively. I dislike the rest {self.item_plural}: {dislike_movies}."
            , now=datetime.datetime.now()
        )

        if current_page >= getattr(self.args, "max_pages", current_page):
            self.exit_flag = True
            self.memory.add_memory(f"After browsing {current_page} page(s), the experiment ended because it reached the maximum page limit."
                , now=datetime.datetime.now())
            return reaction

        min_browse_pages = min(
            getattr(self.args, "min_browse_pages", 1),
            getattr(self.args, "max_pages", current_page),
        )
        if current_page < min_browse_pages:
            self.exit_flag = False
            self.memory.add_memory(
                f"The experiment requires at least {min_browse_pages} page(s), so I continue to page {current_page+1} even if my first impression is strong.",
                now=datetime.datetime.now(),
            )
            return reaction

        # User makes the next decision.
        next_decision = self.make_next_decision(current_page=current_page)
        if('[EXIT]' in next_decision or '[exit]' in next_decision):
            self.exit_flag = True
            self.memory.add_memory(f"After browsing {current_page} pages, I decided to leave the recommendation system."
                , now=datetime.datetime.now())
        
        else:
            self.memory.add_memory(f"Turn to page {current_page+1} of the recommendation."
                , now=datetime.datetime.now())
        #===========================

        return reaction

    def write_log(self, log, color=None, attrs=None, print=False):
        with open(self.log_file, 'a') as f:
            f.write(log + '\n')
            f.flush()
        if(print):
            cprint(log, color=color, attrs=attrs)
