import argparse
import pandas as pd
from pathlib import Path
import subprocess
import os
import shutil
import json
import datetime
import numpy as np
import tqdm
import random
from transformers import AutoModel, AutoTokenizer
from retrieval.rank_dataset import retrieve_top_k_with_contriver
from baselines import main_process

# Defining subcategories in LaMP-QA corpus
sub_cats = {
    "ae": {
        "anime",
        "boardgames",
        "gaming",
        "literature",
        "movies",
        "music",
        "musicfans",
        "rpg",
        "scifi",
        "sound",
    },
    "lp": {
        "bicycles",
        "cooking",
        "diy",
        "fitness",
        "freelancing",
        "gardening",
        "health",
        "lifehacks",
        "martialarts",
        "outdoors",
        "parenting",
        "pets",
        "sports",
        "sustainability",
        "travel",
        "woodworking",
        "workplace",
        "writers",
    },
    "sc": {
        "academia",
        "buddhism",
        "christianity",
        "english",
        "expatriates",
        "genealogy",
        "hermeneutics",
        "hinduism",
        "history",
        "interpersonal",
        "islam",
        "judaism",
        "law",
        "linguistics",
        "money",
        "philosophy",
        "politics",
        "skeptics",
        "vegetarianism",
    },
}


# # Helper Functions
# def save_temp_dataset(dataset: list[dict], inputs_addr: str):
#     """
#     Saves same format json to the same directory as the input json. This temporary file gets deleted after the program executes.

#     Arguments:
#         - dataset: List of dictionaries, json.
#         - inputs_addr: String path of the input json file
#     """

#     input_path = Path(inputs_addr)
#     current_time = str(datetime.datetime.now())
#     temp_path = input_path.parent / f"TEMP_{current_time}_{input_path.stem}.json"

#     with open(temp_path, "w") as f:
#         json.dump(dataset, f, indent=4) # same formatted

#     return temp_path


def filter_individual_profile(profile: list, kept_categories: set):
    """
    Applies category filtering on a single profile. Any entry in profile that is not in kept_categories will be filtered out.

    Arguments:
        - profile: A list of dictionaries. Each dictionary storing id, category, and question.
        - kept_categories: A set of categories to include in profile.
    """

    return [entry for entry in profile if entry["category"] in kept_categories]


def filter_profiles(dataset: list[dict], kept_domains: list, sub_cats: dict):
    """
    Applies domain filtering to the dataset. It will keep profile entries with subtopics that relate to the kept domains, rest is filtered out.

    Arguments:
        - dataset: List of dictionaries denoting a json array
        - kept_domains: A list of domains (ae, lp, sc)
        - sub_cats: Previously defined dictionary of sets mapping domain and related sub-categories
    """

    # merge categories of kept domains
    kept_categories = set()
    for domain in kept_domains:
        kept_categories.update(sub_cats[domain])

    # apply individual filtering
    for item in dataset:
            item["profile"] = filter_individual_profile(item["profile"], kept_categories)

    return dataset


def filter_target(dataset, target_subcats, filter_size):
    """
    Filters posts of the target domain from each profile in the dataset.

    Arguments:
        - dataset: json array
        - target_subcats: A list of subcategories
        - filter_size: the maximum number of target domain posts to keep in profile
    """

    filtered_dataset = []
    for user in dataset:
        profile = user["profile"]
        count = 0
        new_profile = []
        for item in profile:
            if item["category"] in target_subcats:
                if count < filter_size:
                    new_profile.append(item)
                    count += 1
            else:
                new_profile.append(item)
        user_copy = user.copy()
        user_copy["profile"] = new_profile
        filtered_dataset.append(user_copy)
    return filtered_dataset

# Keeping same argument set with additionals to define filtering rules
parser = argparse.ArgumentParser(
    prog="Asymmetric Personalization Experiment Runner",
    description="The suite will run filter profiles and run existing LaMP-QA.",
)

# Asymmetric specific parameters
parser.add_argument("--limit_target", type=int, default=10) # If you set this to 0 it is asymmetric-cold-start, if you set it to same size as context it is full-profile personalization
parser.add_argument(
    "--entropy_minimum", type=float, default=0.0
)  # Minimum required entropy of a user to be included
parser.add_argument("--entropy_percentile", type=float, default=0) # Minimum percentile value to filter users
parser.add_argument("--public", action="store_true") # enables 2-aug
parser.add_argument("--cat_organized", action="store_true") # enables 2-aug-categorized

# Retrieval parameters
parser.add_argument(
    "--retrieval_model_name", type=str, default="facebook/contriever-msmarco"
)
parser.add_argument("--batch_size", type=int, default=4)

# Baseline parameters
parser.add_argument("--model_addr", type=str, required=True)
parser.add_argument("--inputs_addr", type=str, required=True)
parser.add_argument("--output_addr", type=str, required=True)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", type=int, default=8192)
parser.add_argument("--num_generated_outputs", type=int, default=1)
parser.add_argument("--num_contexts", type=int, default=10)
parser.add_argument("--max_retries", type=int, default=10)
parser.add_argument("--rag", action="store_true")
parser.add_argument("--cache_dir", default="./cache")

# Main Suite
if __name__ == "__main__":
    args = parser.parse_args()

    # Read input json and apply filtering rules
    with open(args.inputs_addr, "r") as f:
        dataset = json.load(f)
    
    for data in dataset:
        data["public_posts"] = []
    
    if args.entropy_minimum > 0 and args.entropy_percentile > 0:
        raise ValueError("Only set percentile or minimum value for entropy not both.")

    if args.entropy_minimum > 0:
        entropy_limit = args.entropy_minimum
    elif args.entropy_percentile > 0:
        entropy_limit = np.percentile([data.get("entropy", 0) for data in dataset], args.entropy_percentile)
    else:
        entropy_limit = 0

    dataset = [item for item in dataset if item.get("entropy", 0) >= entropy_limit]

    file_name = Path(args.inputs_addr).name
    target_cat = file_name[:2] # infers target domain from filename, please follow the same naming

    # Continued, public knowledge retrieval
    tokenizer = AutoTokenizer.from_pretrained(args.retrieval_model_name)
    contriver = AutoModel.from_pretrained(args.retrieval_model_name).to("cuda:0")
    contriver.eval()

    if args.public:
        public_r = 0.1 # percentage of public users
        dataset_new = []
        total_points = len(dataset)
        num_samples = int(total_points * public_r)

        for i, data in tqdm.tqdm(enumerate(dataset)):
            candidates = list(range(total_points))
            candidates.remove(i)
            random_public_indexes = random.sample(candidates, num_samples)
            corpus = []
            for ind in random_public_indexes:
                profile = dataset[ind]["profile"]
                for post in profile: # uses all public assumed users' posts
                    if post["category"] in sub_cats[target_cat]:
                        corpus.append(post)
            
            corpus_text = [obj["text"] for obj in corpus]
            
            ranked_public_posts = retrieve_top_k_with_contriver(contriver, tokenizer, corpus_text, corpus, data["question"], len(corpus), args.batch_size)
            data["public_posts"] = ranked_public_posts[:args.num_contexts]
            dataset_new.append(data)
        dataset = dataset_new
    
    if args.rag:
        dataset_new = []
        for data in tqdm.tqdm(dataset):
            profile = data["profile"]
            corpus = [x["text"] for x in profile]
            query = data["question"]
            ranked_profile = retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, len(profile), args.batch_size)
            ranked_profile = [ranked_profile[i] for i in range(len(ranked_profile))]
            data["profile"] = ranked_profile
            dataset_new.append(data)
        dataset = filter_target(dataset_new, sub_cats[target_cat], args.limit_target)
    
    main_process(dataset_orig=dataset, cache_dir=args.cache_dir, model_addr=args.model_addr, tokenizer=tokenizer, num_contexts=args.num_contexts, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, max_retries=args.max_retries, rag=args.rag, public=args.public, cat_organized=args.cat_organized, output_addr=args.output_addr)
