import argparse
import pandas as pd
from pathlib import Path
import subprocess
import os
import shutil
import json
import datetime
import numpy as np

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


# Helper Functions
def save_temp_dataset(dataset: list[dict], inputs_addr: str):
    """
    Saves same format json to the same directory as the input json. This temporary file gets deleted after the program executes.

    Arguments:
        - dataset: List of dictionaries, json.
        - inputs_addr: String path of the input json file
    """

    input_path = Path(inputs_addr)
    current_time = str(datetime.datetime.now())
    temp_path = input_path.parent / f"TEMP_{current_time}_{input_path.stem}.json"

    with open(temp_path, "w") as f:
        json.dump(dataset, f, indent=4) # same formatted

    return temp_path


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


def stream_subprocess(process):
    for line in process.stdout:
        print(line, end="")


# Keeping same argument set with additionals to define filtering rules
parser = argparse.ArgumentParser(
    prog="Asymmetric Personalization Experiment Runner",
    description="The suite will run filter profiles and run existing LaMP-QA.",
)

# Asymmetric specific parameters
# parser.add_argument(
#     "--exp_name", type=str, default="exp1", required=True
# )

parser.add_argument("--limit_target", type=int, default=10)
parser.add_argument(
    "--keep_domains", type=str, default=""
)  # A comma separated string ex: ae,lp OR lp,sc,ae OR ae
parser.add_argument(
    "--entropy_minimum", type=float, default=0.0
)  # Minimum required entropy of a user to be included
parser.add_argument("--entropy_percentile", type=float, default=0) # Minimum percentile value to filter users

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
parser.add_argument("--clean_cache", action="store_true")

# Example use: python asymmetric_baselines.py --cache_dir /work/pi_hzamani_umass_edu/ozel_cache/ --model_addr Qwen/Qwen2.5-7B-Instruct --keep_domains ae,lp,sc --inputs_addr data/processed/ae_processed_test.json --output_addr data/processed/out/rag/full_profile/ae_rag_full_profile_test_output.json 

# OpenAI support will come later!
# parser.add_argument("--openai", action="store_true")
# parser.add_argument("--api_key_addr", type=str, default="")

# Main Suite
if __name__ == "__main__":
    args = parser.parse_args()

    # os.makedirs(Path("data") / args.exp_name)

    if args.clean_cache:
        print("Cleaning cache")
        shutil.rmtree(args.cache_dir)

    # Process the requested comma separated string
    kept_domains = args.keep_domains.split(",") if args.keep_domains else []
    print(f"Keeping {kept_domains} in profile")

    if any(domain not in sub_cats.keys() for domain in kept_domains):
        raise ValueError(
            f"Received unrecognized domain(s): {kept_domains}, only {list(sub_cats.keys())} allowed."
        )

    # Read input json and apply filtering rules
    with open(args.inputs_addr, "r") as f:
        dataset = json.load(f)
    
    if args.entropy_minimum > 0 and args.entropy_percentile > 0:
        raise ValueError("Only set percentile or minimum value for entropy not both.")

    if args.entropy_minimum > 0:
        entropy_limit = args.entropy_minimum
    elif args.entropy_percentile > 0:
        entropy_limit = np.percentile([data.get("entropy", 0) for data in dataset], args.entropy_percentile)
    else:
        entropy_limit = 0

    dataset = [item for item in dataset if item.get("entropy", 0) >= entropy_limit]

    temp_save_path = save_temp_dataset(
        filter_profiles(dataset, kept_domains, sub_cats), args.inputs_addr
    )

    print(f"Temp data file saved at {temp_save_path}.")

    # Continued
    # call retrieval/rank_dataset.py
    if args.rag:
        command = [
            "python",
            "retrieval/rank_dataset.py",
            "--model_name",
            args.retrieval_model_name,
            "--input_dataset_addr",
            str(temp_save_path),
            "--output_dataset_addr",
            str(temp_save_path),
            "--batch_size",
            str(args.batch_size),
        ]
        print(f"\tRunning Ranking: {command}")
        rank_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        stream_subprocess(rank_process)
        rank_process.wait()  # wait, sequential execution
    
        with open(temp_save_path, "r") as f:
            dataset = json.load(f)
        file_name = Path(args.inputs_addr).name
        target_cat = file_name[:2]
        dataset = filter_target(dataset, sub_cats[target_cat], args.limit_target)
        with open(temp_save_path, "w") as f:
            json.dump(dataset, f)

    # call baselines.py
    command = [
        "python",
        "baselines.py",
        "--model_addr",
        args.model_addr,
        "--inputs_addr",
        str(temp_save_path),
        "--output_addr",
        args.output_addr,
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_tokens",
        str(args.max_tokens),
        "--num_generated_outputs",
        str(args.num_generated_outputs),
        "--num_contexts",
        str(args.num_contexts),
        "--max_retries",
        str(args.max_retries),
        "--cache_dir",
        args.cache_dir,
    ]

    if args.rag:
        command.append("--rag")

    print(f"\tRunning baseline: {command}")
    baseline_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stream_subprocess(baseline_process)
    baseline_process.wait()
    print(f"Experiment complete! Output can be found at {args.output_addr}")

    # Clean Up
    os.remove(temp_save_path)
    print("Cleaned temporary files.")
