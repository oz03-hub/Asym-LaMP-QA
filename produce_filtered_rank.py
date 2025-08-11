import json
from retrieval.rank_dataset import retrieve_top_k_with_contriver
from asymmetric_baselines import sub_cats
from transformers import AutoModel, AutoTokenizer
import argparse
import tqdm


def attach_domain(dataset, sub_cats):
    new_dataset = []
    for data in dataset:
        profile = data["profile"]
        categories = [post["category"] for post in profile]
        domains = []

        for category in categories:
            if category in sub_cats["ae"]:
                domains.append("ae")
            elif category in sub_cats["lp"]:
                domains.append("lp")
            elif category in sub_cats["sc"]:
                domains.append("sc")

        data["domains"] = domains
        new_dataset.append(data)

    return new_dataset


def precision_at_k(ranked_list, relevant_domain, k):
    count = 0
    for domain in ranked_list[:k]:
        if domain == relevant_domain:
            count += 1
    return count / k


def precision_at_r(ranked_list, relevant_domain):
    r_count = sum(1 for dom in ranked_list if dom == relevant_domain)
    return precision_at_k(ranked_list, relevant_domain, r_count)


def recall_at_k(ranked_list, relevant_domain, k):
    count = 0
    actual_count = sum([1 for dom in ranked_list if dom == relevant_domain])
    for domain in ranked_list[:k]:
        if domain == relevant_domain:
            count += 1
    return count / actual_count


def mp_at_k(dataset_new, k, target):
    running_precision = 0
    for i in range(len(dataset_new)):
        running_precision += precision_at_k(dataset_new[i]["domains"], target, k)
    macro_precision = running_precision / len(dataset_new)
    return macro_precision


def mp_at_r(dataset_new, target):
    running_precision = 0
    for i in range(len(dataset_new)):
        running_precision += precision_at_r(dataset_new[i]["domains"], target)
    macro_precision = running_precision / len(dataset_new)
    return macro_precision


def mr_at_k(dataset_new, k, target):
    running_recall = 0
    for i in range(len(dataset_new)):
        running_recall += recall_at_k(dataset_new[i]["domains"], target, k)
    macro_recall = running_recall / len(dataset_new)
    return macro_recall


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--target", required=True, type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    f_path = args.input
    target = args.target

    with open(f_path, "r") as f:
        dataset = json.load(f)
    dataset = dataset

    rank_dict = dict()

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    contriver = AutoModel.from_pretrained("facebook/contriever-msmarco").to("cuda:0")
    contriver.eval()

    dataset_new = []
    for data in tqdm.tqdm(dataset):
        profile = data["profile"]
        corpus = [x["text"] for x in profile]
        query = data["question"]
        ids = [x["id"] for x in profile]

        ranked_profile = retrieve_top_k_with_contriver(
            contriver, tokenizer, corpus, profile, query, len(profile), 2
        )
        # ranked_profile = [{'id': ids[i], 'text': ranked_profile[i]['text']} for i in range(len(ranked_profile))]
        ranked_profile = [ranked_profile[i] for i in range(len(ranked_profile))]
        data["profile"] = ranked_profile
        dataset_new.append(data)

    dataset_new = attach_domain(dataset_new, sub_cats)
    out_str = ""
    for i in [5, 10, 20]:
        out_str += f"{i} rank results\n"
        mp = mp_at_k(dataset_new, i, target)
        mr = mr_at_k(dataset_new, i, target)
        mpr = mp_at_r(dataset_new, target)
        out_str += f"Macro Precision at {i}: {mp:.4f}\n"
        out_str += f"Macro P@R: {mpr:.4f}\n"
        out_str += f"Macro Recall at {i}: {mr:.4f}\n"
        out_str += "\n"

    with open(args.output, "w") as f:
        f.write(out_str)
    print("Finished")
