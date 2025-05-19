import argparse
import pandas as pd
import json
import tiktoken
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix


#needs a prompt for each combination of entries in instance
#assigns score in [0,1]
class MatchingRanker:
    def rank(self, instance, dry_run=False):
        anchor = instance["anchor"]
        prompts = []
        scores = []
        for c in instance["candidates"]:
            prompt = f"Rate match:\nAnchor: {anchor}\nCandidate: {c}\nReturn score 0-1."
            prompts.append(prompt)
            scores.append(0.5 if dry_run else self.query(prompt))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked, prompts

    def query(self, prompt):
        return 0.5  # Replace with real LLM call

#needs one prompt per instance
#returns topk from sorted list by LLM
class ComparingRanker:
    def rank(self, instance, topK, dry_run=False):
        anchor = instance["anchor"]
        candidates = instance["candidates"]
        prompt = f"Compare candidates for anchor: {anchor}\n" + \
                 "\n".join([f"{i}: {c}" for i, c in enumerate(candidates)]) + \
                 f"\nReturn top-{topK} indices."
        return list(range(min(topK, len(candidates)))), [prompt]


#final decision after ranking 
#bool
class BinarySelector:
    def select(self, anchor, candidates, dry_run=False):
        prompts = []
        results = []
        for c in candidates:
            prompt = f"Is this a match?\nAnchor: {anchor}\nCandidate: {c}\nAnswer True or False."
            prompts.append(prompt)
            results.append(False if dry_run else self.query(prompt))
        return results, prompts

    def query(self, prompt):
        #TODO: Make LLM call
        return False


def classify_prompt_type(prompt):
    if "Return score 0-1" in prompt:
        return "rank"
    elif "Answer True or False" in prompt:
        return "select"
    return "unknown"

def analyze_prompts(prompts, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    price_per_token = {
        "gpt-3.5-turbo": 0.0000015,
        "gpt-4o": 0.000005,
        "gpt-4": 0.000030
    }.get(model, 0.0000015)

    records = []
    for i, p in enumerate(prompts):
        tokens = len(enc.encode(p))
        records.append({
            "id": i,
            "type": classify_prompt_type(p),
            "token_count": tokens,
            "prompt": p,
            "token_cost_usd": round(tokens * price_per_token, 6)
        })
    return records

def save_dry_run(prompts, model, path="dry_run_dump.json"):
    data = analyze_prompts(prompts, model=model)
    summary = {
        "model": model,
        "total_prompts": len(data),
        "total_tokens": sum(p["token_count"] for p in data),
        "total_cost_usd": round(sum(p["token_cost_usd"] for p in data), 6),
        "rank_prompts": sum(p["type"] == "rank" for p in data),
        "select_prompts": sum(p["type"] == "select" for p in data)
    }
    result = {"prompts": data, "summary": summary}
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Dry run dump saved to: {path}")



def comem_call(instance, topK, strategy, dry_run=False):
    ranker = MatchingRanker() if strategy == "matching" else ComparingRanker()
    selector = BinarySelector()
    
    #apply ranknig strategy
    if strategy == "matching":
        indexes, rank_prompts = ranker.rank(instance, dry_run=dry_run)
    else:
        indexes, rank_prompts = ranker.rank(instance, topK, dry_run=dry_run)

    indexes_k = indexes[:topK]
    instance_k = {
        "anchor": instance["anchor"],
        "candidates": [instance["candidates"][i] for i in indexes_k]
    }

    preds = [False] * len(instance["candidates"])
    preds_k, select_prompts = selector.select(instance_k["anchor"], instance_k["candidates"], dry_run=dry_run)

    for i, pred in enumerate(preds_k):
        preds[indexes_k[i]] = pred

    return preds, rank_prompts + select_prompts



def main(args):
    #load data and group by id to form instances/blocks
    df = pd.read_csv(args.dataset)
    grouped = df.groupby("id_left")[["record_left", "record_right", "label"]]
    instances = [
        {
            "anchor": g["record_left"].iloc[0],
            "candidates": g["record_right"].tolist(),
            "labels": g["label"].tolist()
        }
        for _, g in grouped
    ]

    strategy = "matching" if args.m else "comparing"
    all_preds, all_labels, all_prompts = [], [], []
   
    #call pipeline on each instance
    for inst in instances:
        preds, prompts = comem_call(inst, args.topk, strategy, dry_run=args.dry_run)
        all_preds.extend(preds)
        all_labels.extend(inst["labels"])
        all_prompts.extend(prompts)
    
    if not args.dry_run:
        print(classification_report(all_labels, all_preds, digits=4))
        print(confusion_matrix(all_labels, all_preds))

    if args.dry_run:
        save_dry_run(all_prompts, model=args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("-m", action="store_true", help="Use matching ranker")
    parser.add_argument("-c", action="store_true", help="Use comparing ranker")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM model for pricing/tokenization")
    args = parser.parse_args()

    if args.m == args.c:
        parser.error("choose one ranking algo")

    main(args)

