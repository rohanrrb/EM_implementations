import argparse
import pandas as pd
import json
import os
from openai import OpenAI
import tiktoken
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MatchingRanker:
    def rank(self, instance, dry_run=False):
        anchor = instance["anchor"]
        prompts = []
        scores = []
        responses = []
        for c in instance["candidates"]:
            prompt = f"Rate match:\nAnchor: {anchor}\nCandidate: {c}\nReturn score 0-1."
            prompts.append(prompt)
            if dry_run:
                scores.append(0.5)
                responses.append("0.5")
            else:
                response = self.query(prompt)
                scores.append(float(response))
                responses.append(response)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked, prompts, responses

    def query(self, prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in MatchingRanker.query: {e}")
            return "0.5"


class ComparingRanker:
    def rank(self, instance, topK, dry_run=False):
        anchor = instance["anchor"]
        candidates = instance["candidates"]
        prompt = f"Compare candidates for anchor: {anchor}\n" + \
                 "\n".join([f"{i}: {c}" for i, c in enumerate(candidates)]) + \
                 f"\nReturn top-{topK} indices as a Python list."
        if dry_run:
            return list(range(min(topK, len(candidates)))), [prompt], ["[0, 1, 2, 3]"]
        else:
            response = self.query(prompt)
            try:
                parsed = eval(response)
                return parsed, [prompt], [response]
            except Exception as e:
                print(f"Error parsing ComparingRanker response: {e}")
                return list(range(min(topK, len(candidates)))), [prompt], [response]

    def query(self, prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in ComparingRanker.query: {e}")
            return "[0]"


class BinarySelector:
    def select(self, anchor, candidates, dry_run=False):
        prompts = []
        results = []
        responses = []
        for c in candidates:
            prompt = f"Is this a match?\nAnchor: {anchor}\nCandidate: {c}\nAnswer True or False."
            prompts.append(prompt)
            if dry_run:
                results.append(False)
                responses.append("False")
            else:
                response = self.query(prompt)
                results.append("true" in response.lower())
                responses.append(response)
        return results, prompts, responses

    def query(self, prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in BinarySelector.query: {e}")
            return "False"


def classify_prompt_type(prompt):
    if "Return score 0-1" in prompt:
        return "rank"
    elif "Answer True or False" in prompt:
        return "select"
    elif "Return top-" in prompt and "indices" in prompt:
        return "compare"
    return "unknown"


def analyze_prompts(prompts, responses, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    price_per_token = {
        "gpt-3.5-turbo": 0.0000015,
        "gpt-4o": 0.000005,
        "gpt-4": 0.000030
    }.get(model, 0.0000015)

    records = []
    for i, (p, r) in enumerate(zip(prompts, responses)):
        tokens = len(enc.encode(p))
        records.append({
            "id": i,
            "type": classify_prompt_type(p),
            "token_count": tokens,
            "prompt": p,
            "response": r,
            "token_cost_usd": round(tokens * price_per_token, 6)
        })
    return records


def save_prompt_dump(prompts, responses, model, path="prompt_dump.json"):
    data = analyze_prompts(prompts, responses, model=model)
    summary = {
        "model": model,
        "total_prompts": len(data),
        "total_tokens": sum(p["token_count"] for p in data),
        "total_cost_usd": round(sum(p["token_cost_usd"] for p in data), 6),
        "rank_prompts": sum(p["type"] == "rank" for p in data),
        "select_prompts": sum(p["type"] == "select" for p in data),
        "compare_prompts": sum(p["type"] == "compare" for p in data)
    }
    result = {"prompts": data, "summary": summary}
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Prompt dump saved to: {path}")


def comem_call(instance, topK, strategy, dry_run=False):
    ranker = MatchingRanker() if strategy == "matching" else ComparingRanker()
    selector = BinarySelector()

    if strategy == "matching":
        indexes, rank_prompts, rank_responses = ranker.rank(instance, dry_run=dry_run)
    else:
        indexes, rank_prompts, rank_responses = ranker.rank(instance, topK, dry_run=dry_run)

    indexes_k = indexes[:topK]
    instance_k = {
        "anchor": instance["anchor"],
        "candidates": [instance["candidates"][i] for i in indexes_k]
    }

    preds = [False] * len(instance["candidates"])
    preds_k, select_prompts, select_responses = selector.select(instance_k["anchor"], instance_k["candidates"], dry_run=dry_run)

    for i, pred in enumerate(preds_k):
        preds[indexes_k[i]] = pred

    return preds, rank_prompts + select_prompts, rank_responses + select_responses


def main(args):
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
    all_preds, all_labels, all_prompts, all_responses = [], [], [], []

    for inst in instances:
        preds, prompts, responses = comem_call(inst, args.topk, strategy, dry_run=args.dry_run)
        all_preds.extend(preds)
        all_labels.extend(inst["labels"])
        all_prompts.extend(prompts)
        all_responses.extend(responses)

    if not args.dry_run:
        print(classification_report(all_labels, all_preds, digits=4))
        print(confusion_matrix(all_labels, all_preds))

    save_prompt_dump(all_prompts, all_responses, model=args.model)


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

