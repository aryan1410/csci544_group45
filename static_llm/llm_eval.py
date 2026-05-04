import csv
import time
import string
import re
import os
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from datasets import load_dataset

load_dotenv(Path(__file__).parent.parent / ".env")

CUSTOM_QA_FILE = "../data/qa_dataset.txt"

MODEL = "llama-3.3-70b-versatile"
F1_THRESHOLD = 0.2
MAX_QUESTIONS_PER_DS = 50
SAMPLE_SEED = 42
SUBSET_FILE = "../data/questions_subset.json"
OUTPUT_FILE = "eval_results.csv"
SYSTEM_PROMPT = (
    "You are a factual question-answering system. "
    "Reply with only the answer — a single word or short phrase. "
    "Never repeat the subject. Never write a full sentence. No explanation, no punctuation at the end."
)

FEW_SHOT = [
    {"role": "user",      "content": "What is the capital of Japan?"},
    {"role": "assistant", "content": "Tokyo"},
    {"role": "user",      "content": "What is Albert Einstein's occupation?"},
    {"role": "assistant", "content": "physicist"},
    {"role": "user",      "content": "What country is the Eiffel Tower located in?"},
    {"role": "assistant", "content": "France"},
]

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = normalize(prediction).split()
    gt_toks = normalize(ground_truth).split()
    if not pred_toks or not gt_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gt_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gt_toks)
    return (2 * precision * recall) / (precision + recall)


def best_f1(prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(token_f1(prediction, gt) for gt in ground_truths)

def query_model(client: Groq, question: str) -> tuple[str, float]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=64,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT,
            {"role": "user", "content": question},
        ],
    )
    latency = time.perf_counter() - start
    answer = response.choices[0].message.content.strip()
    return answer, latency


def stratified_sample(rows: list[dict], strata_key: str, n: int, rng: random.Random) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get(strata_key, "unknown")].append(row)

    if len(buckets) <= 1:
        return rng.sample(rows, min(n, len(rows)))

    result: list[dict] = []
    total = len(rows)
    allocations = {k: max(1, round(len(v) / total * n)) for k, v in buckets.items()}
    while sum(allocations.values()) > n:
        largest = max(allocations, key=allocations.__getitem__)
        allocations[largest] -= 1
    while sum(allocations.values()) < n:
        for k in sorted(buckets, key=lambda k: len(buckets[k]), reverse=True):
            if allocations[k] < len(buckets[k]):
                allocations[k] += 1
                break

    for k, alloc in allocations.items():
        result.extend(rng.sample(buckets[k], min(alloc, len(buckets[k]))))

    rng.shuffle(result)
    return result

def load_trivia_qa(n: int, rng: random.Random) -> list[dict]:
    print("Loading TriviaQA (rc.nocontext / validation)…")
    ds = load_dataset(
        "trivia_qa", "rc.nocontext", split="validation", trust_remote_code=False
    )
    rows = []
    for item in ds:
        gts = [item["answer"]["value"]] + list(item["answer"].get("aliases", []))
        rows.append({
            "dataset": "TriviaQA",
            "question": item["question"],
            "ground_truths": gts,
            "_source": item.get("question_source", "unknown"),
        })
    print(f"  Full dataset: {len(rows)} questions across "
          f"{len({r['_source'] for r in rows})} sources.")
    sampled = stratified_sample(rows, "_source", n, rng)
    for r in sampled:
        del r["_source"]
    print(f"  Sampled {len(sampled)} questions (stratified by source).")
    return sampled


def load_pop_qa(n: int, rng: random.Random) -> list[dict]:
    print("Loading PopQA…")
    ds = load_dataset("akariasai/popqa", split="test")
    print(f"  Full dataset: {len(ds)} questions.")

    rows = []
    for item in ds:
        raw = item["possible_answers"]
        gts = json.loads(raw) if isinstance(raw, str) else list(raw)
        gts = [str(a) for a in gts if a]
        if not gts:
            continue
        rows.append({
            "dataset": "PopQA",
            "question": item["question"],
            "ground_truths": gts,
            "_source": str(item.get("prop", "unknown")),
        })

    sampled = stratified_sample(rows, "_source", n, rng)
    prop_count = len({r["_source"] for r in sampled})
    for r in sampled:
        del r["_source"]
    print(f"  Sampled {len(sampled)} questions (stratified across {prop_count} property types).")
    return sampled


def load_or_build_subset() -> list[dict]:
    if os.path.exists(SUBSET_FILE):
        with open(SUBSET_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"Loaded {len(questions)} questions from existing {SUBSET_FILE}")
        print("  (Delete this file to re-sample)\n")
        return questions

    rng = random.Random(SAMPLE_SEED)
    questions = []
    questions.extend(load_trivia_qa(MAX_QUESTIONS_PER_DS, rng))
    questions.extend(load_pop_qa(MAX_QUESTIONS_PER_DS, rng))

    with open(SUBSET_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(questions)} questions to {SUBSET_FILE}")
    print("  Share this file with teammates so everyone runs on identical questions.\n")
    return questions

def load_custom_qa(filepath: str) -> list[dict]:
    rows = []
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found — skipping custom questions.")
        return rows
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    q, a = None, None
    for line in lines:
        if re.match(r"Question\s+\d+:", line, re.IGNORECASE):
            q = re.sub(r"^Question\s+\d+:\s*", "", line, flags=re.IGNORECASE)
        elif re.match(r"Answer\s+\d+:", line, re.IGNORECASE):
            a = re.sub(r"^Answer\s+\d+:\s*", "", line, flags=re.IGNORECASE)
        if q and a:
            rows.append({"dataset": "Custom", "question": q, "ground_truths": [a]})
            q, a = None, None
    print(f"Loaded {len(rows)} custom questions from {filepath}.")
    return rows


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GROQ_API_KEY before running.")

    client = Groq(api_key=api_key)

    questions = load_or_build_subset()
    custom_questions = load_custom_qa(CUSTOM_QA_FILE)
    all_questions = questions + custom_questions

    total = len(all_questions)
    print(f"\nEvaluating {total} questions ({len(questions)} subset + {len(custom_questions)} custom) with {MODEL}…")
    print(f"Output → {OUTPUT_FILE}\n")

    fieldnames = ["Dataset", "Question", "Answer", "Ground Truth", "F1", "Hallucination", "Latency (s)"]

    total_f1 = 0.0
    total_hallucinations = 0
    total_latency = 0.0
    completed = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(all_questions, 1):
            question = item["question"]
            ground_truths = item["ground_truths"]
            dataset = item["dataset"]

            print(f"[{idx:>3}/{total}] [{dataset}] {question[:90]}")

            try:
                answer, latency = query_model(client, question)
                f1 = best_f1(answer, ground_truths)
                hallucination = 1 if f1 < F1_THRESHOLD else 0

                total_f1 += f1
                total_hallucinations += hallucination
                total_latency += latency
                completed += 1

                writer.writerow({
                    "Dataset": dataset,
                    "Question": question,
                    "Answer": answer,
                    "Ground Truth": " | ".join(ground_truths[:3]),
                    "F1": round(f1, 4),
                    "Hallucination": hallucination,
                    "Latency (s)": round(latency, 3),
                })
                csvfile.flush()

                print(
                    f"       Answer: {answer[:60]}\n"
                    f"       F1={f1:.3f}  Hallucination={hallucination}  Latency={latency:.2f}s\n"
                )

            except Exception as exc:
                err = str(exc)
                if "429" in err or "quota" in err.lower() or "resource_exhausted" in err.lower():
                    if "perday" in err.lower() or "per_day" in err.lower() or "daily" in err.lower():
                        print("  Daily quota exhausted — stopping. Try again tomorrow or upgrade your plan.")
                        break
                    print("  Rate limited — waiting 60s…")
                    time.sleep(60)
                    continue
                print(f"  ERROR: {err[:200]}\n")
                continue

    if completed:
        print("=" * 60)
        print(f"Saved {completed} rows to {OUTPUT_FILE}")
        print(f"Average F1:            {total_f1 / completed:.4f}")
        print(f"Hallucination Rate:    {total_hallucinations / completed:.4f}  ({total_hallucinations}/{completed})")
        print(f"Average Latency:       {total_latency / completed:.2f}s")
        print("=" * 60)
    else:
        print("No results — check errors above.")


if __name__ == "__main__":
    main()
