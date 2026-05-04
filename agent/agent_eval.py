import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests

BACKEND_URL = "http://localhost:8000/run_sync"

PAYLOAD_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "template": "bullet_summary",
    "search_budget": 2,
}

QUESTIONS_JSON = Path("../data/questions_subset.json")
QA_TXT = Path("../data/qa_dataset.txt")
OUTPUT_CSV = Path("agent_eval_results.csv")

CSV_COLUMNS = ["Dataset", "Question", "Answer", "Ground Truth", "F1", "Hallucination", "Latency (s)"]

def load_json_questions() -> list[dict]:
    with QUESTIONS_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "dataset": item["dataset"],
            "question": item["question"],
            "ground_truths": item["ground_truths"],
        }
        for item in data
    ]


def load_txt_questions() -> list[dict]:
    text = QA_TXT.read_text(encoding="utf-8")
    questions = []
    # Each block: "Question N: ...\nAnswer N: ..."
    blocks = re.split(r"\n{2,}", text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        q_match = re.match(r"Question\s+\d+:\s*(.*)", lines[0], re.IGNORECASE)
        a_match = re.match(r"Answer\s+\d+:\s*(.*)", lines[1], re.IGNORECASE)
        if q_match and a_match:
            questions.append({
                "dataset": "Custom",
                "question": q_match.group(1).strip(),
                "ground_truths": [a_match.group(1).strip()],
            })
    return questions


def _normalize_tokens(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = _normalize_tokens(pred)
    gt_tokens = _normalize_tokens(gt)
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1(pred: str, ground_truths: list[str]) -> float:
    return max(token_f1(pred, gt) for gt in ground_truths)


def query_backend(question: str) -> tuple[str, float]:
    payload = {
        "query": question,
        "messages": [],
        "config": PAYLOAD_CONFIG,
    }
    t0 = time.time()
    resp = requests.post(BACKEND_URL, json=payload, timeout=120)
    latency = round(time.time() - t0, 3)

    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"Backend error: {data['error']}")

    answer = data["report"]["content"].strip()
    return answer, latency


def main():
    all_questions: list[dict] = []
    if QUESTIONS_JSON.exists():
        all_questions.extend(load_json_questions())
        print(f"Loaded {len(load_json_questions())} questions from {QUESTIONS_JSON}")
    else:
        print(f"WARNING: {QUESTIONS_JSON} not found, skipping.")

    if QA_TXT.exists():
        txt_qs = load_txt_questions()
        all_questions.extend(txt_qs)
        print(f"Loaded {len(txt_qs)} questions from {QA_TXT}")
    else:
        print(f"WARNING: {QA_TXT} not found, skipping.")

    total = len(all_questions)
    print(f"\nTotal questions to evaluate: {total}")
    print(f"Output: {OUTPUT_CSV}\n")

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, item in enumerate(all_questions, 1):
            dataset = item["dataset"]
            question = item["question"]
            ground_truths = item["ground_truths"]
            gt_str = " | ".join(ground_truths)

            print(f"[{i}/{total}] {dataset} — {question[:80]}")

            try:
                answer, latency = query_backend(question)
            except Exception as e:
                print(f"  ERROR: {e}")
                answer = "ERROR"
                latency = 0.0

            f1 = round(max_f1(answer, ground_truths), 4)
            hallucination = 1 if f1 == 0.0 else 0

            print(f"  Answer: {answer!r}  |  F1: {f1}  |  Hallucination: {hallucination}  |  Latency: {latency}s")

            writer.writerow({
                "Dataset": dataset,
                "Question": question,
                "Answer": answer,
                "Ground Truth": gt_str,
                "F1": f1,
                "Hallucination": hallucination,
                "Latency (s)": latency,
            })
            csvfile.flush()

    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
