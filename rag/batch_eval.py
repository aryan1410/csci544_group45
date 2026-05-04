import os
import csv
import json
import time
import string
import re
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from triviabot import load_vectordb, create_qa_chain

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

INPUT_JSON = "../data/questions_subset.json"
INPUT_TXT = "../data/qa_dataset.txt"
OUTPUT_CSV = "rag_results.csv"

JSON_LIMIT = None
TXT_LIMIT = 100
HALLUC_THRESHOLD = 0.4

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_questions_from_json(json_path, limit=None):
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    rows = []
    for item in items:
        question = item.get("question", "").strip()
        truths = item.get("ground_truths", [])
        ground_truth = truths[0].strip() if truths else ""

        if question and ground_truth:
            rows.append({
                "dataset": item.get("dataset", "unknown"),
                "question": question,
                "ground_truth": ground_truth,
                "all_ground_truths": truths if truths else [ground_truth]
            })

    return rows[:limit] if limit else rows


def load_questions_from_txt(txt_path, limit=None):
    rows = []

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    question = None

    for line in lines:
        line = line.strip()

        if line.startswith("Question"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                question = parts[1].strip()

        elif line.startswith("Answer") and question:
            parts = line.split(":", 1)
            if len(parts) == 2:
                answer = parts[1].strip()

                rows.append({
                    "dataset": "custom_qa",
                    "question": question,
                    "ground_truth": answer,
                    "all_ground_truths": [answer]
                })

            question = None

    return rows[:limit] if limit else rows


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def best_f1(prediction: str, all_ground_truths: list[str]) -> float:
    if not all_ground_truths:
        return 0.0
    return max(token_f1(prediction, gt) for gt in all_ground_truths)


def strip_leading_answer_phrases(text: str) -> str:
    text = text.strip()

    prefixes = [
        r"^final answer\s*:\s*",
        r"^answer\s*:\s*",
        r"^the answer is\s+",
        r"^it is\s+",
        r"^it's\s+",
        r"^this is\s+",
        r"^the correct answer is\s+",
    ]

    for pattern in prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def extract_short_answer(prediction: str, all_ground_truths: list[str]) -> str:
    if not prediction:
        return ""

    original = prediction.strip()
    norm_pred = normalize(original)

    matched_aliases = []
    for alias in all_ground_truths:
        norm_alias = normalize(alias)
        if norm_alias and norm_alias in norm_pred:
            matched_aliases.append(alias.strip())

    if matched_aliases:
        matched_aliases.sort(key=lambda x: len(normalize(x)), reverse=True)
        return matched_aliases[0]

    short = strip_leading_answer_phrases(original)

    short = short.split("\n")[0].strip()

    short = re.split(r"[.;!?]", short)[0].strip()

    short = short.strip(" \"'()[]{}")

    short = " ".join(short.split())

    return short


def score_faithfulness(question: str, answer: str, contexts: list[str]):
    context_str = "\n\n".join(contexts)
    prompt = f"""You are evaluating whether an AI answer is faithful to the retrieved context.

Context:
{context_str}

Question: {question}
Answer: {answer}

Instructions:
- Identify every factual claim in the Answer.
- Check if each claim is directly supported by the Context.
- Return ONLY a number between 0.0 and 1.0 representing the fraction of claims supported.
- 1.0 = all claims supported, 0.0 = no claims supported.
- Return ONLY the number, nothing else."""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return round(float(resp.choices[0].message.content.strip()), 4)
    except Exception as e:
        print(f"  [faithfulness error] {e}")
        return ""


def score_answer_relevancy(question: str, answer: str):
    prompt = f"""You are evaluating whether an AI answer is relevant to the question asked.

Question: {question}
Answer: {answer}

Instructions:
- Score how well the answer addresses the question asked.
- Penalize answers that are incomplete, off-topic, or contain unnecessary information.
- Return ONLY a number between 0.0 and 1.0.
- 1.0 = perfectly relevant, 0.0 = completely irrelevant.
- Return ONLY the number, nothing else."""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return round(float(resp.choices[0].message.content.strip()), 4)
    except Exception as e:
        print(f"  [answer_relevancy error] {e}")
        return ""


def safe_score(question, answer, contexts):
    faith = score_faithfulness(question, answer, contexts)
    relevancy = score_answer_relevancy(question, answer)
    if isinstance(faith, float) and isinstance(relevancy, float):
        ragas_score = round((faith + relevancy) / 2, 4)
    else:
        ragas_score = ""
    return faith, relevancy, ragas_score


def main():
    print("[1/4] Loading vector DB...")
    vectordb = load_vectordb()
    if vectordb is None:
        print("ERROR: Run `python knowledge_base_builder.py` first.")
        return

    print("[2/4] Creating QA chain...")
    qa_chain = create_qa_chain(vectordb)

    print(f"[3/4] Loading questions from {INPUT_JSON} and {INPUT_TXT}...")
    json_questions = load_questions_from_json(INPUT_JSON, limit=JSON_LIMIT)
    txt_questions = load_questions_from_txt(INPUT_TXT, limit=TXT_LIMIT)
    questions = json_questions + txt_questions

    print(f"Loaded {len(json_questions)} JSON questions.")
    print(f"Loaded {len(txt_questions)} TXT questions.")
    print(f"Total questions: {len(questions)}")

    results = []

    print("[4/4] Running evaluation...\n")
    for i, row in enumerate(questions, start=1):
        question = row["question"]
        ground_truth = row["ground_truth"]
        all_ground_truths = row["all_ground_truths"]

        print(f"[{i}/{len(questions)}] {question}")

        try:
            t0 = time.time()
            response = qa_chain.invoke({"query": question})
            latency = round(time.time() - t0, 2)

            raw_answer = response.get("result", "").strip()
            source_docs = response.get("source_documents", [])
            contexts = [doc.page_content for doc in source_docs]
            retrieved_context = "\n\n---\n\n".join(contexts)

            scored_answer = extract_short_answer(raw_answer, all_ground_truths)

            f1 = best_f1(scored_answer, all_ground_truths)
            hallucinated = 1 if f1 < HALLUC_THRESHOLD else 0

            faith, relevancy, ragas_score = safe_score(question, raw_answer, contexts)

            print(f"  Raw Answer:       {raw_answer}")
            print(f"  Scored Answer:    {scored_answer}")
            print(f"  Ground Truth:     {ground_truth}")
            print(f"  F1:               {f1}")
            print(f"  Hallucinated:     {hallucinated}")
            print(f"  Latency:          {latency}s")
            print(f"  Faithfulness:     {faith}")
            print(f"  Answer Relevancy: {relevancy}")
            print(f"  RAGAS Score:      {ragas_score}")

            results.append({
                "dataset": row["dataset"],
                "question": question,
                "ground_truth": ground_truth,
                "raw_answer": raw_answer,
                "scored_answer": scored_answer,
                "f1": f1,
                "hallucinated": hallucinated,
                "faithfulness": faith,
                "answer_relevancy": relevancy,
                "ragas_score": ragas_score,
                "latency_s": latency,
                "retrieved_context": retrieved_context,
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "dataset": row["dataset"],
                "question": question,
                "ground_truth": ground_truth,
                "raw_answer": "",
                "scored_answer": "",
                "f1": "",
                "hallucinated": "",
                "faithfulness": "",
                "answer_relevancy": "",
                "ragas_score": "",
                "latency_s": "",
                "retrieved_context": "",
            })

    print(f"\nSaving to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "question", "ground_truth", "raw_answer", "scored_answer",
            "f1", "hallucinated",
            "faithfulness", "answer_relevancy", "ragas_score",
            "latency_s", "retrieved_context"
        ])
        writer.writeheader()
        writer.writerows(results)

    scored = [r for r in results if isinstance(r["f1"], float)]
    if scored:
        avg_f1 = round(sum(r["f1"] for r in scored) / len(scored), 4)
        halluc_rate = round(sum(r["hallucinated"] for r in scored) / len(scored) * 100, 1)
        avg_lat = round(sum(r["latency_s"] for r in scored) / len(scored), 2)

        faith_rows = [r["faithfulness"] for r in scored if isinstance(r["faithfulness"], float)]
        rel_rows = [r["answer_relevancy"] for r in scored if isinstance(r["answer_relevancy"], float)]
        ragas_rows = [r["ragas_score"] for r in scored if isinstance(r["ragas_score"], float)]

        avg_faith = round(sum(faith_rows) / len(faith_rows), 4) if faith_rows else ""
        avg_rel = round(sum(rel_rows) / len(rel_rows), 4) if rel_rows else ""
        avg_ragas = round(sum(ragas_rows) / len(ragas_rows), 4) if ragas_rows else ""

        print(f"\n{'=' * 50}")
        print(f"RESULTS SUMMARY  ({len(scored)}/{len(results)} scored)")
        print(f"{'=' * 50}")
        print(f"Avg F1:               {avg_f1}")
        print(f"Hallucination Rate:   {halluc_rate}%")
        print(f"Avg Faithfulness:     {avg_faith}")
        print(f"Avg Answer Relevancy: {avg_rel}")
        print(f"Avg RAGAS Score:      {avg_ragas}")
        print(f"Avg Latency:          {avg_lat}s")
        print(f"CSV saved:            {OUTPUT_CSV}")


if __name__ == "__main__":
    main()