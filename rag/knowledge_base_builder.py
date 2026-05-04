import json
import os

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
openai_api_key = os.getenv("OPENAI_API_KEY")

FAISS_INDEX_PATH = "faiss_trivia_index"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

ANNUAL_REPORT_FILES = [
    "apple_fy2023_10k.txt",
    "microsoft_fy2023_10k.txt",
    "alphabet_fy2022_10k.txt",
    "meta_fy2022_10k.txt",
    "amazon_fy2022_10k.txt",
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512 * 4,
    chunk_overlap=64 * 4,
    length_function=len,
)



def load_annual_report(txt_path: str) -> list[Document]:
    filename = os.path.basename(txt_path)
    company = filename.replace("_10k.txt", "").replace("_", " ").title()
    print(f"[{company}] Loading from {txt_path}...")
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    doc = Document(
        page_content=content,
        metadata={"source": filename, "company": company, "type": "annual_report"},
    )
    print(f"[{company}] Loaded ({len(content):,} chars).")
    return [doc]



def load_trivia_qa_docs() -> list[Document]:
    from datasets import load_dataset

    print("Loading TriviaQA (rc.nocontext / validation) from HuggingFace...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=False)
    docs = []
    for item in ds:
        question = item["question"]
        aliases = list(item["answer"].get("aliases", []))
        main_answer = item["answer"]["value"]
        all_answers = [main_answer] + [a for a in aliases if a != main_answer]
        text = f"Question: {question}\nAnswer: {', '.join(all_answers)}"
        docs.append(Document(
            page_content=text,
            metadata={"source": "trivia_qa", "type": "trivia_qa"},
        ))
    print(f"TriviaQA: loaded {len(docs):,} Q&A pairs.")
    return docs


def load_pop_qa_docs() -> list[Document]:
    from datasets import load_dataset

    print("Loading PopQA (akariasai/popqa / test) from HuggingFace...")
    ds = load_dataset("akariasai/popqa", split="test")
    docs = []
    for item in ds:
        question = item["question"]
        raw = item["possible_answers"]
        answers = json.loads(raw) if isinstance(raw, str) else list(raw)
        answers = [str(a) for a in answers if a]
        if not answers:
            continue
        text = f"Question: {question}\nAnswer: {', '.join(answers)}"
        docs.append(Document(
            page_content=text,
            metadata={"source": "popqa", "type": "popqa"},
        ))
    print(f"PopQA: loaded {len(docs):,} Q&A pairs.")
    return docs



def build_knowledge_base():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    all_docs: list[Document] = []

    trivia_docs = load_trivia_qa_docs()
    all_docs.extend(trivia_docs)

    popqa_docs = load_pop_qa_docs()
    all_docs.extend(popqa_docs)

    report_docs: list[Document] = []
    missing: list[str] = []
    for filename in ANNUAL_REPORT_FILES:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            report_docs.extend(load_annual_report(path))
        else:
            missing.append(filename)
            print(f"WARNING: '{filename}' not found in {DATA_DIR}, skipping.")

    if missing:
        print(f"\n{len(missing)} annual report file(s) missing — KB may be incomplete.")

    if report_docs:
        print(f"\nChunking {len(report_docs)} annual report(s) at 512 tokens / 64 token overlap...")
        chunked_reports = splitter.split_documents(report_docs)
        print(f"Annual report chunks: {len(chunked_reports)}")
        all_docs.extend(chunked_reports)

    if not all_docs:
        print("ERROR: No documents loaded. Aborting.")
        return None

    print(f"\nTotal documents to embed: {len(all_docs):,}")
    print("Embedding (this may take several minutes)...")
    vectordb = FAISS.from_documents(all_docs, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)

    print(f"\nDone. Index saved to '{FAISS_INDEX_PATH}/'")
    print(f"  TriviaQA Q&A pairs: {len(trivia_docs):,}")
    print(f"  PopQA Q&A pairs:    {len(popqa_docs):,}")
    print(f"  Annual report chunks: {len(all_docs) - len(trivia_docs) - len(popqa_docs):,}")
    return vectordb


if __name__ == "__main__":
    build_knowledge_base()
