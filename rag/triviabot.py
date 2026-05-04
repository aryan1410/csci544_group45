
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.prompts import PromptTemplate

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
openai_api_key = os.getenv("OPENAI_API_KEY")

FAISS_INDEX_PATH = "faiss_trivia_index"
TOP_K_RETRIEVE = 5
RERANK_TOP_N = 3
COSINE_THRESHOLD = 0.45

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a factoid question-answering system.

Use only the provided context to answer the question.

Context:
{context}

Question:
{question}

Rules:
- Return only the shortest exact answer phrase found in the context.
- Do not explain.
- Do not write a full sentence.
- Do not include extra words.
- If the answer is a person, place, date, number, or entity, output only that entity.
- If multiple forms are possible, return the most concise standard form.
- If the answer cannot be found in the context, return exactly: Unknown

Answer:
""".strip()
)


def load_vectordb():
    if not os.path.exists(FAISS_INDEX_PATH):
        print(
            f"[triviabot] Index not found at '{FAISS_INDEX_PATH}/'. "
            "Run `python knowledge_base_builder.py` to build it."
        )
        return None

    print(f"[triviabot] Loading index from '{FAISS_INDEX_PATH}/'...")
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def create_qa_chain(vectordb):
    base_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVE}
    )

    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=COSINE_THRESHOLD,
        k=RERANK_TOP_N
    )

    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=reranking_retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT}
    )