import os
from pathlib import Path
import re

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openrouter import ChatOpenRouter


load_dotenv()


PDF_PATH = Path("data/employee_handbook.pdf")
TOP_K = 4

# Validated output schema

class GroundedResponse(BaseModel):
    answer: str = Field(
        description="Answer based only on the Employee Handbook."
    )
    citations: list[str] = Field(
        description="Page citations supporting the answer."
    )
    refused: bool = Field(
        description="True when the handbook does not contain the answer."
    )

# 2. Build retriever

def build_retriever():
    """Load, chunk, embed and index the Employee Handbook."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Employee handbook not found at: {PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )


retriever = build_retriever()

# 3. Model

model = ChatOpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0,
    timeout=30_000,
    max_retries=2,
)

# 4. Retrieve relevant handbook chunks

def retrieve_context(query: str):

    if not query.strip():
        raise ValueError("Question cannot be empty.")

    documents = retriever.invoke(query)

    context_parts = []
    pages = []

    for document in documents:

        page = document.metadata.get("page", 0) + 1
        citation = f"[Page {page}]"

        context_parts.append(
            f"{citation}\n{document.page_content}"
        )

        if citation not in pages:
            pages.append(citation)

    context = "\n\n".join(context_parts)

    return context, pages

# 5. Generate grounded answer

def answer_question(question: str) -> GroundedResponse:

    context, retrieved_pages = retrieve_context(question)

    prompt = f"""
You are an Employee HR Assistant.

Answer the employee's question using ONLY the Employee Handbook
context provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent company policies.
3. If the context clearly contains the answer, answer it concisely.
4. Include the supporting page citation exactly as shown in the context.
5. If the context does NOT contain enough information to answer the
   question, respond exactly with:
   NOT_FOUND

Employee question:
{question}

Employee Handbook context:
{context}
"""

    response = model.invoke(prompt)

    answer = response.content.strip()

    # Grounded refusal
    if answer == "NOT_FOUND":
        return GroundedResponse(
            answer=(
                "The requested information could not be found "
                "in the Employee Handbook."
            ),
            citations=[],
            refused=True,
        )

    # Keep only citations actually mentioned by the model
    page_numbers = re.findall(r"[\(\[]Page\s*(\d+)[\)\]]",answer,flags=re.IGNORECASE,)

    citations = [f"[Page {page}]" for page in page_numbers]

    return GroundedResponse(
        answer=answer,
        citations=citations,
        refused=False,
    )

# 6. Run examples

if __name__ == "__main__":

    questions = [
        "How many annual leave days do eligible full-time employees receive?",
        "Does the company provide employees with a free gym membership?",
    ]

    print("\n=== GROUNDED KNOWLEDGE ===")

    for question in questions:

        response = answer_question(question)

        print(f"\nQuestion: {question}")
        print(f"Answer: {response.answer}")
        print(f"Citations: {response.citations}")
        print(f"Refused: {response.refused}")