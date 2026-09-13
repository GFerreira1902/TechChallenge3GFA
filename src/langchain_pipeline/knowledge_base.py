"""Base de conhecimento (RAG) construida sobre os protocolos internos sinteticos.

Roda inteiramente em CPU (embeddings leves via sentence-transformers) para nao
competir por VRAM com o fine-tuning, que pode estar rodando na GPU ao mesmo tempo.
"""
from __future__ import annotations

import json
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROTOCOLS_PATH = "data/raw/synthetic_protocols.json"
INDEX_DIR = "data/processed/faiss_protocols_index"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Modelo de embeddings pequeno, forcado a rodar em CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def load_protocol_documents(path: str = PROTOCOLS_PATH) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Document(
            page_content=protocolo["conteudo"],
            metadata={
                "protocol_id": protocolo["protocol_id"],
                "titulo": protocolo["titulo"],
                "categoria": protocolo["categoria"],
            },
        )
        for protocolo in data["protocolos"]
    ]


def load_protocol_documents_for_search(path: str = PROTOCOLS_PATH) -> list[Document]:
    """Alias explicito para recuperar todos os protocolos em buscas hibridas."""
    return load_protocol_documents(path)


def build_index(index_dir: str = INDEX_DIR) -> FAISS:
    """Constroi (ou reconstroi) o indice FAISS a partir dos protocolos sinteticos."""
    documents = load_protocol_documents()
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    return vectorstore


def load_index(index_dir: str = INDEX_DIR) -> FAISS:
    """Carrega o indice FAISS do disco, reconstruindo se ainda nao existir."""
    if not os.path.exists(index_dir):
        return build_index(index_dir)

    embeddings = get_embeddings()
    return FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    vs = build_index()
    print(f"Indice FAISS construido com {vs.index.ntotal} protocolos em {INDEX_DIR}")
