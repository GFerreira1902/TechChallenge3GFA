"""Pipeline RAG: recupera protocolos relevantes, injeta contexto do paciente e
gera a resposta do assistente medico, sempre citando as fontes utilizadas
(requisito de explainability da Fase 3).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from src.langchain_pipeline.knowledge_base import load_index, load_protocol_documents_for_search
from src.langchain_pipeline.local_llm_client import LocalFineTunedLLMClient
from src.langchain_pipeline.patient_records import format_patient_context, get_patient

SEARCH_STOPWORDS = {
    "a", "as", "o", "os", "de", "do", "da", "dos", "das", "e", "em",
    "para", "por", "com", "sem", "um", "uma", "quais", "qual", "sao",
    "são", "como", "sobre", "proximas", "próximas", "condutas", "recomendadas",
}

SYSTEM_PROMPT = (
    "Você é um assistente virtual médico do hospital, treinado para auxiliar "
    "profissionais de saúde com base nos protocolos internos e no prontuário do "
    "paciente fornecidos abaixo. Responda de forma clara e objetiva, citando o "
    "identificador do protocolo utilizado (ex.: PROT-001) quando aplicável. "
    "Você nunca prescreve tratamentos diretamente: toda sugestão deve ser "
    "validada por um profissional de saúde responsável antes de qualquer "
    "conduta clínica. Se a informação não estiver nos protocolos fornecidos, "
    "diga isso explicitamente em vez de inventar uma resposta."
)


class MedicalAssistantRAG:
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        top_k: int = 3,
    ):
        self.vectorstore = load_index()
        self.llm_client = llm_client or LocalFineTunedLLMClient()
        self.top_k = top_k

    def _retrieve(self, question: str) -> list[dict]:
        semantic_docs = self.vectorstore.similarity_search(question, k=self.top_k)
        query_terms = {
            term for term in re.findall(r"[a-zà-ú]+", question.lower())
            if len(term) > 3 and term not in SEARCH_STOPWORDS
        }
        candidates = {doc.metadata["protocol_id"]: doc for doc in semantic_docs}

        # A similaridade semantica pode confundir protocolos medicos proximos.
        # O casamento explicito de termos no titulo/conteudo garante que "sepse"
        # priorize PROT-002 e que "insuficiencia cardiaca" priorize PROT-011.
        for doc in load_protocol_documents_for_search():
            searchable = f"{doc.metadata['titulo']} {doc.page_content}".lower()
            lexical_score = sum(
                1 for term in query_terms if term in searchable
            )
            if lexical_score and doc.metadata["protocol_id"] not in candidates:
                candidates[doc.metadata["protocol_id"]] = doc

        ranked_docs = sorted(
            candidates.values(),
            key=lambda doc: (
                5 * sum(term in doc.metadata["titulo"].lower() for term in query_terms)
                + sum(term in doc.page_content.lower() for term in query_terms),
                doc.metadata["protocol_id"],
            ),
            reverse=True,
        )[: self.top_k]
        docs = ranked_docs or semantic_docs
        return [
            {
                "protocol_id": doc.metadata["protocol_id"],
                "titulo": doc.metadata["titulo"],
                "conteudo": doc.page_content,
            }
            for doc in docs
        ]

    def _retrieve_for_patient(self, question: str, paciente_id: Optional[str]) -> list[dict]:
        """Prioriza os protocolos explicitamente associados ao prontuario ficticio."""
        patient = get_patient(paciente_id) if paciente_id else None
        protocol_ids = patient.get("protocolos_relevantes", []) if patient else []
        if not protocol_ids:
            return self._retrieve(question)

        documents = {doc.metadata["protocol_id"]: doc for doc in load_protocol_documents_for_search()}
        preferred = [documents[protocol_id] for protocol_id in protocol_ids if protocol_id in documents]
        return [
            {"protocol_id": doc.metadata["protocol_id"], "titulo": doc.metadata["titulo"], "conteudo": doc.page_content}
            for doc in preferred
        ][: self.top_k]

    def ask(self, question: str, paciente_id: Optional[str] = None) -> dict[str, Any]:
        """Responde a pergunta clinica com base nos protocolos (+ prontuario, se informado)."""
        sources = self._retrieve_for_patient(question, paciente_id)
        protocolos_texto = "\n\n".join(
            f"[{s['protocol_id']}] {s['titulo']}\n{s['conteudo']}" for s in sources
        )

        patient = get_patient(paciente_id) if paciente_id else None
        patient_context = format_patient_context(patient) if patient else "Nenhum paciente informado."

        user_message = (
            f"Protocolos internos relevantes:\n{protocolos_texto}\n\n"
            f"Contexto do paciente:\n{patient_context}\n\n"
            f"Pergunta do profissional de saude: {question}"
        )

        answer = self.llm_client.ask(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            metadata={
                "paciente_id": paciente_id,
                "fontes": [s["protocol_id"] for s in sources],
                "fontes_detalhadas": [
                    {"protocol_id": s["protocol_id"], "titulo": s["titulo"]}
                    for s in sources
                ],
            },
        )

        return {"resposta": answer, "fontes": sources, "paciente_id": paciente_id}


if __name__ == "__main__":
    assistant = MedicalAssistantRAG()
    resultado = assistant.ask(
        "Quais exames sao necessarios em um caso de dor toracica aguda?",
        paciente_id="PAC-001",
    )
    print(resultado["resposta"])
    print("Fontes:", [f["protocol_id"] for f in resultado["fontes"]])
