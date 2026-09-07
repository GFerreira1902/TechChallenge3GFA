"""Pipeline RAG: recupera protocolos relevantes, injeta contexto do paciente e
gera a resposta do assistente medico, sempre citando as fontes utilizadas
(requisito de explainability da Fase 3).
"""
from __future__ import annotations

from typing import Any, Optional

from src.langchain_pipeline.knowledge_base import load_index
from src.langchain_pipeline.local_llm_client import LocalFineTunedLLMClient
from src.langchain_pipeline.patient_records import format_patient_context, get_patient

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
        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        return [
            {
                "protocol_id": doc.metadata["protocol_id"],
                "titulo": doc.metadata["titulo"],
                "conteudo": doc.page_content,
            }
            for doc in docs
        ]

    def ask(self, question: str, paciente_id: Optional[str] = None) -> dict[str, Any]:
        """Responde a pergunta clinica com base nos protocolos (+ prontuario, se informado)."""
        sources = self._retrieve(question)
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
