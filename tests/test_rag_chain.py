from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.langchain_pipeline.rag_chain import MedicalAssistantRAG


def _make_fake_vectorstore(docs):
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = docs
    return vectorstore


def test_ask_retrieves_sources_and_includes_them_in_prompt():
    """A resposta deve citar as fontes recuperadas e o LLM deve receber o contexto."""
    docs = [
        Document(
            page_content="Conteudo do protocolo de sepse.",
            metadata={"protocol_id": "PROT-002", "titulo": "Protocolo de Sepse"},
        )
    ]

    fake_llm = MagicMock()
    fake_llm.ask.return_value = "Resposta baseada no protocolo de sepse."

    with patch("src.langchain_pipeline.rag_chain.load_index", return_value=_make_fake_vectorstore(docs)):
        assistant = MedicalAssistantRAG(llm_client=fake_llm, top_k=1)
        resultado = assistant.ask("Como tratar sepse?")

    assert resultado["fontes"] == [
        {"protocol_id": "PROT-002", "titulo": "Protocolo de Sepse", "conteudo": "Conteudo do protocolo de sepse."}
    ]
    assert resultado["resposta"] == "Resposta baseada no protocolo de sepse."

    called_user_message = fake_llm.ask.call_args.kwargs["user_message"]
    assert "PROT-002" in called_user_message
    assert "Como tratar sepse?" in called_user_message


def test_ask_includes_patient_context_when_paciente_id_informado():
    docs = [Document(page_content="Conteudo.", metadata={"protocol_id": "PROT-001", "titulo": "Teste"})]
    fake_llm = MagicMock()
    fake_llm.ask.return_value = "Resposta."
    fake_patient = {
        "paciente_id": "PAC-001",
        "idade": 68,
        "sexo": "M",
        "diagnostico_principal": "Dor toracica",
        "historico": "Hipertenso.",
        "medicacoes_atuais": [],
        "exames_pendentes": [],
        "alertas": [],
    }

    with patch("src.langchain_pipeline.rag_chain.load_index", return_value=_make_fake_vectorstore(docs)), patch(
        "src.langchain_pipeline.rag_chain.get_patient", return_value=fake_patient
    ):
        assistant = MedicalAssistantRAG(llm_client=fake_llm)
        assistant.ask("Pergunta qualquer", paciente_id="PAC-001")

    called_user_message = fake_llm.ask.call_args.kwargs["user_message"]
    assert "PAC-001" in called_user_message
    assert "Dor toracica" in called_user_message


def test_ask_without_paciente_id_uses_placeholder_context():
    docs = [Document(page_content="Conteudo.", metadata={"protocol_id": "PROT-001", "titulo": "Teste"})]
    fake_llm = MagicMock()
    fake_llm.ask.return_value = "Resposta."

    with patch("src.langchain_pipeline.rag_chain.load_index", return_value=_make_fake_vectorstore(docs)):
        assistant = MedicalAssistantRAG(llm_client=fake_llm)
        resultado = assistant.ask("Pergunta qualquer")

    assert resultado["paciente_id"] is None
    called_user_message = fake_llm.ask.call_args.kwargs["user_message"]
    assert "Nenhum paciente informado" in called_user_message
