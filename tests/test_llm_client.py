import os
from unittest.mock import MagicMock, patch

from src.guardrails.audit_logger import AuditLogger
from src.langchain_pipeline.llm_client import MedicalAssistantLLMClient


def test_client_initialization_no_key_enables_mock_mode():
    """Sem GROQ_API_KEY, o cliente deve ativar o modo simulado ao invés de falhar."""
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=True):
        client = MedicalAssistantLLMClient(api_key=None)
        assert client.mock_mode is True
        assert client.client is None


def test_ask_mock_mode_still_audits(tmp_path):
    """Mesmo em modo simulado, a interação deve ser registrada para auditoria."""
    history_file = tmp_path / "history.json"
    with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=True):
        client = MedicalAssistantLLMClient(
            api_key=None, audit_logger=AuditLogger(history_path=str(history_file))
        )
        response = client.ask(system_prompt="sys", user_message="Qual o protocolo?")

    assert "MODO SIMULADO" in response
    assert os.path.exists(history_file)


@patch("src.langchain_pipeline.llm_client.Groq")
def test_ask_success_calls_groq_and_audits(mock_groq_class, tmp_path):
    """Com API key configurada, a resposta deve vir do client Groq e ser auditada."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Resposta clínica baseada no protocolo interno."
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    mock_groq_class.return_value = mock_client

    history_file = tmp_path / "history.json"
    client = MedicalAssistantLLMClient(
        api_key="fake_key", audit_logger=AuditLogger(history_path=str(history_file))
    )

    response = client.ask(system_prompt="sys", user_message="Qual o protocolo?")

    assert response == "Resposta clínica baseada no protocolo interno."
    assert os.path.exists(history_file)
