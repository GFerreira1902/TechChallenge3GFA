from unittest.mock import MagicMock, patch

import pytest

from src.langgraph_flows.clinical_flow import (
    node_emitir_alerta,
    node_finalizar,
    node_gerar_laudo,
    node_receber_paciente,
    node_sugerir_tratamento,
    node_verificar_alertas,
    node_verificar_exames_pendentes,
    pergunta_para_paciente,
    route_apos_alertas,
    selecionar_paciente,
)

FAKE_PATIENT = {
    "paciente_id": "PAC-001",
    "diagnostico_principal": "Dor toracica",
    "exames_pendentes": ["ECG"],
    "alertas": ["Troponina limitrofe"],
}


def test_node_receber_paciente_carrega_prontuario():
    with patch("src.langgraph_flows.clinical_flow.get_patient", return_value=FAKE_PATIENT):
        result = node_receber_paciente({"paciente_id": "PAC-001"})

    assert result["paciente"] == FAKE_PATIENT


def test_node_receber_paciente_levanta_erro_para_paciente_inexistente():
    with patch("src.langgraph_flows.clinical_flow.get_patient", return_value=None):
        with pytest.raises(ValueError):
            node_receber_paciente({"paciente_id": "PAC-INEXISTENTE"})


def test_pergunta_para_paciente_usa_diagnostico():
    pergunta = pergunta_para_paciente(FAKE_PATIENT)

    assert "Dor toracica" in pergunta
    assert pergunta.endswith("?")


def test_selecionar_paciente_respeita_id_informado(tmp_path, monkeypatch):
    patients_path = tmp_path / "patients.json"
    patients_path.write_text(
        '{"pacientes": [{"paciente_id": "PAC-TEST", "diagnostico_principal": "Sepse"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.langgraph_flows.clinical_flow.load_patients",
        lambda: {"PAC-TEST": {"paciente_id": "PAC-TEST", "diagnostico_principal": "Sepse"}},
    )

    paciente_id, pergunta = selecionar_paciente("PAC-TEST")

    assert paciente_id == "PAC-TEST"
    assert "Sepse" in pergunta


def test_selecionar_paciente_sorteia_quando_id_nao_informado(monkeypatch):
    monkeypatch.setattr(
        "src.langgraph_flows.clinical_flow.load_patients",
        lambda: {
            "PAC-A": {"paciente_id": "PAC-A", "diagnostico_principal": "Apendicite"},
            "PAC-B": {"paciente_id": "PAC-B", "diagnostico_principal": "Sepse"},
        },
    )
    monkeypatch.setattr(
        "src.langgraph_flows.clinical_flow.random.choice",
        lambda patients: patients[1],
    )

    paciente_id, pergunta = selecionar_paciente()

    assert paciente_id == "PAC-B"
    assert "Sepse" in pergunta


def test_node_verificar_exames_pendentes_detecta_exame():
    result = node_verificar_exames_pendentes({"paciente": FAKE_PATIENT})
    assert result["tem_exames_pendentes"] is True
    assert result["exames_pendentes"] == ["ECG"]


def test_node_verificar_exames_pendentes_sem_exame():
    paciente = {**FAKE_PATIENT, "exames_pendentes": []}
    result = node_verificar_exames_pendentes({"paciente": paciente})
    assert result["tem_exames_pendentes"] is False


def test_node_verificar_alertas_detecta_alerta_critico():
    result = node_verificar_alertas({"paciente": FAKE_PATIENT})
    assert result["tem_alerta_critico"] is True


def test_route_apos_alertas_direciona_para_emitir_alerta():
    assert route_apos_alertas({"tem_alerta_critico": True}) == "emitir_alerta"


def test_route_apos_alertas_direciona_para_sugerir_tratamento():
    assert route_apos_alertas({"tem_alerta_critico": False}) == "sugerir_tratamento"


def test_node_emitir_alerta_audita_e_retorna_mensagem():
    fake_logger = MagicMock()
    state = {"paciente_id": "PAC-001", "alertas_criticos": ["Troponina limitrofe"]}

    result = node_emitir_alerta(state, audit_logger=fake_logger)

    assert "PAC-001" in result["alerta_emitido"]
    assert "PROT-010" in result["alerta_emitido"]
    fake_logger.log.assert_called_once()


def test_node_sugerir_tratamento_aplica_guardrail_de_validacao_humana():
    fake_assistant = MagicMock()
    fake_assistant.ask.return_value = {
        "resposta": "Sugestao sem disclaimer.",
        "fontes": [{"protocol_id": "PROT-001"}],
    }
    state = {"paciente_id": "PAC-001", "paciente": FAKE_PATIENT}

    with patch("src.langgraph_flows.clinical_flow.MedicalAssistantRAG", return_value=fake_assistant):
        result = node_sugerir_tratamento(state)

    assert "validação humana" in result["sugestao_tratamento"].lower()
    assert result["fontes"] == [{"protocol_id": "PROT-001"}]


def test_node_finalizar_monta_resumo_com_alerta_e_exames():
    state = {
        "paciente_id": "PAC-001",
        "tem_exames_pendentes": True,
        "exames_pendentes": ["ECG"],
        "alerta_emitido": "ALERTA CRITICO - Paciente PAC-001",
        "sugestao_tratamento": "Sugestao final.",
        "fontes": [{"protocol_id": "PROT-001"}],
        "laudo": "LAUDO CLINICO de teste",
    }

    result = node_finalizar(state)

    assert "PAC-001" in result["resumo"]
    assert "ECG" in result["resumo"]
    assert "ALERTA CRITICO" in result["resumo"]
    assert "PROT-001" in result["resumo"]
    assert "LAUDO CLINICO de teste" in result["resumo"]


def test_node_gerar_laudo_usa_report_generator_e_retorna_laudo():
    fake_llm = MagicMock()
    state = {
        "paciente_id": "PAC-001",
        "paciente": FAKE_PATIENT,
        "sugestao_tratamento": "Sugestao final.",
        "fontes": [{"protocol_id": "PROT-001"}],
        "alerta_emitido": None,
    }

    with patch(
        "src.langgraph_flows.clinical_flow.gerar_laudo",
        return_value={
            "laudo": "LAUDO CLINICO gerado.",
            "avisos_seguranca": [],
            "pdf_path": "outputs/reports/laudo_PAC-001.pdf",
        },
    ) as mock_gerar_laudo:
        result = node_gerar_laudo(state, llm_client=fake_llm)

    assert result["laudo"] == "LAUDO CLINICO gerado."
    assert result["laudo_pdf_path"] == "outputs/reports/laudo_PAC-001.pdf"
    mock_gerar_laudo.assert_called_once_with(
        llm_client=fake_llm,
        paciente=FAKE_PATIENT,
        sugestao_tratamento="Sugestao final.",
        fontes=[{"protocol_id": "PROT-001"}],
        alerta_emitido=None,
    )
