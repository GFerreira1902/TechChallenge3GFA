from unittest.mock import MagicMock

from src.langchain_pipeline.report_generator import gerar_laudo

FAKE_PATIENT = {
    "paciente_id": "PAC-007",
    "diagnostico_principal": "Pos-operatorio de apendicectomia",
    "exames_pendentes": [],
}


def test_gerar_laudo_chama_llm_e_aplica_guardrail():
    fake_llm = MagicMock()
    fake_llm.ask.return_value = "LAUDO CLINICO\nPaciente: PAC-007\n..."

    resultado = gerar_laudo(
        llm_client=fake_llm,
        paciente=FAKE_PATIENT,
        sugestao_tratamento="Avaliar alta conforme PROT-006.",
        fontes=[{"protocol_id": "PROT-006"}],
    )

    assert "PAC-007" in resultado["laudo"]
    assert "validação humana" in resultado["laudo"].lower()

    called_user_message = fake_llm.ask.call_args.kwargs["user_message"]
    assert "PAC-007" in called_user_message
    assert "PROT-006" in called_user_message


def test_gerar_laudo_inclui_alerta_quando_presente():
    fake_llm = MagicMock()
    fake_llm.ask.return_value = "LAUDO CLINICO gerado."

    gerar_laudo(
        llm_client=fake_llm,
        paciente=FAKE_PATIENT,
        sugestao_tratamento="Conduta X.",
        fontes=[],
        alerta_emitido="ALERTA CRITICO - Paciente PAC-007",
    )

    called_user_message = fake_llm.ask.call_args.kwargs["user_message"]
    assert "ALERTA CRITICO" in called_user_message


def test_gerar_laudo_usa_nenhum_quando_sem_fontes_ou_exames():
    fake_llm = MagicMock()
    fake_llm.ask.return_value = "LAUDO CLINICO gerado."

    gerar_laudo(
        llm_client=fake_llm,
        paciente=FAKE_PATIENT,
        sugestao_tratamento="Conduta X.",
        fontes=[],
    )

    called_user_message = fake_llm.ask.call_args.kwargs["user_message"]
    assert "Protocolos consultados: nenhum" in called_user_message
    assert "Exames pendentes: nenhum" in called_user_message
