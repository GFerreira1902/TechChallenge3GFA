"""Geracao automatica de laudos clinicos (documento formatado), a partir da
sugestao de conduta produzida pelo RAG. Fecha o requisito do enunciado sobre
"modelos de laudos, receitas e procedimentos internos" como tipo de saida do
assistente (o modelo de referencia esta em data/raw/synthetic_laudo_template.md).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from src.guardrails.safety_rules import enforce_human_validation

LAUDO_SYSTEM_PROMPT = (
    "Você é um assistente que redige laudos clínicos estruturados para um "
    "hospital. Use exatamente este formato, preenchendo cada seção de forma "
    "objetiva e curta, em português:\n\n"
    "LAUDO CLÍNICO\n"
    "Paciente: <id>\n"
    "Data/hora: <data>\n\n"
    "1. Diagnóstico principal\n<texto>\n\n"
    "2. Exames pendentes ou realizados\n<texto>\n\n"
    "3. Conduta sugerida\n<texto>\n\n"
    "4. Protocolos internos consultados (fonte)\n<texto>\n\n"
    "5. Alertas ativos\n<texto>\n\n"
    "Não invente informações que não estejam no contexto fornecido. Você "
    "nunca prescreve tratamentos diretamente sem ressalva de validação humana."
)


def gerar_laudo(
    llm_client: Any,
    paciente: dict,
    sugestao_tratamento: str,
    fontes: list[dict],
    alerta_emitido: Optional[str] = None,
) -> dict:
    """Gera um laudo clinico formatado com base no atendimento processado.

    Retorna um dict com o texto do laudo e os avisos de seguranca levantados.
    """
    protocolos = ", ".join(f["protocol_id"] for f in fontes) or "nenhum"
    exames = ", ".join(paciente.get("exames_pendentes", [])) or "nenhum"

    user_message = (
        f"Paciente: {paciente['paciente_id']}\n"
        f"Data/hora: {datetime.now().isoformat()}\n"
        f"Diagnóstico principal: {paciente['diagnostico_principal']}\n"
        f"Exames pendentes: {exames}\n"
        f"Conduta sugerida pelo assistente: {sugestao_tratamento}\n"
        f"Protocolos consultados: {protocolos}\n"
        f"Alerta crítico: {alerta_emitido or 'nenhum'}\n\n"
        "Redija o laudo clínico completo no formato especificado."
    )

    laudo_bruto = llm_client.ask(
        system_prompt=LAUDO_SYSTEM_PROMPT,
        user_message=user_message,
        metadata={"paciente_id": paciente["paciente_id"], "tipo": "laudo_clinico"},
    )

    laudo_validado, avisos = enforce_human_validation(laudo_bruto)
    return {"laudo": laudo_validado, "avisos_seguranca": avisos}
