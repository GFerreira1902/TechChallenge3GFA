# Diagrama do Fluxo — Assistente Virtual Médico (Fase 3)

Este documento descreve, com diagramas Mermaid, a arquitetura do assistente: o
pipeline de RAG orquestrado com **LangChain** e o fluxo de decisão clínica
orquestrado com **LangGraph**, incluindo os pontos de guardrails e auditoria.

## 1. Visão geral do pipeline

```mermaid
flowchart TD
    A[Médico / Profissional de saúde] -->|pergunta + paciente_id| B(LangGraph: Fluxo de Decisão Clínica)
    B --> C[Prontuário do paciente<br/>data/raw/synthetic_patients.json]
    B --> D[Pipeline RAG - LangChain]
    D --> E[Base de conhecimento FAISS<br/>protocolos autorizados por paciente]
    D --> F[LLM customizado<br/>Qwen2.5-1.5B + LoRA fine-tuned em MedQuAD]
    B --> G[Guardrails<br/>validação humana / bloqueio de prescrição direta]
    B --> J[Laudo clínico em PDF<br/>outputs/reports/]
    B --> H[Auditoria<br/>outputs/audit_log.json]
    B --> I[Resumo final do atendimento]
    I --> A
```

## 2. Fluxo de decisão clínica (LangGraph)

Implementado em `src/langgraph_flows/clinical_flow.py`.

```mermaid
stateDiagram-v2
    [*] --> receber_paciente
    receber_paciente --> verificar_exames_pendentes
    verificar_exames_pendentes --> verificar_alertas
    verificar_alertas --> emitir_alerta: tem_alerta_critico = true
    verificar_alertas --> sugerir_tratamento: tem_alerta_critico = false
    emitir_alerta --> sugerir_tratamento
    sugerir_tratamento --> gerar_laudo
    gerar_laudo --> finalizar
    finalizar --> [*]

    note right of receber_paciente
        Consulta o prontuário estruturado
        (src/langchain_pipeline/patient_records.py)
    end note

    note right of emitir_alerta
        Registra alerta crítico via AuditLogger
        (ex.: PROT-010 - alertas críticos)
    end note

    note right of gerar_laudo
        Gera o laudo clínico formatado
        (src/langchain_pipeline/report_generator.py),
        seguindo o modelo em
        data/raw/synthetic_laudo_template.md
    end note

    note right of sugerir_tratamento
        Chama o pipeline RAG (LangChain):
        retrieval de protocolos + contexto do
        paciente + LLM fine-tuned + guardrails
        de validação humana
    end note
```

## 3. Pipeline RAG (LangChain) — detalhe do nó `sugerir_tratamento`

Implementado em `src/langchain_pipeline/rag_chain.py`.

```mermaid
sequenceDiagram
    participant Flow as LangGraph (sugerir_tratamento)
    participant RAG as MedicalAssistantRAG
    participant FAISS as FAISS (protocolos internos)
    participant LLM as MedicalAssistantLLMClient
    participant Guard as Guardrails (safety_rules.py)
    participant Audit as AuditLogger

    Flow->>RAG: ask(pergunta, paciente_id)
    RAG->>FAISS: busca semântica + termos clínicos
    FAISS-->>RAG: protocolos autorizados do paciente
    RAG->>RAG: monta prompt (protocolos + contexto do paciente)
    RAG->>LLM: ask(system_prompt, user_message)
    LLM->>Audit: log(prompt, resposta, fontes)
    LLM-->>RAG: resposta gerada
    RAG-->>Flow: resposta + fontes citadas
    Flow->>Flow: ancora conduta no protocolo principal
    Flow->>Guard: enforce_human_validation(conduta)
    Guard-->>Flow: resposta com disclaimer + avisos
```

## 4. Componentes e responsabilidades

| Componente           | Arquivo                                      | Responsabilidade                                                                 |
| -------------------- | -------------------------------------------- | -------------------------------------------------------------------------------- |
| Fine-tuning (QLoRA)  | `src/fine_tuning/train.py`                   | Adapta o Qwen2.5-1.5B-Instruct aos dados do MedQuAD (protocolos/FAQs sintéticos) |
| Base de conhecimento | `src/langchain_pipeline/knowledge_base.py`   | Índice FAISS sobre os protocolos internos sintéticos (embeddings CPU)            |
| Prontuários          | `src/langchain_pipeline/patient_records.py`  | Consulta estruturada aos dados fictícios de pacientes                            |
| Cliente LLM          | `src/langchain_pipeline/llm_client.py`       | Chamada ao LLM (Groq/fine-tuned) com fallback seguro                             |
| RAG                  | `src/langchain_pipeline/rag_chain.py`        | Orquestra retrieval + contexto do paciente + geração + citação de fontes         |
| Fluxo de decisão     | `src/langgraph_flows/clinical_flow.py`       | Orquestra exames pendentes → alertas → conduta protocolar → laudo (LangGraph)    |
| Laudo/PDF            | `src/langchain_pipeline/report_generator.py` | Gera o documento estruturado e exporta o PDF                                     |
| Guardrails           | `src/guardrails/safety_rules.py`             | Bloqueia linguagem de prescrição direta e garante disclaimer de validação humana |
| Auditoria            | `src/guardrails/audit_logger.py`             | Log JSON de toda interação (explicabilidade/rastreabilidade)                     |
