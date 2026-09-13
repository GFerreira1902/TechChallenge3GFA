# Tech Challenge Fase 3: Assistente Virtual Médico com Fine-tuning e LangChain/LangGraph

**Integrante / Matrícula:** Guilherme Ferreira de Arruda / rm373210

**Repositório GitHub:** [TechChallenge3GFA](https://github.com/GFerreira1902/TechChallenge3GFA)

## Visão Geral do Projeto

Projeto da Fase 3 do Tech Challenge (Pós-Graduação IA para Devs - FIAP), independente da Fase 2
(`TechChallenge2GFA`). O objetivo é construir um assistente virtual médico treinado com dados
próprios (sintéticos/anonimizados) do hospital, capaz de:

- Responder dúvidas clínicas de médicos com base em protocolos internos;
- Sugerir procedimentos com base em protocolos e histórico do paciente;
- Gerar automaticamente um laudo clínico formatado a partir do atendimento;
- Orquestrar fluxos de decisão automatizados e seguros (verificação de exames pendentes,
  sugestão de tratamento, emissão de alertas) via **LangGraph**;
- Garantir explicabilidade (fonte da informação) e auditoria (logging) de cada resposta.

> Repositório desvinculado da Fase 2: sem histórico de git compartilhado e remote próprio
> (`TechChallenge2GFA`), sem nenhum arquivo específico do domínio anterior (classificação de
> câncer de mama / GA). O padrão de explicabilidade/auditoria da Fase 2
> (`docs/reference/fase2_llm_explainer.py`) foi reaproveitado e generalizado em
> `src/langchain_pipeline/llm_client.py` (cliente Groq com fallback seguro/mock mode) e
> `src/guardrails/audit_logger.py` (log de auditoria em JSON de cada interação).

## Arquitetura e Tecnologias

- **LLM base:** `Qwen/Qwen2.5-1.5B-Instruct`, com fine-tuning **QLoRA** (4-bit NF4 +
  LoRA de baixo rank via `peft`/`transformers`/`bitsandbytes`), rodando em GPU de 4GB
  (GTX 1650) com proteções de memória (ver `src/fine_tuning/train.py`).
- **Dataset:** [MedQuAD](https://github.com/abachaa/MedQuAD) (via Hugging Face
  `lavita/MedQuAD`), limpo e filtrado (`src/fine_tuning/prepare_dataset.py`), mais
  protocolos/prontuários 100% sintéticos e fictícios (`data/raw/synthetic_*.json`).
- **Orquestração:** `LangChain` (RAG sobre os protocolos via FAISS) + `LangGraph`
  (fluxo de decisão clínica em `src/langgraph_flows/clinical_flow.py`).
- **LLM customizado no pipeline:** `src/langchain_pipeline/local_llm_client.py` carrega
  o modelo base + adaptador LoRA fine-tuned localmente (é a LLM que o RAG usa por padrão).
  `src/langchain_pipeline/llm_client.py` (Groq, reaproveitado da Fase 2) fica disponível
  como alternativa/fallback.
- **Segurança:** `src/guardrails/safety_rules.py` bloqueia linguagem de prescrição direta
  e garante disclaimer de validação humana; `src/guardrails/audit_logger.py` audita
  toda interação em `outputs/audit_log.json`.
- **Interface:** CLI (scripts em `src/`); ver `src/demo.py` para um roteiro completo.
- **Geração de documentos:** `src/langchain_pipeline/report_generator.py` gera um laudo
  clínico formatado (modelo de referência em `data/raw/synthetic_laudo_template.md`),
  já integrado como etapa do fluxo LangGraph.
- **Qualidade:** `Pytest` (31 testes cobrindo RAG, LangGraph, guardrails, geração de
  laudo e prontuários).

## Estrutura do Projeto

```
TechChallenge3GFA/
├── data/
│   ├── raw/synthetic_protocols.json      # Protocolos internos fictícios (base do RAG)
│   ├── raw/synthetic_patients.json       # Prontuários fictícios
│   ├── raw/synthetic_laudo_template.md   # Modelo de laudo clínico (referencia)
│   └── processed/                        # MedQuAD limpo/amostrado (gerado, não versionado)
├── docs/
│   ├── diagrama_fluxo.md      # Diagrama do fluxo LangChain/LangGraph (Mermaid)
│   └── reference/             # Material de referência da Fase 2
├── outputs/
│   ├── models/                 # Adaptadores LoRA (gerado, não versionado)
│   ├── reports/                # PDFs de laudos gerados (gerado, não versionado)
│   └── audit_log.json          # Log de auditoria (gerado, não versionado)
├── src/
│   ├── fine_tuning/
│   │   ├── prepare_dataset.py          # Limpa/filtra/amostra o MedQuAD
│   │   ├── train.py                    # Fine-tuning QLoRA do Qwen2.5-1.5B
│   │   └── compare_base_vs_finetuned.py  # Avaliação qualitativa base vs. fine-tuned
│   ├── langchain_pipeline/
│   │   ├── knowledge_base.py    # Índice FAISS sobre os protocolos
│   │   ├── patient_records.py   # Consulta ao prontuário sintético
│   │   ├── llm_client.py        # Cliente Groq (Fase 2, fallback)
│   │   ├── local_llm_client.py  # Cliente do LLM fine-tuned local (padrão do RAG)
│   │   ├── rag_chain.py         # Pipeline RAG completo
│   │   └── report_generator.py # Geração do laudo clínico formatado
│   ├── langgraph_flows/
│   │   └── clinical_flow.py     # Fluxo de decisão clínica (LangGraph)
│   ├── guardrails/
│   │   ├── safety_rules.py      # Bloqueio de prescrição direta / disclaimer
│   │   └── audit_logger.py      # Log de auditoria
│   └── demo.py                  # Roteiro de demonstração ponta a ponta
├── tests/                # 31 testes pytest
├── requirements.txt
└── README.md
```

## Instruções de Instalação e Execução

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> No Windows, se o console exibir acentos quebrados, rode antes:
> `chcp 65001` e `$env:PYTHONIOENCODING="utf-8"`.

### 1. Preparar o dataset (MedQuAD limpo/amostrado)

```powershell
python -m src.fine_tuning.prepare_dataset --max-samples 1500
```

### 2. Fine-tuning (QLoRA) — já treinado, adaptador em `outputs/models/`

```powershell
python -m src.fine_tuning.train --num-train-epochs 1 --save-steps 30 `
    --output-dir outputs/models/qwen2.5-1.5b-lora-medquad-v2
```

> Requer GPU. Ajustado para GPUs de 4GB (batch=1, 4-bit, cap de VRAM em 80%). Em
> hardware mais limitado, feche outros apps que usam a GPU antes de rodar.

### 3. Avaliar o modelo (base vs. fine-tuned)

```powershell
python -m src.fine_tuning.compare_base_vs_finetuned `
    --adapter-path outputs/models/qwen2.5-1.5b-lora-medquad-v2/final_adapter
```

### 4. Testar o assistente (RAG) isoladamente

```powershell
python -m src.langchain_pipeline.rag_chain
```

### 5. Testar o fluxo de decisão clínica (LangGraph)

```powershell
python -m src.langgraph_flows.clinical_flow
```

Sem argumentos, o fluxo sorteia um paciente fictício e cria uma pergunta de
acordo com o diagnóstico dele a cada execução. Para repetir um cenário:

```powershell
python -m src.langgraph_flows.clinical_flow --paciente-id PAC-002
```

### 6. Rodar a demonstração completa (recomendado para conferir tudo de uma vez)

```powershell
python -m src.demo
```

Cobre, em sequência: comparação da LLM personalizada, fluxo automatizado do
LangGraph, uma pergunta clínica contextualizada via RAG, geração de laudo em PDF
e os logs de auditoria. Os PDFs ficam em `outputs/reports/`.

## Rodando a Suíte de Testes

```bash
python -m pytest tests/ -v
```

31 testes, cobrindo `rag_chain`, `clinical_flow`, `report_generator`, `safety_rules`,
`patient_records`, `llm_client` e `audit_logger` (todos com mocks — não exigem GPU nem rede).

## Entregáveis da Fase 3

- **Repositório GitHub Oficial:** [Acessar Código Fonte](https://github.com/GFerreira1902/TechChallenge3GFA)
- Código-fonte: pipeline de fine-tuning, integração LangChain, fluxos LangGraph;
- Dataset anonimizado/sintético;
- Relatório técnico: processo de fine-tuning, descrição do assistente, diagrama do fluxo
  LangChain/LangGraph, avaliação do modelo;
- Vídeo de demonstração (até 15 min).

---

_Projeto elaborado para a Fase 3 do Tech Challenge - Pós-Graduação IA para Devs FIAP._
