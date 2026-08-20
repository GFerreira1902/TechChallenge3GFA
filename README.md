# Tech Challenge Fase 3: Assistente Virtual Médico com Fine-tuning e LangChain/LangGraph

**Integrante / Matrícula:** Guilherme Ferreira de Arruda / rm373210

## Visão Geral do Projeto

Projeto da Fase 3 do Tech Challenge (Pós-Graduação IA para Devs - FIAP), independente da Fase 2
(`TechChallenge2GFA`). O objetivo é construir um assistente virtual médico treinado com dados
próprios (sintéticos/anonimizados) do hospital, capaz de:

- Responder dúvidas clínicas de médicos com base em protocolos internos;
- Sugerir procedimentos com base em protocolos e histórico do paciente;
- Orquestrar fluxos de decisão automatizados e seguros (verificação de exames pendentes,
  sugestão de tratamento, emissão de alertas) via **LangGraph**;
- Garantir explicabilidade (fonte da informação) e auditoria (logging) de cada resposta.

> Repositório desvinculado da Fase 2: sem histórico de git compartilhado, remote próprio e
> nenhum arquivo específico do domínio anterior (classificação de câncer de mama / GA).
> O arquivo `docs/reference/fase2_llm_explainer.py` foi mantido apenas como referência para
> uma eventual decisão de reaproveitamento do padrão de explicabilidade/logging.

## Arquitetura e Tecnologias (planejado)

- **LLM base:** modelo pequeno open-weight (ex.: `Llama-3.2-1B/3B-Instruct`, `Phi-3-mini`) com
  fine-tuning via **LoRA/QLoRA** (`peft` + `transformers` + `bitsandbytes`).
- **Orquestração:** `LangChain` (RAG sobre protocolos/prontuários sintéticos) + `LangGraph`
  (fluxo de decisão clínica).
- **Dataset:** [MedQuAD](https://github.com/abachaa/MedQuAD) (perguntas e respostas de saúde),
  complementado por protocolos/laudos sintéticos fictícios.
- **Segurança:** guardrails de atuação (sem prescrição direta), logging estruturado para
  auditoria.
- **Interface:** a definir (CLI/Streamlit).
- **Qualidade:** `Pytest`.

## Estrutura do Projeto

```
TechChallenge3GFA/
├── data/
│   ├── raw/            # Dados brutos (MedQuAD, protocolos sintéticos)
│   └── processed/      # Dados tratados/anonimizados prontos para fine-tuning e RAG
├── docs/
│   └── reference/       # Material de referência da Fase 2 (decisão de reaproveitamento pendente)
├── notebooks/          # Exploração e experimentação
├── outputs/
│   ├── models/         # Adaptadores LoRA / checkpoints
│   └── figures/        # Gráficos de avaliação
├── src/
│   ├── fine_tuning/     # Pipeline de preprocessing + fine-tuning (LoRA/QLoRA)
│   ├── langchain_pipeline/  # RAG, prompts, integração com o LLM customizado
│   ├── langgraph_flows/     # Definição dos grafos de decisão clínica
│   └── guardrails/          # Regras de segurança, validação e logging
├── tests/
├── requirements.txt
└── README.md
```

## Instruções de Instalação e Execução

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

> Instruções de execução do pipeline de fine-tuning, do assistente e dos fluxos LangGraph
> serão detalhadas conforme o projeto avançar.

## Rodando a Suíte de Validação

```bash
python -m pytest tests/ -v
```

## Entregáveis da Fase 3

- Código-fonte: pipeline de fine-tuning, integração LangChain, fluxos LangGraph;
- Dataset anonimizado/sintético;
- Relatório técnico: processo de fine-tuning, descrição do assistente, diagrama do fluxo
  LangChain/LangGraph, avaliação do modelo;
- Vídeo de demonstração (até 15 min).

---

_Projeto elaborado para a Fase 3 do Tech Challenge - Pós-Graduação IA para Devs FIAP._