"""Prepara o dataset MedQuAD (Hugging Face `lavita/MedQuAD`) para fine-tuning por instrução.

Como não há acesso a protocolos reais do hospital, o MedQuAD (perguntas e respostas
clínicas de fontes públicas como NIH/MedlinePlus/GARD) é usado como substituto
sintético/anonimizado, conforme sugerido no enunciado da Fase 3.

Gera:
- data/processed/medquad_clean.jsonl   -> dataset completo limpo (todas as amostras)
- data/processed/medquad_train.jsonl   -> subconjunto de treino (amostrado)
- data/processed/medquad_val.jsonl     -> subconjunto de validação (amostrado)

Cada linha é um exemplo no formato "messages" (system/user/assistant), pronto para
`tokenizer.apply_chat_template` no script de fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import random

from datasets import load_dataset

SYSTEM_PROMPT = (
    "Você é um assistente virtual médico, treinado com protocolos internos e "
    "perguntas frequentes de profissionais de saúde. Responda de forma clara, "
    "objetiva e baseada em evidências. Você nunca prescreve tratamentos "
    "diretamente: toda sugestão deve ser validada por um profissional de saúde "
    "responsável antes de qualquer conduta clínica."
)

MIN_ANSWER_CHARS = 20
MAX_ANSWER_CHARS = 1500
MIN_QUESTION_CHARS = 8


def clean_examples(dataset) -> list[dict]:
    """Filtra e formata os pares pergunta/resposta em exemplos de instrução.

    Usa acesso vetorizado por coluna (em vez de iterar linha a linha) porque a
    decodificação Arrow por linha do `datasets` é muito lenta em 47k+ registros.
    """
    questions = dataset["question"]
    answers = dataset["answer"]
    question_ids = dataset["question_id"]
    document_sources = dataset["document_source"]
    document_urls = dataset["document_url"]
    question_types = dataset["question_type"]

    seen_ids = set()
    examples = []

    for i in range(len(dataset)):
        question = (questions[i] or "").strip()
        answer = (answers[i] or "").strip()
        question_id = question_ids[i]

        if not question or not answer:
            continue
        if question_id in seen_ids:
            continue
        if len(question) < MIN_QUESTION_CHARS:
            continue
        if not (MIN_ANSWER_CHARS <= len(answer) <= MAX_ANSWER_CHARS):
            continue

        seen_ids.add(question_id)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
                "source": {
                    "document_source": document_sources[i],
                    "document_url": document_urls[i],
                    "question_type": question_types[i],
                },
            }
        )

    return examples


def write_jsonl(path: str, examples: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=4000, help="Tamanho do subconjunto de treino+validação amostrado")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Proporção de validação dentro do subconjunto amostrado")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Carregando lavita/MedQuAD do Hugging Face Hub...")
    dataset = load_dataset("lavita/MedQuAD")["train"]

    print(f"Registros brutos: {len(dataset)}")
    examples = clean_examples(dataset)
    print(f"Registros apos limpeza/deduplicacao: {len(examples)}")

    clean_path = os.path.join(args.output_dir, "medquad_clean.jsonl")
    write_jsonl(clean_path, examples)
    print(f"Dataset completo limpo salvo em {clean_path}")

    random.shuffle(examples)
    sample = examples[: args.max_samples]
    val_size = max(1, int(len(sample) * args.val_ratio))
    val_sample = sample[:val_size]
    train_sample = sample[val_size:]

    train_path = os.path.join(args.output_dir, "medquad_train.jsonl")
    val_path = os.path.join(args.output_dir, "medquad_val.jsonl")
    write_jsonl(train_path, train_sample)
    write_jsonl(val_path, val_sample)

    print(f"Treino: {len(train_sample)} exemplos -> {train_path}")
    print(f"Validacao: {len(val_sample)} exemplos -> {val_path}")


if __name__ == "__main__":
    main()
