"""Compara respostas do modelo base vs. modelo fine-tuned (adaptador LoRA) para
as mesmas perguntas, como evidencia qualitativa para o relatorio tecnico.

Uso:
    python -m src.fine_tuning.compare_base_vs_finetuned --adapter-path outputs/models/qwen2.5-1.5b-lora-medquad-v2/final_adapter
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.fine_tuning.train import BASE_MODEL

DEFAULT_ADAPTER_PATH = "outputs/models/qwen2.5-1.5b-lora-medquad/final_adapter"

SYSTEM_PROMPT = (
    "Você é um assistente virtual médico, treinado com protocolos internos e "
    "perguntas frequentes de profissionais de saúde. Responda de forma clara, "
    "objetiva e baseada em evidências. Você nunca prescreve tratamentos "
    "diretamente: toda sugestão deve ser validada por um profissional de saúde "
    "responsável antes de qualquer conduta clínica."
)

QUESTIONS = [
    "What is (are) Appendicitis ?",
    "What are the treatments for high blood pressure ?",
    "What are the symptoms of sepsis ?",
]


def generate(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, do_sample=False, temperature=None, top_p=None
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map={"": 0}
    )

    for question in QUESTIONS:
        print("=" * 80)
        print(f"PERGUNTA: {question}\n")

        base_answer = generate(base_model, tokenizer, question)
        print(f"--- BASE (sem fine-tuning) ---\n{base_answer}\n")

    print(f"Carregando adaptador LoRA fine-tuned de {args.adapter_path}...")
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)

    for question in QUESTIONS:
        print("=" * 80)
        print(f"PERGUNTA: {question}\n")
        ft_answer = generate(ft_model, tokenizer, question)
        print(f"--- FINE-TUNED (MedQuAD) ---\n{ft_answer}\n")


if __name__ == "__main__":
    main()
