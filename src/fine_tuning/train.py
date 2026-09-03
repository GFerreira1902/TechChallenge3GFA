"""Fine-tuning por instrucao (QLoRA) do Qwen2.5-1.5B-Instruct sobre o dataset MedQuAD.

Ajustado para GPUs com pouca VRAM (ex.: GTX 1650, 4GB): quantizacao 4-bit (NF4),
LoRA de baixo rank, batch size 1 com gradient accumulation, gradient checkpointing
e sequencias curtas. O modelo base nunca e sobrescrito: apenas o adaptador LoRA
e salvo em outputs/models/.

Uso:
    python -m src.fine_tuning.train --max-steps 5      # smoke test rapido
    python -m src.fine_tuning.train                    # treino completo (config padrao)
"""
from __future__ import annotations

import argparse
import os

# Precisa ser definido antes do primeiro uso de CUDA pelo torch: reduz fragmentacao
# de memoria, o que ajuda a evitar picos de VRAM em GPUs pequenas (ex.: GTX 1650).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
IGNORE_INDEX = -100

# Fracao maxima da VRAM que o processo pode reservar. Em GPUs de 4GB (ex.: GTX 1650)
# compartilhadas com o desktop do Windows, isso evita que o treino trave o driver/SO
# inteiro: ao ultrapassar o limite, o PyTorch lanca OutOfMemoryError controlado.
GPU_MEMORY_FRACTION = 0.80


def build_example(tokenizer, messages: list[dict], max_seq_length: int) -> dict:
    """Tokeniza a conversa completa e mascara o rotulo (labels) do prompt.

    So calculamos loss sobre os tokens da resposta do assistente; system/user
    ficam com label = IGNORE_INDEX (nao entram na loss).
    """
    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_length)["input_ids"]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
    labels = labels[: len(full_ids)]

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def load_and_tokenize(
    tokenizer,
    data_files: dict,
    max_seq_length: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
):
    dataset = load_dataset("json", data_files=data_files)

    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    if max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )

    def _map_fn(example):
        return build_example(tokenizer, example["messages"], max_seq_length)

    tokenized = dataset.map(_map_fn, remove_columns=dataset["train"].column_names)
    return tokenized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/processed/medquad_train.jsonl")
    parser.add_argument("--val-file", default="data/processed/medquad_val.jsonl")
    parser.add_argument("--output-dir", default="outputs/models/qwen2.5-1.5b-lora-medquad")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Se > 0, sobrepoe num-train-epochs (util para smoke test)")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limita o numero de exemplos de treino")
    parser.add_argument("--max-eval-samples", type=int, default=20, help="Limita o numero de exemplos de validacao (avaliacao e lenta em GPUs pequenas)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-during-training", action="store_true", help="Avalia periodicamente durante o treino (lento); por padrao so avalia uma vez ao final")
    parser.add_argument("--gpu-memory-fraction", type=float, default=GPU_MEMORY_FRACTION, help="Fracao maxima da VRAM reservavel pelo processo (seguranca contra travamento do SO)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=0)
        print(f"Limite de VRAM do processo definido em {args.gpu_memory_fraction:.0%} do total da GPU.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files = {"train": args.train_file, "validation": args.val_file}
    tokenized = load_and_tokenize(
        tokenizer, data_files, args.max_seq_length, args.max_train_samples, args.max_eval_samples
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if args.eval_during_training else "no",
        eval_steps=args.save_steps if args.eval_during_training else None,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    final_dir = os.path.join(args.output_dir, "final_adapter")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("VRAM esgotada durante o treino. Salvando o ultimo checkpoint disponivel antes de encerrar...")
        trainer.model.save_pretrained(os.path.join(args.output_dir, "oom_partial_adapter"))
        raise

    print("Avaliacao final (amostra reduzida)...")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adaptador LoRA salvo em {final_dir}")


if __name__ == "__main__":
    main()
