"""Cliente LLM local: carrega o Qwen2.5-1.5B-Instruct + adaptador LoRA fine-tuned
em MedQuAD (v2) para gerar as respostas do assistente. Esta e a "LLM customizada"
exigida no enunciado da Fase 3 - diferente de src/langchain_pipeline/llm_client.py
(Groq, reaproveitado da Fase 2), que fica disponivel como alternativa/fallback.

Interface compativel com MedicalAssistantLLMClient (mesmo metodo `ask`), para que
o RAG chain possa usar qualquer um dos dois sem mudar de assinatura.
"""
from __future__ import annotations

import os
from typing import Any, ClassVar, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.guardrails.audit_logger import AuditLogger

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = "outputs/models/qwen2.5-1.5b-lora-medquad-v2/final_adapter"
GPU_MEMORY_FRACTION = 0.80


class LocalFineTunedLLMClient:
    """Gera respostas localmente com o modelo fine-tuned (sem depender de API externa)."""

    # Modelo/tokenizer sao carregados uma unica vez e compartilhados entre instancias,
    # pois carregar o Qwen2.5 (mesmo em 4-bit) e custoso.
    _model: ClassVar[Optional[Any]] = None
    _tokenizer: ClassVar[Optional[Any]] = None
    _loaded_adapter_path: ClassVar[Optional[str]] = None

    def __init__(
        self,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        audit_logger: Optional[AuditLogger] = None,
        max_new_tokens: int = 300,
    ):
        self.adapter_path = adapter_path
        self.audit_logger = audit_logger or AuditLogger()
        self.max_new_tokens = max_new_tokens
        self._ensure_model_loaded()

    @classmethod
    def _ensure_model_loaded(cls) -> None:
        if cls._model is not None:
            return

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)

        cls._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
        )
        cls._model = PeftModel.from_pretrained(base_model, DEFAULT_ADAPTER_PATH)
        cls._model.eval()
        cls._loaded_adapter_path = DEFAULT_ADAPTER_PATH

    def ask(
        self,
        system_prompt: str,
        user_message: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Gera a resposta localmente e audita a interacao (igual ao cliente Groq)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                no_repeat_ngram_size=4,
                repetition_penalty=1.15,
            )

        response = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        self.audit_logger.log(
            prompt=user_message,
            response=response,
            metadata={
                **(metadata or {}),
                "model": f"{BASE_MODEL}+LoRA({self._loaded_adapter_path})",
                "simulated": False,
            },
        )
        return response
