import os
import json
from datetime import datetime
from groq import Groq

class MedicalLLMExplainer:
    def __init__(self, api_key=None, history_path='../outputs/llm_history.json'):
        """
        Inicializa o cliente LLM focado em explicabilidade humanizada.
        Utilizamos a Groq (com o modelo Llama 3) para geração ultrarrápida.
        """
        # Carrega a chave da variavel de ambiente ou via argumento
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.mock_mode = False
        
        if not self.api_key:
            # Fallback seguro ativado: Se não houver chave, usa um gerador local simulado
            self.mock_mode = True
            self.client = None
        else:    
            # Timeout de processamento (20s max)
            self.client = Groq(api_key=self.api_key, timeout=20.0)
            
        self.model = "llama-3.3-70b-versatile"
        
        # Histórico de logging de outputs 
        self.history_path = history_path
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        
    def _save_to_history(self, prompt, response, model_prediction, metadata=None):
        """
        Exporta inputs de prompt, predições atreladas e saídas para JSON (Dataset local).
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_prediction": model_prediction,
            "metadata": metadata or {},
            "prompt_enviado": prompt,
            "resposta_llm": response
        }
        
        history = []
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r', encoding='utf-8') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
                    
        history.append(entry)
        
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
            
    def generate_explanation(self, prediction_class, confidence, top_features):
        """
        Recebe a predição do modelo de Machine Learning (Random Forest otimizado)
        e gera um laudo explicativo sensível.
        
        prediction_class: 1 (Provável Maligno) ou 0 (Provável Benigno)
        confidence: a probabilidade(%) que o RF atribuiu.
        top_features: Dicionário ou texto informando as features que mais pesaram.
        """
        
        # Diretrizes do Prompt (Linguagem Sensível/Humana e Ausência Determinista)
        system_prompt = (
            "Você é um Assistente de Inteligência Artificial especializado na saúde da mulher. "
            "Sua função é ler os resultados de um modelo preditivo de câncer de mama "
            "(que usa dados celulares de mamografias) e traduzir esses números em um laudo humanizado, "
            "culturamente sensível e prático, voltado para apoiar o médico mastologista/ginecologista na tomada de decisão."
            "Diretrizes:\n"
            "1. NO INÍCIO DO LAUDO, destaque em negrito, em MAIÚSCULAS e de forma visual isolada o resultado principal da análise preditiva (Ex: **RESULTADO INDICATIVO: ACHADO SUSPEITO DE MALIGNIDADE**).\n"
            "2. NÃO dê diagnóstico definitivo, use termos como 'indicativo' ou 'rastreio compatível com'.\n"
            "3. Seja acolhedor e sensível às preocupações de saúde feminina.\n"
            "4. Explique brevemente o porquê da decisão baseando-se nas 'Principais Variáveis' fornecidas.\n"
            "5. Indique os próximos passos recomendados com base no resultado (ex: biópsia, novos exames, etc).\n"
            "6. NO FINAL DO LAUDO, assine obrigatoriamente e exatamente como: 'Atenciosamente, Assistente de Inteligência Artificial em Saúde da Mulher'."
        )
        
        diagnostico = "Achado suspeito de malignidade" if prediction_class == 1 else "Provável achado benigno"
        
        user_message = (
            f"O modelo de Machine Learning analisou os exames e retornou o seguinte:\n"
            f"- Classificação: {diagnostico}\n"
            f"- Confiança do Algoritmo: {confidence:.2f}%\n"
            f"- Principais Variáveis Celulares que pesaram nessa decisão: {top_features}\n\n"
            f"Com base nesses dados, gere o relatório de explicabilidade."
        )
        
        # Abordagem de Fallback Seguro para rodar sem API_KEY no computador do avaliador
        if self.mock_mode:
            resposta_llm = (
                f"*[MODO SIMULADO - API KEY AUSENTE]*\n\n"
                f"**Análise de Explicabilidade:**\n"
                f"Baseado na avaliação do Algoritmo Evolucionário, o rastreio é {diagnostico.lower()} com uma confiança de {confidence:.2f}%.\n"
                f"Fatores mais decisivos para esse resultado: {top_features}.\n\n"
                f"*Recomendação Clínica: Este é um laudo simulado devido à ausência de chave do Groq vinculada ao ambiente.*\n\n"
                f"Atenciosamente, Assistente de Inteligência Artificial em Saúde da Mulher."
            )
            
            self._save_to_history(
                prompt=user_message,
                response=resposta_llm,
                model_prediction=diagnostico,
                metadata={"confidence": confidence, "features": top_features, "simulated": True}
            )
            return resposta_llm
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=0.4, 
                max_tokens=800
            )
            
            resposta_llm = chat_completion.choices[0].message.content
            
            # Execução de logs
            self._save_to_history(
                prompt=user_message,
                response=resposta_llm,
                model_prediction=diagnostico,
                metadata={"confidence": confidence, "features": top_features}
            )
            
            return resposta_llm
            
        except Exception as e:
            return f"Erro ao contatar a API da Groq: {str(e)}"

if __name__ == "__main__":
    # Teste Rápido de execução
    print("Para usar o llm_explainer, importe a classe e insira sua GROQ_API_KEY no ambiente.")
    print("Ex: setx GROQ_API_KEY 'sua-chave'")