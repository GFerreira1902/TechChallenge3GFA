import pytest
import os
from unittest.mock import patch, MagicMock
from src.llm_explainer import MedicalLLMExplainer

def test_llm_initialization_no_key():
    """Valida se a ausência de chave de autenticação ativa o fallback (mock_mode)."""
    with patch.dict(os.environ, {'GROQ_API_KEY': ''}, clear=True):
        explainer = MedicalLLMExplainer(api_key=None)
        assert explainer.mock_mode is True
        assert explainer.client is None

@patch('src.llm_explainer.Groq')
def test_llm_generate_explanation_success(mock_groq_class, tmp_path):
    """Garante que a construção do laudo humanizado retorna perfeitamente os inputs do LLaMA."""
    
    # Mock da resposta estática do Model Complection
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "Achado compatível com morfologia normal. Próximos passos sugeridos..."
    mock_client.chat.completions.create.return_value.choices = [mock_response]
    mock_groq_class.return_value = mock_client
    
    # Criação de um local temporário para não sujar os relatórios verdadeiros
    history_file = tmp_path / "mock_history.json"
    
    explainer = MedicalLLMExplainer(api_key="mock_key", history_path=str(history_file))
    
    resposta = explainer.generate_explanation(
        prediction_class=0, 
        confidence=98.5, 
        top_features="radius=12.2, perimeter=88.5"
    )
    
    assert "compatível com morfologia normal" in resposta
    assert os.path.exists(str(history_file)), "O log do painel não foi guardado corretamente."
