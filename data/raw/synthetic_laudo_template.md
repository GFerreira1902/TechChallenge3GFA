# Modelo de Laudo Clínico (fictício)

> Este é um modelo/template de referência, escrito por nós para representar o
> tipo de documento interno de um hospital ("modelos de laudos, receitas e
> procedimentos internos" citado no enunciado da Fase 3). É usado como
> exemplo de formato para a geração automática de laudos pelo assistente
> (`src/langchain_pipeline/report_generator.py`). Não representa um documento
> real de nenhum paciente ou hospital.

## Estrutura esperada de um laudo

```
LAUDO CLÍNICO
Paciente: <identificador do paciente>
Data/hora: <timestamp>

1. Diagnóstico principal
<diagnóstico do paciente, em uma ou duas frases>

2. Exames pendentes ou realizados
<lista de exames relevantes ao caso>

3. Conduta sugerida
<conduta com base nos protocolos internos consultados, de forma objetiva>

4. Protocolos internos consultados (fonte)
<lista de identificadores de protocolo, ex.: PROT-001>

5. Alertas ativos
<qualquer alerta crítico associado ao paciente>

---
Documento gerado automaticamente por assistente de IA. Não substitui a
avaliação clínica de um profissional de saúde responsável. Requer validação
humana antes de qualquer conduta.
```

## Exemplo preenchido (fictício, apenas para referência de estilo)

```
LAUDO CLÍNICO
Paciente: PAC-007
Data/hora: 2026-01-10T14:32:00

1. Diagnóstico principal
Pós-operatório de apendicectomia, sem intercorrências, 2º dia.

2. Exames pendentes ou realizados
Nenhum exame pendente no momento.

3. Conduta sugerida
Manter analgesia conforme prescrição vigente; reavaliar ferida operatória;
avaliar critérios de alta conforme protocolo PROT-006 (alta hospitalar segura).

4. Protocolos internos consultados (fonte)
PROT-006

5. Alertas ativos
Apto para avaliação de alta - reconciliação medicamentosa pendente.

---
Documento gerado automaticamente por assistente de IA. Não substitui a
avaliação clínica de um profissional de saúde responsável. Requer validação
humana antes de qualquer conduta.
```
