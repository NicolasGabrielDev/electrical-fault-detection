# Detecção e Classificação de Falhas Elétricas

Estudo de caso de mineração de dados para classificação supervisionada de falhas em linhas de transmissão elétrica usando medições de corrente e tensão.

Este projeto baixa o dataset do Kaggle `esathyaprakash/electrical-fault-detection-and-classification`, executa análise exploratória, aplica engenharia de features baseada em física do sistema trifásico, cria pipelines de pré-processamento, treina modelos de classificação, compara métricas e avalia o desempenho por classe de falha.

Versão em inglês: [readme.MD](readme.MD)

## Problema

Linhas de transmissão elétrica podem operar normalmente ou sob diferentes condições de falha. O objetivo é usar variáveis elétricas medidas para classificar a condição do sistema e identificar o tipo de falha.

Este é um problema de classificação supervisionada multiclasse.

## Dataset

- Fonte: Kaggle
- Dataset: `esathyaprakash/electrical-fault-detection-and-classification`
- Arquivo selecionado: `classData.csv`
- Variáveis de entrada: correntes e tensões de fase (`Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`)
- Variável-alvo: `fault_type`

No `classData.csv`, as colunas `G`, `C`, `B` e `A` indicam o envolvimento da falha. O notebook combina essas colunas em uma única variável-alvo multiclasse.

### Tipos de falha

| Saída [G C B A] | Tipo | Descrição |
| --- | --- | --- |
| [0 0 0 0] | Sem falha | Operação normal |
| [1 0 0 1] | LG | Fase A e terra |
| [0 0 1 1] | LL | Fases A e B |
| [1 0 1 1] | LLG | Fases A, B e terra |
| [0 1 1 1] | LLL | Três fases |
| [1 1 1 1] | LLLG | Três fases e terra |

## Engenharia de Features

Além das seis variáveis originais, o notebook cria features derivadas da física do sistema trifásico:

| Feature | Descrição |
| --- | --- |
| `I_seq_zero` | Média de Ia, Ib, Ic — componente de sequência zero da corrente, indicador do envolvimento do terra |
| `V_seq_zero` | Média de Va, Vb, Vc — componente de sequência zero da tensão |
| `I_unbalance` | Desvio padrão de Ia, Ib, Ic — desequilíbrio entre fases |
| `V_unbalance` | Desvio padrão de Va, Vb, Vc — desequilíbrio de tensão entre fases |
| `I_magnitude` | Magnitude total da corrente (norma L2) |
| `V_magnitude` | Magnitude total da tensão (norma L2) |

Essas features melhoraram significativamente a separação entre classes que diferem apenas pelo envolvimento do terra, como LLL e LLLG.

## Modelos

O notebook treina e compara:

- Árvore de Decisão
- Floresta Aleatória
- Gradient Boosting
- Regressão Logística

O fluxo usa pipelines e column transformers do scikit-learn para evitar vazamento de dados. O pré-processamento é ajustado exclusivamente nos dados de treino.

## Resultados

Resumo dos resultados com engenharia de features:

| Modelo | Acurácia | F1-score ponderado | F1 médio CV |
| --- | ---: | ---: | ---: |
| Floresta Aleatória | 0,9981 | 0,9981 | 0,9967 |
| Gradient Boosting | 0,9962 | 0,9962 | 0,9940 |
| Árvore de Decisão | 0,9949 | 0,9949 | 0,9924 |
| Regressão Logística | 0,6682 | 0,6637 | 0,6476 |

A Floresta Aleatória obteve o melhor desempenho. O desvio padrão do F1 em validação cruzada foi de 0,0009, indicando alta estabilidade.

A Regressão Logística teve desempenho inferior porque o problema exige fronteiras de decisão não lineares entre as classes de falha.

## Avaliação por Classe

O notebook exibe matrizes de confusão e relatórios de classificação por classe para cada modelo. Um limiar de recall mínimo configurável (`MINIMUM_RECALL_PER_CLASS`, padrão 0,90) gera alertas automáticos para classes com desempenho abaixo do esperado.

## Arquivos do Projeto

- [electrical_fault_classification.ipynb](electrical_fault_classification.ipynb): notebook completo e executável.
- [model_comparison_results.csv](model_comparison_results.csv): métricas de comparação dos modelos.
- [requirements.txt](requirements.txt): dependências Python.

## Como Executar

Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o notebook:

```bash
jupyter notebook electrical_fault_classification.ipynb
```

Ou execute pelo terminal:

```bash
jupyter nbconvert --to notebook --execute electrical_fault_classification.ipynb --inplace
```

## Observações

O dataset é baixado com `kagglehub`. Se o ambiente exigir autenticação do Kaggle, configure as credenciais localmente sem versioná-las no repositório.

## Limitações

- A análise usa um único dataset público com medições simuladas.
- Não foi feita validação externa com dados reais de campo.
- O modelo classifica o tipo de falha em um instante isolado — não detecta o momento de início da falha nem sua localização na linha.

## Possíveis Melhorias

- Validar com dados reais de operação de subestações.
- Adicionar localização da falha na linha como variável de saída.
- Avaliar calibração de probabilidade dos modelos.
- Incorporar janelamento temporal para detecção de falhas incipientes.