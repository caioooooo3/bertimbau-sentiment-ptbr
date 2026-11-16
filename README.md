# BERTimbau Sentiment Analysis PT-BR 🇧🇷

Fine-tuning do **BERTimbau Base** para classificação binária de sentimentos em avaliações de e-commerce em português brasileiro.

[🤗 Modelo no Hugging Face](https://huggingface.co/caio16dev/bertimbau-sentiment-ptbr)

## Resultado principal

| Métrica | Teste |
|---|---:|
| Accuracy | **96,92%** |
| Precision | **97,95%** |
| Recall | **97,68%** |
| F1 | **97,81%** |
| Macro F1 | **96,29%** |

O modelo foi avaliado em um conjunto de teste mantido separado durante o desenvolvimento.

## Visão geral

O projeto cobre um pipeline completo de NLP:

1. carregamento do B2W-Reviews01;
2. definição dos rótulos de sentimento;
3. limpeza de textos vazios e duplicados;
4. remoção de textos duplicados com labels conflitantes;
5. split estratificado em treino, validação e teste;
6. análise do comprimento dos textos;
7. tokenização com o tokenizer oficial do BERTimbau;
8. fine-tuning com `BertForSequenceClassification`;
9. seleção do melhor checkpoint por Macro F1;
10. avaliação final e publicação no Hugging Face.

## Dataset

Fonte: **B2W-Reviews01**.

Dataset original: **132.373 avaliações**.

Definição do target:

- 1–2 estrelas → `NEGATIVO`
- 3 estrelas → removidas por representarem uma região ambígua
- 4–5 estrelas → `POSITIVO`

Após a limpeza:

- **110.881 reviews**
- **78.280 positivos (70,60%)**
- **32.601 negativos (29,40%)**

Split estratificado:

- treino: **88.704**
- validação: **11.088**
- teste: **11.089**

## Tokenização

Modelo base: `neuralmind/bert-base-portuguese-cased`.

A análise de uma amostra de 10.000 reviews mostrou:

- mediana: 26 tokens
- P90: 70
- P95: 97
- P99: 179

Foi escolhido `max_length=128` para equilibrar cobertura do conteúdo e custo computacional.

## Fine-tuning

Configuração principal:

- learning rate: `2e-5`
- epochs: `2`
- train batch size: `16`
- eval batch size: `32`
- weight decay: `0.01`
- padding dinâmico
- melhor checkpoint selecionado por **Macro F1**

### Validação

| Métrica | Resultado |
|---|---:|
| Accuracy | 97,25% |
| Precision | 98,22% |
| Recall | 97,88% |
| F1 | 98,05% |
| Macro F1 | 96,69% |

### Teste por classe

| Classe | Precision | Recall | F1 |
|---|---:|---:|---:|
| NEGATIVO | 94,45% | 95,09% | 94,77% |
| POSITIVO | 97,95% | 97,68% | 97,81% |

Matriz de confusão no teste:

```text
[[3100, 160],
 [ 182, 7647]]
```

## Uso rápido

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="caio16dev/bertimbau-sentiment-ptbr"
)

print(classifier("Produto excelente, chegou rápido e funciona perfeitamente."))
```

Exemplo de saída:

```text
[{'label': 'POSITIVO', 'score': 0.9995}]
```

## Estrutura do repositório

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks
│   └── README.md
└── src
    ├── inference.py
    └── train.py
```

## Reproduzir o treinamento

Instale as dependências:

```bash
pip install -r requirements.txt
```

Depois:

```bash
python src/train.py
```

> Para fine-tuning completo, é recomendado utilizar GPU.

## Limitações

- O modelo foi treinado em avaliações de e-commerce e pode apresentar desempenho diferente em outros domínios.
- O problema é binário e não possui classe neutra.
- Avaliações de 3 estrelas foram removidas.
- Os scores do classificador não passaram por um estudo específico de calibração.

## Tecnologias

Python · PyTorch · Hugging Face Transformers · Hugging Face Datasets · Scikit-learn · BERT · BERTimbau

## Autor

**Caio Vinicius Paes da Silva**

- Hugging Face: [caio16dev](https://huggingface.co/caio16dev)
- GitHub: [caioooooo3](https://github.com/caioooooo3)
- LinkedIn: [caio-viniciuss](https://www.linkedin.com/in/caio-viniciuss/)
