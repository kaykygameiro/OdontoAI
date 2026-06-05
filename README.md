# OdontoAI - Classificacao de Imagens Odontologicas

OdontoAI e um MVP/prototipo academico educacional para classificacao de imagens odontologicas com redes neurais convolucionais, ResNet-50, Streamlit e Grad-CAM.

O projeto organiza um pipeline completo de montagem do dataset, limpeza, divisao em treino/validacao/teste, auditorias contra vazamento de dados, treinamento, avaliacao, geracao de figuras e interface web para inferencia.

## Avisos importantes

Este sistema tem finalidade academica e educacional. Ele nao realiza diagnostico clinico e nao substitui a avaliacao profissional de um cirurgiao-dentista.

Resultados do modelo devem ser interpretados com cautela, especialmente em cenarios reais, bases externas ou imagens com distribuicao diferente da usada no experimento.

## Classes avaliadas

| Classe interna | Descricao |
| --- | --- |
| Calculus | Tartaro / calculo dentario |
| Caries | Carie |
| Gingivitis | Gengivite |
| Hypodontia | Hipodontia |
| Mouth_Ulcer | Ulcera bucal |
| Tooth_Discoloration | Descoloracao dentaria |

## Arquitetura usada

O modelo final utiliza ResNet-50 com transfer learning.

O treinamento foi organizado em duas fases:

1. Treinamento inicial da camada final.
2. Ajuste fino da `layer4` e da camada final.

A interpretabilidade visual e feita com Grad-CAM, apenas como apoio qualitativo.

## Pipeline oficial

O pipeline final utiliza imagens originais do dataset. Pastas `augmented` do Kaggle foram ignoradas, e o dataset YOLO tambem foi ignorado porque era voltado a deteccao, nao classificacao.

Duplicatas exatas e conflitos foram removidos por MD5. Imagens visualmente semelhantes foram agrupadas por pHash, e o split foi feito por grupos visuais para reduzir risco de vazamento entre treino, validacao e teste.

Data augmentation e aplicado apenas no conjunto de treino. Validacao e teste nao usam augmentation.

Ordem oficial dos scripts:

```bash
conda activate odontoai
python src/montar_dataset_oficial.py
python src/limpar_dataset.py
python src/preparar_splits_por_grupos.py
python src/auditar_splits.py
python src/auditar_similaridade_visual.py
python src/train_diseases.py
python src/matriz_confusao.py
python src/gerar_curva_aprendizado.py
python src/gerar_figuras_artigo.py
python src/auditar_gradcam.py
streamlit run src/app.py
```

## Instalacao

Crie e ative um ambiente Python. Exemplo com Conda:

```bash
conda create -n odontoai python=3.11
conda activate odontoai
pip install -r requirements.txt
```

Tambem e possivel usar `venv`, desde que as dependencias de `requirements.txt` sejam instaladas.

## Resultados finais

| Item | Resultado |
| --- | ---: |
| Dataset final | 3.507 imagens |
| Treino | 2.455 imagens |
| Validacao | 528 imagens |
| Teste | 524 imagens |
| Acuracia no teste | 96,18% |
| F1-score macro | 90,78% |
| F1-score ponderado | 96,06% |

Auditoria final:

- 0 duplicatas MD5 entre treino, validacao e teste.
- 0 pares visualmente suspeitos por pHash entre treino, validacao e teste.

## Limitacoes

- O sistema nao substitui avaliacao profissional.
- O dataset e limitado e pode nao representar todos os cenarios clinicos reais.
- Algumas classes possuem menos exemplos, o que pode afetar recall e generalizacao.
- A validacao em bases externas ainda e necessaria.
- Grad-CAM e uma ferramenta qualitativa de interpretabilidade, nao uma prova clinica de diagnostico.

## Arquivos nao versionados

As pastas abaixo sao mantidas apenas localmente e nao devem ser enviadas ao GitHub neste momento:

- `data/`: datasets brutos, limpos e divididos.
- `models/`: checkpoints e modelos treinados.
- `outputs/`: metricas, matrizes, figuras e resultados gerados.
- `docs/`: documentacao detalhada e material do artigo, ainda em preparacao.

O arquivo de modelo `.pth` nao faz parte do repositorio. Ele deve ser gerado localmente executando o treinamento (`python src/train_diseases.py`) ou, futuramente, disponibilizado por link externo apropriado.

## Estrutura versionada esperada

Para a publicacao inicial no GitHub, a estrutura versionada deve conter apenas:

```text
src/
README.md
COMANDOS.md
requirements.txt
.gitignore
```
