# OdontoAI - Classificação de Imagens Odontológicas

OdontoAI é um MVP/protótipo acadêmico educacional para classificação de imagens odontológicas utilizando redes neurais convolucionais, ResNet-50, Streamlit e Grad-CAM.

O projeto organiza um pipeline completo de montagem do dataset, limpeza, divisão em treino, validação e teste, auditorias contra vazamento de dados, treinamento, avaliação, geração de figuras e interface web para inferência.

## Avisos importantes

Este sistema possui finalidade acadêmica e educacional. Ele não realiza diagnóstico clínico e não substitui a avaliação profissional de um cirurgião-dentista.

Os resultados do modelo devem ser interpretados com cautela, especialmente em cenários reais, bases externas ou imagens com distribuição diferente daquela utilizada no experimento.

## Classes avaliadas

| Classe interna      | Descrição                  |
| ------------------- | -------------------------- |
| Calculus            | Tártaro / cálculo dentário |
| Caries              | Cárie                      |
| Gingivitis          | Gengivite                  |
| Hypodontia          | Hipodontia                 |
| Mouth_Ulcer         | Úlcera bucal               |
| Tooth_Discoloration | Descoloração dentária      |

## Arquitetura utilizada

O modelo final utiliza a arquitetura ResNet-50 com transfer learning.

O treinamento foi organizado em duas fases:

1. Treinamento inicial apenas da camada final.
2. Ajuste fino da `layer4` e da camada final.

A interpretabilidade visual é realizada com Grad-CAM, utilizada apenas como ferramenta qualitativa para mostrar as áreas de foco que justificam a classificação do modelo.

## Dataset utilizado

Este projeto utiliza como base o dataset público **Oral Diseases**, disponível no Kaggle:

https://www.kaggle.com/datasets/salmansajid05/oral-diseases

A base original contém imagens odontológicas organizadas em diferentes categorias e também inclui pastas com imagens previamente aumentadas e um subconjunto no formato YOLO. Neste projeto, foram utilizadas apenas as imagens originais disponíveis para a tarefa de classificação multiclasse. As pastas `augmented` e o subconjunto YOLO foram ignorados no experimento final para reduzir o risco de vazamento visual e manter o foco em classificação de imagem inteira.

Após a seleção das imagens originais, o dataset passou por limpeza de duplicatas e conflitos por MD5, agrupamento de imagens visualmente semelhantes por pHash e divisão em treino, validação e teste por grupos visuais.

## Pipeline oficial
O data augmentation é aplicado apenas ao conjunto de treino. Os conjuntos de validação e teste não utilizam augmentation.

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

## Instalação

Crie e ative um ambiente Python. Exemplo com Conda:

```bash
conda create -n odontoai python=3.11
conda activate odontoai
pip install -r requirements.txt
```

Também é possível utilizar `venv`, desde que as dependências listadas em `requirements.txt` sejam instaladas.

## Resultados finais

| Item               |     Resultado |
| ------------------ | ------------: |
| Dataset final      | 3.507 imagens |
| Treino             | 2.455 imagens |
| Validação          |   528 imagens |
| Teste              |   524 imagens |
| Acurácia no teste  |        96,18% |
| F1-score macro     |        90,78% |
| F1-score ponderado |        96,06% |

Auditoria final:

* 0 duplicatas MD5 entre treino, validação e teste.
* 0 pares visualmente suspeitos por pHash entre treino, validação e teste.

## Limitações

* O sistema não substitui avaliação profissional.
* O dataset é limitado e pode não representar todos os cenários clínicos reais.
* Algumas classes possuem menos exemplos, o que pode afetar recall e generalização.
* A validação em bases externas ainda é necessária.
* Grad-CAM é uma ferramenta qualitativa de interpretabilidade, não uma prova clínica de diagnóstico.

