# Comandos principais

Pipeline oficial final do projeto OdontoAI.

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

## O que cada comando faz

`conda activate odontoai`

Ativa o ambiente local do projeto.

`python src/montar_dataset_oficial.py`

Monta o dataset oficial usando imagens originais e ignorando pastas `augmented` e dataset YOLO.

`python src/limpar_dataset.py`

Remove duplicatas exatas e conflitos de rotulo com base em MD5.

`python src/preparar_splits_por_grupos.py`

Agrupa imagens visualmente semelhantes por pHash e cria os conjuntos de treino, validacao e teste por grupos visuais.

`python src/auditar_splits.py`

Verifica a separacao dos splits e procura duplicatas MD5 entre treino, validacao e teste.

`python src/auditar_similaridade_visual.py`

Audita possiveis imagens visualmente semelhantes entre os splits usando pHash.

`python src/train_diseases.py`

Treina o modelo ResNet-50 com augmentation apenas no conjunto de treino.

`python src/matriz_confusao.py`

Gera a matriz de confusao e metricas de avaliacao do conjunto de teste.

`python src/gerar_curva_aprendizado.py`

Gera a curva de aprendizado a partir dos historicos do treinamento.

`python src/gerar_figuras_artigo.py`

Gera figuras auxiliares para apresentacao ou artigo academico.

`python src/auditar_gradcam.py`

Gera e audita visualizacoes Grad-CAM para interpretabilidade qualitativa.

`streamlit run src/app.py`

Inicia a interface web local do prototipo.

## Observacao sobre arquivos locais

As pastas `data/`, `models/`, `outputs/` e `docs/` nao devem ser enviadas ao GitHub neste momento. Elas ficam locais e sao ignoradas pelo `.gitignore`.
