from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
ARTIGO_DIR = ROOT_DIR / "outputs" / "artigo"


# Nomes usados apenas para exibição nas figuras e tabelas do artigo.
NOMES_PT = {
    "Calculus": "Tártaro",
    "Caries": "Cárie",
    "Gingivitis": "Gengivite",
    "Hypodontia": "Hipodontia",
    "Mouth_Ulcer": "Úlcera bucal",
    "Tooth_Discoloration": "Descoloração dentária",
}


def carregar_classes():
    """Carrega e valida a ordem das classes sem alterar os nomes internos."""
    classes_path = REPORTS_DIR / "classes.txt"
    class_to_idx_path = REPORTS_DIR / "class_to_idx.json"

    classes = [
        linha.strip()
        for linha in classes_path.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    ]

    with class_to_idx_path.open("r", encoding="utf-8") as arquivo:
        class_to_idx = json.load(arquivo)

    classes_por_indice = [
        classe
        for classe, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]

    if classes != classes_por_indice:
        raise ValueError(
            "A ordem de classes.txt não coincide com os índices de class_to_idx.json."
        )

    return classes


def nomes_em_portugues(classes):
    """Converte a lista de classes internas para os nomes exibidos em português."""
    return [NOMES_PT[classe] for classe in classes]


def salvar_matriz_confusao_absoluta(matriz, labels_pt):
    """Gera a matriz de confusão absoluta com rótulos em português."""
    caminho_saida = ARTIGO_DIR / "matriz_confusao_absoluta_pt.png"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_pt,
        yticklabels=labels_pt,
        cbar=True,
    )
    plt.title("Matriz de Confusão - Conjunto de Teste")
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    return caminho_saida


def salvar_matriz_confusao_normalizada(matriz, labels_pt):
    """Gera a matriz de confusão normalizada por classe real, em porcentagem."""
    caminho_saida = ARTIGO_DIR / "matriz_confusao_normalizada_pt.png"

    somas_linhas = matriz.sum(axis=1, keepdims=True)
    matriz_normalizada = np.divide(
        matriz,
        somas_linhas,
        out=np.zeros_like(matriz, dtype=float),
        where=somas_linhas != 0,
    ) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matriz_normalizada,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels_pt,
        yticklabels=labels_pt,
        cbar=True,
    )
    plt.title("Matriz de Confusão Normalizada (%) - Conjunto de Teste")
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    return caminho_saida


def encontrar_inicio_fine_tuning(historico):
    """Localiza a época global em que a fase de fine-tuning começa."""
    if "phase" in historico.columns and (historico["phase"] == "fase_2").any():
        return historico.index[historico["phase"] == "fase_2"][0] + 1

    if "learning_rate" in historico.columns:
        mudancas_lr = historico["learning_rate"].ne(historico["learning_rate"].shift())
        indices_mudanca = historico.index[mudancas_lr].tolist()
        if len(indices_mudanca) > 1:
            return indices_mudanca[1] + 1

    return None


def salvar_curva_aprendizado():
    """Gera as curvas de perda e acurácia lado a lado."""
    caminho_saida = ARTIGO_DIR / "curva_aprendizado_pt.png"
    historico = pd.read_csv(REPORTS_DIR / "historico_treino.csv")
    historico["epoca_global"] = np.arange(1, len(historico) + 1)
    inicio_fine_tuning = encontrar_inicio_fine_tuning(historico)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        historico["epoca_global"],
        historico["train_loss"],
        marker="o",
        label="Perda de treino",
    )
    axes[0].plot(
        historico["epoca_global"],
        historico["val_loss"],
        marker="o",
        label="Perda de validação",
    )
    axes[0].set_title("Curva de perda")
    axes[0].set_xlabel("Época global")
    axes[0].set_ylabel("Perda")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        historico["epoca_global"],
        historico["train_acc"] * 100,
        marker="o",
        label="Acurácia de treino",
    )
    axes[1].plot(
        historico["epoca_global"],
        historico["val_acc"] * 100,
        marker="o",
        label="Acurácia de validação",
    )
    axes[1].set_title("Curva de acurácia")
    axes[1].set_xlabel("Época global")
    axes[1].set_ylabel("Acurácia (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if inicio_fine_tuning is not None:
        for eixo in axes:
            eixo.axvline(
                x=inicio_fine_tuning,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Início do fine-tuning",
            )

        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels)
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels)

    fig.suptitle("Curva de Aprendizado - ResNet-50", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return caminho_saida


def salvar_distribuicao_splits(classes, labels_pt):
    """Gera a distribuição empilhada das imagens por split e classe."""
    caminho_saida = ARTIGO_DIR / "distribuicao_splits_pt.png"
    distribuicao = pd.read_csv(REPORTS_DIR / "distribuicao_splits.csv")
    distribuicao = distribuicao.set_index("classe").loc[classes]
    distribuicao = distribuicao.rename(index=dict(zip(classes, labels_pt)))

    colunas = ["treino", "validacao", "teste"]
    rotulos = ["Treino", "Validação", "Teste"]

    ax = distribuicao[colunas].plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    ax.set_title("Distribuição das Imagens por Classe")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de imagens")
    ax.set_xticklabels(distribuicao.index, rotation=25, ha="right")
    ax.legend(rotulos, title="Split")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    return caminho_saida


def salvar_tabela_metricas(metricas, classes):
    """Exporta a tabela de métricas do teste com nomes em português."""
    caminho_saida = ARTIGO_DIR / "tabela_metricas_teste_pt.csv"
    relatorio = metricas["classification_report"]
    linhas = []

    for classe in classes:
        valores = relatorio[classe]
        linhas.append(
            {
                "Classe": NOMES_PT[classe],
                "Precisão": valores["precision"],
                "Revocação": valores["recall"],
                "F1-score": valores["f1-score"],
                "Suporte": int(valores["support"]),
            }
        )

    suporte_total = int(relatorio["weighted avg"]["support"])
    linhas.append(
        {
            "Classe": "Acurácia geral",
            "Precisão": "",
            "Revocação": "",
            "F1-score": relatorio["accuracy"],
            "Suporte": suporte_total,
        }
    )
    linhas.append(
        {
            "Classe": "Média macro",
            "Precisão": relatorio["macro avg"]["precision"],
            "Revocação": relatorio["macro avg"]["recall"],
            "F1-score": relatorio["macro avg"]["f1-score"],
            "Suporte": int(relatorio["macro avg"]["support"]),
        }
    )
    linhas.append(
        {
            "Classe": "Média ponderada",
            "Precisão": relatorio["weighted avg"]["precision"],
            "Revocação": relatorio["weighted avg"]["recall"],
            "F1-score": relatorio["weighted avg"]["f1-score"],
            "Suporte": suporte_total,
        }
    )

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(caminho_saida, index=False, encoding="utf-8-sig")

    return caminho_saida


def formatar_percentual(valor):
    """Converte uma metrica decimal para percentual com virgula."""
    return f"{valor * 100:.2f}%".replace(".", ",")


def salvar_resumo_resultados(metricas):
    """Cria um resumo textual a partir de outputs/reports/metricas_teste.json."""
    caminho_saida = ARTIGO_DIR / "resumo_resultados_pt.txt"
    relatorio = metricas["classification_report"]
    total_teste = int(metricas["total_imagens_teste"])
    acuracia = formatar_percentual(metricas["accuracy"])
    f1_macro = formatar_percentual(relatorio["macro avg"]["f1-score"])
    f1_ponderado = formatar_percentual(relatorio["weighted avg"]["f1-score"])

    texto = (
        "Resumo dos resultados do OdontoAI no conjunto de teste\n\n"
        f"Total de imagens no teste: {total_teste}\n"
        f"Acurácia no teste: {acuracia}\n"
        f"F1-score macro: {f1_macro}\n"
        f"F1-score ponderado: {f1_ponderado}\n\n"
        "A classe Cárie apresentou o menor recall e baixo suporte no teste, "
        "o que exige cautela na interpretação dos resultados. O dataset é "
        "desbalanceado, e as métricas das classes com menos exemplos são mais "
        "sensíveis a poucos erros.\n\n"
        "Grad-CAM deve ser interpretado apenas como ferramenta qualitativa de "
        "apoio à inspeção visual, não como prova de diagnóstico.\n\n"
        "O sistema é um MVP/protótipo acadêmico educacional e não substitui "
        "a avaliação profissional realizada por cirurgião-dentista."
    )
    caminho_saida.write_text(texto, encoding="utf-8")

    return caminho_saida


def main():
    """Executa a geração dos arquivos finais para artigo/TCC."""
    sns.set_theme(style="whitegrid", context="paper")
    ARTIGO_DIR.mkdir(parents=True, exist_ok=True)

    classes = carregar_classes()
    labels_pt = nomes_em_portugues(classes)

    with (REPORTS_DIR / "metricas_teste.json").open("r", encoding="utf-8") as arquivo:
        metricas = json.load(arquivo)

    matriz_confusao = np.array(metricas["confusion_matrix"], dtype=int)
    if matriz_confusao.shape != (len(classes), len(classes)):
        raise ValueError("A matriz de confusão não corresponde ao número de classes.")

    arquivos_gerados = [
        salvar_matriz_confusao_absoluta(matriz_confusao, labels_pt),
        salvar_matriz_confusao_normalizada(matriz_confusao, labels_pt),
        salvar_curva_aprendizado(),
        salvar_distribuicao_splits(classes, labels_pt),
        salvar_tabela_metricas(metricas, classes),
        salvar_resumo_resultados(metricas),
    ]

    print("Arquivos gerados:")
    for caminho in arquivos_gerados:
        print(caminho)


if __name__ == "__main__":
    main()
