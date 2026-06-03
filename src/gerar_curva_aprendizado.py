from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
HISTORY_PATH = REPORTS_DIR / "historico_treino.csv"
OUTPUT_PATH = REPORTS_DIR / "curva_aprendizado.png"

# Colunas esperadas no historico gerado pelo treinamento.
COLUNAS_OBRIGATORIAS = {
    "phase",
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "learning_rate",
}


def carregar_historico() -> pd.DataFrame:
    """Le o historico de treino e valida se as colunas esperadas existem."""
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Historico de treino nao encontrado: {HISTORY_PATH}")

    df = pd.read_csv(HISTORY_PATH)
    colunas_ausentes = COLUNAS_OBRIGATORIAS - set(df.columns)

    if colunas_ausentes:
        raise ValueError(f"Colunas ausentes no historico: {sorted(colunas_ausentes)}")

    if df.empty:
        raise ValueError("O historico de treino esta vazio.")

    # A epoca global respeita a ordem em que as linhas foram salvas no CSV.
    df = df.copy()
    df["epoca_global"] = range(1, len(df) + 1)

    return df


def obter_inicio_fine_tuning(df: pd.DataFrame) -> float | None:
    """Retorna a posicao da linha vertical que marca o inicio da fase 2."""
    linhas_fase_2 = df[df["phase"] == "fase_2"]

    if linhas_fase_2.empty:
        return None

    primeira_epoca_fase_2 = int(linhas_fase_2["epoca_global"].iloc[0])
    return primeira_epoca_fase_2 - 0.5


def gerar_grafico(df: pd.DataFrame) -> None:
    """Gera e salva o grafico de loss e acuracia."""
    inicio_fine_tuning = obter_inicio_fine_tuning(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curva de loss.
    axes[0].plot(
        df["epoca_global"],
        df["train_loss"],
        marker="o",
        label="Loss de treino",
    )
    axes[0].plot(
        df["epoca_global"],
        df["val_loss"],
        marker="o",
        label="Loss de validacao",
    )
    axes[0].set_title("Curva de Loss")
    axes[0].set_xlabel("Epoca global")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Curva de acuracia.
    axes[1].plot(
        df["epoca_global"],
        df["train_acc"],
        marker="o",
        label="Acuracia de treino",
    )
    axes[1].plot(
        df["epoca_global"],
        df["val_acc"],
        marker="o",
        label="Acuracia de validacao",
    )
    axes[1].set_title("Curva de Acuracia")
    axes[1].set_xlabel("Epoca global")
    axes[1].set_ylabel("Acuracia")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Linha vertical para separar a fase de treinamento da fc e o fine-tuning.
    if inicio_fine_tuning is not None:
        for ax in axes:
            ax.axvline(
                x=inicio_fine_tuning,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Início do fine-tuning",
            )
            handles, labels = ax.get_legend_handles_labels()
            labels_unicos = dict(zip(labels, handles))
            ax.legend(labels_unicos.values(), labels_unicos.keys())

    fig.suptitle("Curva de Aprendizado - ResNet-50", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    """Carrega o historico e gera a curva de aprendizado."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_historico()
    gerar_grafico(df)

    print(f"Curva de aprendizado salva em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
