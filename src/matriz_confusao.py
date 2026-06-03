from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50


# Configuracoes do DataLoader para avaliacao.
BATCH_SIZE = 16
NUM_WORKERS = 2

# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT_DIR / "data" / "processed" / "test"
MODEL_PATH = ROOT_DIR / "models" / "modelo_doencas.pth"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# Arquivos de saida da avaliacao.
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report_teste.txt"
METRICS_PATH = REPORTS_DIR / "metricas_teste.json"
CONFUSION_MATRIX_ABS_PATH = REPORTS_DIR / "matriz_confusao_teste_absoluta.png"
CONFUSION_MATRIX_NORM_PATH = REPORTS_DIR / "matriz_confusao_teste_normalizada.png"


def verificar_arquivos() -> None:
    """Verifica se o teste e o checkpoint existem antes de avaliar."""
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Pasta de teste nao encontrada: {TEST_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo treinado nao encontrado: {MODEL_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def criar_transform(checkpoint: dict) -> transforms.Compose:
    """Cria o transform do teste usando os parametros salvos no checkpoint."""
    image_size = checkpoint["image_size"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def criar_test_loader(
    checkpoint: dict,
    device: torch.device,
) -> tuple[datasets.ImageFolder, DataLoader]:
    """Cria o dataset e dataloader somente para o conjunto de teste."""
    transform = criar_transform(checkpoint)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    class_to_idx_checkpoint = checkpoint["class_to_idx"]
    if test_dataset.class_to_idx != class_to_idx_checkpoint:
        raise ValueError(
            "A ordem das classes do conjunto de teste esta diferente da ordem "
            "salva no checkpoint.\n"
            f"class_to_idx do teste: {test_dataset.class_to_idx}\n"
            f"class_to_idx do checkpoint: {class_to_idx_checkpoint}"
        )

    pin_memory = device.type == "cuda"
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return test_dataset, test_loader


def carregar_modelo(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    """Reconstrói a ResNet-50 e carrega os pesos treinados."""
    classes = checkpoint["classes"]

    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def avaliar_modelo(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Executa inferencia no teste e retorna rotulos reais e preditos."""
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imagens, rotulos in test_loader:
            imagens = imagens.to(device, non_blocking=True)

            saidas = model(imagens)
            predicoes = torch.argmax(saidas, dim=1)

            y_true.extend(rotulos.cpu().tolist())
            y_pred.extend(predicoes.cpu().tolist())

    return y_true, y_pred


def salvar_matriz_absoluta(matriz: np.ndarray, classes: list[str]) -> None:
    """Salva a matriz de confusao absoluta em PNG."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predição da IA")
    plt.ylabel("Classe real")
    plt.title("Matriz de Confusão - Conjunto de Teste")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_ABS_PATH, dpi=300)
    plt.close()


def salvar_matriz_normalizada(matriz: np.ndarray, classes: list[str]) -> None:
    """Salva a matriz de confusao normalizada por classe real em PNG."""
    soma_linhas = matriz.sum(axis=1, keepdims=True)
    matriz_normalizada = np.divide(
        matriz,
        soma_linhas,
        out=np.zeros_like(matriz, dtype=float),
        where=soma_linhas != 0,
    )
    matriz_percentual = matriz_normalizada * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matriz_percentual,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=100,
    )
    plt.xlabel("Predição da IA")
    plt.ylabel("Classe real")
    plt.title("Matriz de Confusão Normalizada (%) - Conjunto de Teste")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_NORM_PATH, dpi=300)
    plt.close()


def salvar_metricas(
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
    matriz: np.ndarray,
) -> float:
    """Calcula e salva metricas do conjunto de teste."""
    accuracy = accuracy_score(y_true, y_pred)

    relatorio_texto = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
    )
    CLASSIFICATION_REPORT_PATH.write_text(relatorio_texto, encoding="utf-8")

    relatorio_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    metricas = {
        "accuracy": accuracy,
        "classification_report": relatorio_dict,
        "confusion_matrix": matriz.tolist(),
        "total_imagens_teste": len(y_true),
        "soma_matriz_confusao": int(matriz.sum()),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, indent=4, ensure_ascii=False)

    return accuracy


def main() -> None:
    """Avalia o modelo treinado somente no conjunto de teste."""
    verificar_arquivos()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device em uso: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    test_dataset, test_loader = criar_test_loader(checkpoint, device)
    model = carregar_modelo(checkpoint, device)

    y_true, y_pred = avaliar_modelo(model, test_loader, device)

    matriz = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )
    accuracy = salvar_metricas(y_true, y_pred, classes, matriz)

    salvar_matriz_absoluta(matriz, classes)
    salvar_matriz_normalizada(matriz, classes)

    total_teste = len(test_dataset)
    soma_matriz = int(matriz.sum())

    print(f"Total de imagens avaliadas: {total_teste}")
    print(f"Classes: {classes}")
    print(f"Acuracia no teste: {accuracy:.4f}")
    print(f"Soma da matriz de confusao: {soma_matriz}")
    print(f"Soma igual ao total de teste: {soma_matriz == total_teste}")

    print("\nArquivos salvos:")
    print(f"- {CLASSIFICATION_REPORT_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {CONFUSION_MATRIX_ABS_PATH}")
    print(f"- {CONFUSION_MATRIX_NORM_PATH}")


if __name__ == "__main__":
    main()
