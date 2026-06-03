from pathlib import Path
import json

import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50


# Configuracoes principais do treinamento.
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 224
USE_AMP = True

# Hiperparametros da fase 1: treina apenas a camada final.
PHASE_1_EPOCHS = 5
PHASE_1_LR = 1e-3

# Hiperparametros da fase 2: ajusta layer4 e camada final.
PHASE_2_EPOCHS = 10
PHASE_2_LR = 1e-4

# Early stopping baseado no menor val_loss.
EARLY_STOPPING_PATIENCE = 4

# Normalizacao padrao do ImageNet, compativel com a ResNet-50 pre-treinada.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "data" / "processed" / "train"
VAL_DIR = ROOT_DIR / "data" / "processed" / "val"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
MODEL_PATH = MODEL_DIR / "modelo_doencas.pth"
HISTORY_PATH = REPORTS_DIR / "historico_treino.csv"
CLASSES_PATH = REPORTS_DIR / "classes.txt"
CLASS_TO_IDX_PATH = REPORTS_DIR / "class_to_idx.json"


def verificar_pastas() -> None:
    """Verifica se as pastas de treino e validacao existem."""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Pasta de treino nao encontrada: {TRAIN_DIR}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Pasta de validacao nao encontrada: {VAL_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def criar_transform() -> transforms.Compose:
    """Cria o transform sem data augmentation adicional."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def criar_dataloaders(device: torch.device) -> tuple[datasets.ImageFolder, datasets.ImageFolder, DataLoader, DataLoader]:
    """Cria datasets e dataloaders de treino e validacao."""
    transform = criar_transform()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def salvar_classes(train_dataset: datasets.ImageFolder) -> None:
    """Salva a lista de classes e o mapeamento class_to_idx."""
    CLASSES_PATH.write_text("\n".join(train_dataset.classes) + "\n", encoding="utf-8")

    with CLASS_TO_IDX_PATH.open("w", encoding="utf-8") as arquivo:
        json.dump(train_dataset.class_to_idx, arquivo, indent=4, ensure_ascii=False)


def calcular_class_weights(train_dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    """Calcula pesos por classe usando a distribuicao do conjunto de treino."""
    targets = torch.tensor(train_dataset.targets, dtype=torch.long)
    contagens = torch.bincount(targets, minlength=len(train_dataset.classes)).float()

    if torch.any(contagens == 0):
        classes_vazias = [
            train_dataset.classes[indice]
            for indice, quantidade in enumerate(contagens.tolist())
            if quantidade == 0
        ]
        raise ValueError(f"Classes sem imagens no treino: {classes_vazias}")

    total = contagens.sum()
    pesos = total / (len(train_dataset.classes) * contagens)
    return pesos.to(device)


def criar_modelo(num_classes: int, device: torch.device) -> nn.Module:
    """Carrega a ResNet-50 pre-treinada e substitui a camada final."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    modelo = resnet50(weights=weights)
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    return modelo.to(device)


def congelar_backbone(modelo: nn.Module) -> None:
    """Congela toda a ResNet-50 e deixa apenas a fc treinavel."""
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    for parametro in modelo.fc.parameters():
        parametro.requires_grad = True


def descongelar_layer4_e_fc(modelo: nn.Module) -> None:
    """Mantem o inicio do backbone congelado e libera layer4 e fc."""
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    for parametro in modelo.layer4.parameters():
        parametro.requires_grad = True

    for parametro in modelo.fc.parameters():
        parametro.requires_grad = True


def criar_otimizador(modelo: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    """Cria Adam apenas para parametros treinaveis."""
    parametros_treinaveis = [
        parametro for parametro in modelo.parameters() if parametro.requires_grad
    ]
    return torch.optim.Adam(parametros_treinaveis, lr=learning_rate)


def obter_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Retorna o learning rate atual do primeiro grupo de parametros."""
    return optimizer.param_groups[0]["lr"]


def executar_epoca_treino(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    usar_amp: bool,
) -> tuple[float, float]:
    """Executa uma epoca de treino e retorna loss e acuracia."""
    modelo.train()
    soma_loss = 0.0
    acertos = 0
    total = 0

    for imagens, rotulos in dataloader:
        imagens = imagens.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if usar_amp and scaler is not None:
            with autocast():
                saidas = modelo(imagens)
                loss = criterion(saidas, rotulos)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            saidas = modelo(imagens)
            loss = criterion(saidas, rotulos)
            loss.backward()
            optimizer.step()

        tamanho_lote = imagens.size(0)
        soma_loss += loss.item() * tamanho_lote
        _, predicoes = torch.max(saidas, 1)
        acertos += (predicoes == rotulos).sum().item()
        total += tamanho_lote

    return soma_loss / total, acertos / total


def executar_epoca_validacao(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    usar_amp: bool,
) -> tuple[float, float]:
    """Executa uma epoca de validacao e retorna loss e acuracia."""
    modelo.eval()
    soma_loss = 0.0
    acertos = 0
    total = 0

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(device, non_blocking=True)
            rotulos = rotulos.to(device, non_blocking=True)

            if usar_amp:
                with autocast():
                    saidas = modelo(imagens)
                    loss = criterion(saidas, rotulos)
            else:
                saidas = modelo(imagens)
                loss = criterion(saidas, rotulos)

            tamanho_lote = imagens.size(0)
            soma_loss += loss.item() * tamanho_lote
            _, predicoes = torch.max(saidas, 1)
            acertos += (predicoes == rotulos).sum().item()
            total += tamanho_lote

    return soma_loss / total, acertos / total


def salvar_checkpoint(
    modelo: nn.Module,
    train_dataset: datasets.ImageFolder,
    best_val_loss: float,
    best_val_acc: float,
) -> None:
    """Salva o melhor modelo encontrado durante o treinamento."""
    checkpoint = {
        "model_state_dict": modelo.state_dict(),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
    }
    torch.save(checkpoint, MODEL_PATH)


def treinar_fase(
    phase: str,
    modelo: nn.Module,
    train_dataset: datasets.ImageFolder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    best_val_loss: float,
    best_val_acc: float,
    historico: list[dict],
    usar_amp: bool,
) -> tuple[float, float]:
    """Treina uma fase completa, com scheduler e early stopping."""
    optimizer = criar_otimizador(modelo, learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
    )
    scaler = GradScaler() if usar_amp else None
    epocas_sem_melhora = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = executar_epoca_treino(
            modelo=modelo,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            usar_amp=usar_amp,
        )

        val_loss, val_acc = executar_epoca_validacao(
            modelo=modelo,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            usar_amp=usar_amp,
        )

        scheduler.step(val_loss)
        learning_rate_atual = obter_learning_rate(optimizer)

        historico.append(
            {
                "phase": phase,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": learning_rate_atual,
            }
        )

        print(
            f"{phase} | Epoca {epoch:02d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
            f"lr: {learning_rate_atual:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epocas_sem_melhora = 0
            salvar_checkpoint(
                modelo=modelo,
                train_dataset=train_dataset,
                best_val_loss=best_val_loss,
                best_val_acc=best_val_acc,
            )
            print(f"Melhor modelo salvo em: {MODEL_PATH}")
        else:
            epocas_sem_melhora += 1

        if epocas_sem_melhora >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping acionado na {phase}.")
            break

    return best_val_loss, best_val_acc


def main() -> None:
    """Executa o treinamento da ResNet-50 em duas fases."""
    verificar_pastas()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usar_amp = USE_AMP and device.type == "cuda"

    print(f"Device em uso: {device}")
    print(f"Mixed precision habilitado: {usar_amp}")

    train_dataset, val_dataset, train_loader, val_loader = criar_dataloaders(device)
    salvar_classes(train_dataset)

    print(f"Imagens de treino: {len(train_dataset)}")
    print(f"Imagens de validacao: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    class_weights = calcular_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    modelo = criar_modelo(num_classes=len(train_dataset.classes), device=device)

    historico = []
    best_val_loss = float("inf")
    best_val_acc = 0.0

    print("\nFASE 1: treinando apenas a camada final fc.")
    congelar_backbone(modelo)
    best_val_loss, best_val_acc = treinar_fase(
        phase="fase_1",
        modelo=modelo,
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=PHASE_1_EPOCHS,
        learning_rate=PHASE_1_LR,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        historico=historico,
        usar_amp=usar_amp,
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    print("Melhor checkpoint da Fase 1 recarregado antes da Fase 2.")

    print("\nFASE 2: ajustando layer4 e fc.")
    descongelar_layer4_e_fc(modelo)
    best_val_loss, best_val_acc = treinar_fase(
        phase="fase_2",
        modelo=modelo,
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=PHASE_2_EPOCHS,
        learning_rate=PHASE_2_LR,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        historico=historico,
        usar_amp=usar_amp,
    )

    df_historico = pd.DataFrame(historico)
    df_historico.to_csv(HISTORY_PATH, index=False, encoding="utf-8")

    print("\nTreinamento finalizado.")
    print(f"Melhor val_loss: {best_val_loss:.4f}")
    print(f"Melhor val_acc: {best_val_acc:.4f}")
    print(f"Historico salvo em: {HISTORY_PATH}")
    print(f"Melhor modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
