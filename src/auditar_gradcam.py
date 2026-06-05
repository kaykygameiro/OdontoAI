from pathlib import Path
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT_DIR / "data" / "processed" / "test"
MODEL_PATH = ROOT_DIR / "models" / "modelo_doencas.pth"
GRADCAM_DIR = ROOT_DIR / "outputs" / "gradcam" / "auditoria"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
CSV_PATH = REPORTS_DIR / "auditoria_gradcam.csv"
GRADE_PATH = REPORTS_DIR / "auditoria_gradcam_grade.png"

# Extensões de imagem aceitas no conjunto de teste.
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Nomes amigáveis em português. As classes internas do checkpoint não são alteradas.
NOMES_PT = {
    "Calculus": "Tártaro / cálculo dentário",
    "Caries": "Cárie",
    "Gingivitis": "Gengivite",
    "Hypodontia": "Hipodontia",
    "Mouth_Ulcer": "Úlcera bucal",
    "Tooth_Discoloration": "Descoloração dentária",
}


class GradCAM:
    """Calcula Grad-CAM real a partir de ativações e gradientes da camada alvo."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._registrar_hooks()

    def _registrar_hooks(self) -> None:
        """Registra hooks para capturar ativações no forward e gradientes no backward."""
        self.handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, _module, _inputs, output) -> None:
        """Guarda as ativações produzidas pela camada alvo."""
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        """Guarda os gradientes recebidos pela camada alvo."""
        self.gradients = grad_output[0].detach()

    def gerar(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Gera o mapa Grad-CAM normalizado entre 0 e 1."""
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        score = output[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Não foi possível capturar ativações ou gradientes.")

        # Os pesos dos canais são a média espacial dos gradientes.
        pesos = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (pesos * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remover_hooks(self) -> None:
        """Remove hooks para evitar efeitos colaterais."""
        for handle in self.handles:
            handle.remove()


def nome_pt(classe: str) -> str:
    """Retorna o nome em português para exibição."""
    return NOMES_PT.get(classe, classe)


def caminho_relativo_projeto(caminho: Path) -> str:
    """Converte um caminho para formato relativo legível no CSV."""
    return str(caminho.relative_to(ROOT_DIR)).replace("\\", "/")


def verificar_entradas() -> None:
    """Verifica se teste e checkpoint existem antes da auditoria."""
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Pasta de teste não encontrada: {TEST_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo treinado não encontrado: {MODEL_PATH}")

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def carregar_checkpoint_e_modelo(device: torch.device) -> tuple[dict, torch.nn.Module]:
    """Reconstrói a ResNet-50 e carrega os pesos do checkpoint final."""
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return checkpoint, model


def criar_transform(checkpoint: dict) -> transforms.Compose:
    """Cria o pré-processamento do teste sem data augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
        ]
    )


def listar_imagens_por_classe(por_classe: int) -> list[tuple[str, Path]]:
    """Seleciona automaticamente até N imagens por classe do conjunto de teste."""
    selecao = []

    for pasta_classe in sorted(caminho for caminho in TEST_DIR.iterdir() if caminho.is_dir()):
        imagens = sorted(
            caminho
            for caminho in pasta_classe.rglob("*")
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
        )

        for imagem in imagens[:por_classe]:
            selecao.append((pasta_classe.name, imagem))

    return selecao


def predizer(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    classes: list[str],
) -> tuple[int, str, float]:
    """Executa inferência e retorna índice, classe prevista e probabilidade."""
    with torch.no_grad():
        logits = model(input_tensor)
        probabilidades = torch.softmax(logits, dim=1).squeeze(0)

    probabilidade, indice_predito = torch.max(probabilidades, dim=0)
    indice = int(indice_predito.item())
    return indice, classes[indice], float(probabilidade.item())


def sobrepor_gradcam(imagem: Image.Image, heatmap: np.ndarray, image_size: int) -> np.ndarray:
    """Sobrepõe o mapa Grad-CAM na imagem original redimensionada."""
    imagem_redimensionada = imagem.resize((image_size, image_size))
    imagem_rgb = np.array(imagem_redimensionada)

    heatmap_redimensionado = cv2.resize(heatmap, (image_size, image_size))
    heatmap_uint8 = np.uint8(255 * heatmap_redimensionado)
    heatmap_colorido = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colorido = cv2.cvtColor(heatmap_colorido, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(imagem_rgb, 0.55, heatmap_colorido, 0.45, 0)


def salvar_figura_lado_a_lado(
    imagem_original: Image.Image,
    imagem_gradcam: np.ndarray,
    classe_real: str,
    classe_prevista: str,
    probabilidade: float,
    acertou: bool,
    destino: Path,
) -> None:
    """Salva uma figura com imagem original e Grad-CAM lado a lado."""
    status = "ACERTO" if acertou else "ERRO"
    titulo = (
        f"Real: {nome_pt(classe_real)} | Prevista: {nome_pt(classe_prevista)} | "
        f"Probabilidade: {probabilidade * 100:.2f}% | {status}"
    )

    figura, eixos = plt.subplots(1, 2, figsize=(9, 4.5))
    eixos[0].imshow(imagem_original)
    eixos[0].set_title("Imagem original")
    eixos[0].axis("off")

    eixos[1].imshow(imagem_gradcam)
    eixos[1].set_title("Grad-CAM sobreposto")
    eixos[1].axis("off")

    figura.suptitle(titulo, fontsize=10)
    figura.tight_layout()
    figura.savefig(destino, dpi=180)
    plt.close(figura)


def salvar_grade_resumo(linhas: list[dict]) -> None:
    """Cria uma grade visual com as imagens Grad-CAM da auditoria."""
    if not linhas:
        return

    colunas = 3
    linhas_grade = int(np.ceil(len(linhas) / colunas))
    figura, eixos = plt.subplots(linhas_grade, colunas, figsize=(12, 4 * linhas_grade))
    eixos = np.array(eixos).reshape(-1)

    for eixo, linha in zip(eixos, linhas):
        imagem = Image.open(ROOT_DIR / linha["caminho_gradcam"]).convert("RGB")
        eixo.imshow(imagem)
        status = "acerto" if linha["acertou"] else "erro"
        eixo.set_title(
            f"{nome_pt(linha['classe_real'])} -> {nome_pt(linha['classe_prevista'])}\n"
            f"{linha['probabilidade'] * 100:.2f}% | {status}",
            fontsize=8,
        )
        eixo.axis("off")

    for eixo in eixos[len(linhas):]:
        eixo.axis("off")

    figura.suptitle("Auditoria Grad-CAM - conjunto de teste", fontsize=14)
    figura.tight_layout()
    figura.savefig(GRADE_PATH, dpi=180)
    plt.close(figura)


def auditar_gradcam(por_classe: int) -> list[dict]:
    """Executa a auditoria visual Grad-CAM sem treinar ou alterar dados."""
    verificar_entradas()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device em uso: {device}")

    checkpoint, model = carregar_checkpoint_e_modelo(device)
    classes = checkpoint["classes"]
    transform = criar_transform(checkpoint)

    imagens = listar_imagens_por_classe(por_classe)
    if not imagens:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {TEST_DIR}")

    gradcam = GradCAM(model=model, target_layer=model.layer4[-1])
    linhas = []

    try:
        for indice, (classe_real, caminho_imagem) in enumerate(imagens, start=1):
            imagem = Image.open(caminho_imagem).convert("RGB")
            input_tensor = transform(imagem).unsqueeze(0).to(device)

            indice_predito, classe_prevista, probabilidade = predizer(
                model=model,
                input_tensor=input_tensor,
                classes=classes,
            )
            heatmap = gradcam.gerar(input_tensor, indice_predito)
            imagem_gradcam = sobrepor_gradcam(
                imagem=imagem,
                heatmap=heatmap,
                image_size=checkpoint["image_size"],
            )

            acertou = classe_real == classe_prevista
            nome_saida = (
                f"{indice:03d}_{classe_real}_prev_{classe_prevista}_"
                f"{caminho_imagem.stem}.png"
            )
            destino = GRADCAM_DIR / nome_saida

            salvar_figura_lado_a_lado(
                imagem_original=imagem,
                imagem_gradcam=imagem_gradcam,
                classe_real=classe_real,
                classe_prevista=classe_prevista,
                probabilidade=probabilidade,
                acertou=acertou,
                destino=destino,
            )

            linhas.append(
                {
                    "caminho_imagem": caminho_relativo_projeto(caminho_imagem),
                    "classe_real": classe_real,
                    "classe_prevista": classe_prevista,
                    "acertou": acertou,
                    "probabilidade": probabilidade,
                    "caminho_gradcam": caminho_relativo_projeto(destino),
                }
            )

            status = "acertou" if acertou else "errou"
            print(
                f"{indice}/{len(imagens)} - {classe_real} -> {classe_prevista} "
                f"({probabilidade * 100:.2f}%) - {status}"
            )
    finally:
        gradcam.remover_hooks()

    return linhas


def salvar_csv(linhas: list[dict]) -> None:
    """Salva os metadados da auditoria em CSV."""
    df = pd.DataFrame(
        linhas,
        columns=[
            "caminho_imagem",
            "classe_real",
            "classe_prevista",
            "acertou",
            "probabilidade",
            "caminho_gradcam",
        ],
    )
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gera auditoria visual Grad-CAM no conjunto de teste."
    )
    parser.add_argument(
        "--por-classe",
        type=int,
        default=5,
        help="Quantidade de imagens selecionadas por classe. Padrao: 5.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada da auditoria Grad-CAM."""
    args = parse_args()
    if args.por_classe <= 0:
        raise ValueError("--por-classe deve ser maior que zero.")

    linhas = auditar_gradcam(por_classe=args.por_classe)
    salvar_csv(linhas)
    salvar_grade_resumo(linhas)

    print("\nAuditoria Grad-CAM finalizada.")
    print(f"Imagens salvas em: {GRADCAM_DIR}")
    print(f"CSV salvo em: {CSV_PATH}")
    print(f"Grade resumo salva em: {GRADE_PATH}")


if __name__ == "__main__":
    main()
