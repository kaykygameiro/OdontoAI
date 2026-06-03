from pathlib import Path
import argparse
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "modelo_doencas.pth"
OUTPUT_DIR = ROOT_DIR / "outputs" / "gradcam"


class GradCAM:
    """Implementa Grad-CAM usando ativacoes e gradientes da camada alvo."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._registrar_hooks()

    def _registrar_hooks(self) -> None:
        """Registra hooks para capturar ativacoes e gradientes."""
        self.handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, _module, _inputs, output) -> None:
        """Guarda as ativacoes produzidas pela camada alvo."""
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        """Guarda os gradientes que chegam na camada alvo."""
        self.gradients = grad_output[0].detach()

    def gerar(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Gera o heatmap Grad-CAM normalizado entre 0 e 1."""
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        score = output[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Nao foi possivel capturar ativacoes ou gradientes.")

        # Peso de cada canal: media espacial dos gradientes.
        pesos = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Combina os canais das ativacoes usando os pesos calculados.
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
        """Remove os hooks registrados para evitar efeitos colaterais."""
        for handle in self.handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    """Le o caminho da imagem informado pela linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gera Grad-CAM para uma imagem usando o modelo treinado."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Caminho da imagem que sera analisada.",
    )
    return parser.parse_args()


def verificar_arquivos(image_path: Path) -> None:
    """Verifica se o checkpoint e a imagem existem."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo treinado nao encontrado: {MODEL_PATH}")

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"O caminho informado nao e um arquivo: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def carregar_checkpoint(device: torch.device) -> dict:
    """Carrega o checkpoint do modelo treinado."""
    return torch.load(MODEL_PATH, map_location=device)


def carregar_modelo(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    """Recria a ResNet-50, carrega os pesos treinados e coloca em eval."""
    classes = checkpoint["classes"]

    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def criar_transform(checkpoint: dict) -> transforms.Compose:
    """Cria o transform usando image_size, mean e std salvos no checkpoint."""
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


def carregar_imagem(image_path: Path, checkpoint: dict, device: torch.device) -> tuple[Image.Image, torch.Tensor]:
    """Carrega a imagem original e cria o tensor normalizado para o modelo."""
    imagem = Image.open(image_path).convert("RGB")
    transform = criar_transform(checkpoint)
    input_tensor = transform(imagem).unsqueeze(0).to(device)
    return imagem, input_tensor


def predizer(model: torch.nn.Module, input_tensor: torch.Tensor) -> tuple[int, float, torch.Tensor]:
    """Executa a predicao e retorna classe prevista, confianca e probabilidades."""
    with torch.no_grad():
        logits = model(input_tensor)
        probabilidades = torch.softmax(logits, dim=1).squeeze(0)

    confianca, class_idx = torch.max(probabilidades, dim=0)
    return int(class_idx.item()), float(confianca.item()), probabilidades


def sobrepor_heatmap(imagem: Image.Image, heatmap: np.ndarray, image_size: int) -> np.ndarray:
    """Sobrepoe o heatmap Grad-CAM na imagem redimensionada."""
    imagem_redimensionada = imagem.resize((image_size, image_size))
    imagem_rgb = np.array(imagem_redimensionada)

    heatmap_redimensionado = cv2.resize(heatmap, (image_size, image_size))
    heatmap_uint8 = np.uint8(255 * heatmap_redimensionado)
    heatmap_colorido = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colorido = cv2.cvtColor(heatmap_colorido, cv2.COLOR_BGR2RGB)

    sobreposicao = cv2.addWeighted(imagem_rgb, 0.55, heatmap_colorido, 0.45, 0)
    return sobreposicao


def limpar_nome_arquivo(texto: str) -> str:
    """Remove caracteres problematicos para nomes de arquivo."""
    texto = texto.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", texto)


def salvar_gradcam(
    image_path: Path,
    classe_prevista: str,
    confianca: float,
    imagem_gradcam: np.ndarray,
) -> Path:
    """Salva a imagem Grad-CAM com nome contendo classe e confianca."""
    nome_original = limpar_nome_arquivo(image_path.stem)
    classe_limpa = limpar_nome_arquivo(classe_prevista)
    confianca_pct = int(round(confianca * 100))

    output_path = OUTPUT_DIR / (
        f"{nome_original}_pred-{classe_limpa}_conf-{confianca_pct}pct.png"
    )

    Image.fromarray(imagem_gradcam).save(output_path)
    return output_path


def imprimir_top3(classes: list[str], probabilidades: torch.Tensor) -> None:
    """Imprime as tres classes mais provaveis."""
    valores, indices = torch.topk(probabilidades.cpu(), k=min(3, len(classes)))

    print("Top 3 classes:")
    for posicao, (valor, indice) in enumerate(zip(valores, indices), start=1):
        print(f"{posicao}. {classes[int(indice)]}: {float(valor):.4f}")


def main() -> None:
    """Gera uma visualizacao Grad-CAM para uma unica imagem."""
    args = parse_args()
    image_path = args.image.resolve()
    verificar_arquivos(image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = carregar_checkpoint(device)
    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]

    model = carregar_modelo(checkpoint, device)
    imagem_original, input_tensor = carregar_imagem(image_path, checkpoint, device)

    class_idx, confianca, probabilidades = predizer(model, input_tensor)
    classe_prevista = classes[class_idx]

    gradcam = GradCAM(model=model, target_layer=model.layer4[-1])
    try:
        heatmap = gradcam.gerar(input_tensor, class_idx)
    finally:
        gradcam.remover_hooks()

    imagem_gradcam = sobrepor_heatmap(imagem_original, heatmap, image_size)
    output_path = salvar_gradcam(
        image_path=image_path,
        classe_prevista=classe_prevista,
        confianca=confianca,
        imagem_gradcam=imagem_gradcam,
    )

    print(f"Imagem analisada: {image_path}")
    print(f"Classe prevista: {classe_prevista}")
    print(f"Confianca: {confianca:.4f}")
    imprimir_top3(classes, probabilidades)
    print(f"Grad-CAM salvo em: {output_path}")


if __name__ == "__main__":
    main()
