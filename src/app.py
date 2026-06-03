from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet50


# Caminho do checkpoint treinado.
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "modelo_doencas.pth"

# Extensões aceitas no upload.
TIPOS_IMAGEM = ["jpg", "jpeg", "png", "bmp", "webp"]

# Descrição simples das classes para exibição educacional.
DESCRICAO_CLASSES = {
    "Calculus": "tártaro/cálculo dentário",
    "Caries": "cárie",
    "Gingivitis": "gengivite",
    "Hypodontia": "ausência congênita de dentes",
    "Mouth_Ulcer": "úlcera/lesão ulcerada na boca",
    "Tooth_Discoloration": "alteração de coloração dentária",
}


class GradCAM:
    """Implementa Grad-CAM usando ativações e gradientes da camada alvo."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._registrar_hooks()

    def _registrar_hooks(self) -> None:
        """Registra hooks para capturar ativações e gradientes."""
        self.handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, _module, _inputs, output) -> None:
        """Guarda as ativações produzidas pela camada alvo."""
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
            raise RuntimeError("Não foi possível capturar ativações ou gradientes.")

        # Peso de cada canal: média espacial dos gradientes.
        pesos = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Combina os canais das ativações usando os pesos calculados.
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


@st.cache_resource
def carregar_modelo() -> tuple[torch.nn.Module, dict, torch.device]:
    """Carrega o checkpoint e recria a ResNet-50 apenas uma vez."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint, device


def criar_transform(checkpoint: dict) -> transforms.Compose:
    """Cria o preprocessamento usando os parâmetros salvos no checkpoint."""
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


def preparar_tensor(
    checkpoint: dict,
    device: torch.device,
    imagem: Image.Image,
) -> torch.Tensor:
    """Transforma uma imagem PIL em tensor normalizado para inferência."""
    transform = criar_transform(checkpoint)
    return transform(imagem).unsqueeze(0).to(device)


def predizer_tensor(
    model: torch.nn.Module,
    classes: list[str],
    input_tensor: torch.Tensor,
) -> tuple[int, str, float, list[tuple[str, float]]]:
    """Executa inferência e retorna classe prevista, confiança e top 3."""
    with torch.no_grad():
        logits = model(input_tensor)
        probabilidades = torch.softmax(logits, dim=1).squeeze(0)

    confianca, indice_predito = torch.max(probabilidades, dim=0)
    top_valores, top_indices = torch.topk(probabilidades, k=min(3, len(classes)))

    top3 = [
        (classes[int(indice)], float(valor.item()))
        for valor, indice in zip(top_valores.cpu(), top_indices.cpu())
    ]

    indice_predito_int = int(indice_predito.item())
    return (
        indice_predito_int,
        classes[indice_predito_int],
        float(confianca.item()),
        top3,
    )


def gerar_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    imagem: Image.Image,
    checkpoint: dict,
    class_idx: int,
) -> np.ndarray:
    """Gera a sobreposição Grad-CAM para a classe prevista."""
    image_size = checkpoint["image_size"]
    gradcam = GradCAM(model=model, target_layer=model.layer4[-1])

    try:
        heatmap = gradcam.gerar(input_tensor, class_idx)
    finally:
        gradcam.remover_hooks()

    imagem_redimensionada = imagem.resize((image_size, image_size))
    imagem_rgb = np.array(imagem_redimensionada)

    heatmap_redimensionado = cv2.resize(heatmap, (image_size, image_size))
    heatmap_uint8 = np.uint8(255 * heatmap_redimensionado)
    heatmap_colorido = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colorido = cv2.cvtColor(heatmap_colorido, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(imagem_rgb, 0.55, heatmap_colorido, 0.45, 0)


def mostrar_sidebar(device: torch.device | None = None) -> None:
    """Mostra informações resumidas do modelo e da avaliação final."""
    st.sidebar.header("Informações do modelo")
    st.sidebar.write("Modelo utilizado: ResNet-50")
    st.sidebar.write("Acurácia no teste: 92,45%")
    st.sidebar.write("Total de imagens no teste: 1.854")
    st.sidebar.write("Maior confusão observada: Calculus e Gingivitis")

    if device is not None:
        st.sidebar.write(f"Device usado: {device}")


def mostrar_alerta_confianca(confianca: float) -> None:
    """Mostra um alerta conforme o nível de confiança da predição."""
    percentual = confianca * 100

    if confianca >= 0.80:
        st.success(f"Predição com alta confiança: {percentual:.2f}%")
    elif confianca >= 0.50:
        st.warning(f"Predição com confiança moderada: {percentual:.2f}%")
    else:
        st.error(f"Predição incerta: {percentual:.2f}%")


def mostrar_top3(top3: list[tuple[str, float]]) -> None:
    """Mostra as três classes mais prováveis com barra de progresso."""
    st.write("**Top 3 classes:**")

    for classe, probabilidade in top3:
        st.write(f"{classe}: {probabilidade * 100:.2f}%")
        st.progress(probabilidade)


def mostrar_descricao_classes() -> None:
    """Exibe uma explicação simples das classes previstas pelo sistema."""
    st.subheader("Classes avaliadas")

    for classe, descricao in DESCRICAO_CLASSES.items():
        st.write(f"**{classe}:** {descricao}")


def main() -> None:
    """Executa a interface Streamlit para demonstração educacional."""
    st.set_page_config(
        page_title="OdontoAI",
        page_icon="🦷",
        layout="centered",
    )

    st.title("OdontoAI - Classificação de Doenças Odontológicas")
    st.write(
        "Este aplicativo demonstra um modelo de inteligência artificial treinado "
        "para classificar imagens odontológicas em seis categorias."
    )
    st.warning(
        "Este sistema é apenas um apoio educacional e não substitui avaliação "
        "profissional de um cirurgião-dentista."
    )

    try:
        model, checkpoint, device = carregar_modelo()
    except FileNotFoundError:
        mostrar_sidebar()
        st.error(
            "O modelo treinado não foi encontrado. Verifique se o arquivo "
            f"`{MODEL_PATH}` existe antes de usar o aplicativo."
        )
        return
    except Exception as erro:
        mostrar_sidebar()
        st.error(f"Não foi possível carregar o modelo: {erro}")
        return

    mostrar_sidebar(device)

    gerar_mapa = st.checkbox("Gerar Grad-CAM", value=True)
    arquivo = st.file_uploader(
        "Envie uma imagem odontológica",
        type=TIPOS_IMAGEM,
    )

    if arquivo is not None:
        try:
            imagem = Image.open(arquivo).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error(
                "Não foi possível ler a imagem enviada. Tente outro arquivo "
                "nos formatos jpg, jpeg, png, bmp ou webp."
            )
            return

        classes = checkpoint["classes"]
        input_tensor = preparar_tensor(checkpoint, device, imagem)
        class_idx, classe_prevista, confianca, top3 = predizer_tensor(
            model=model,
            classes=classes,
            input_tensor=input_tensor,
        )

        coluna_imagem, coluna_gradcam = st.columns(2)

        with coluna_imagem:
            st.subheader("Imagem enviada")
            st.image(imagem, use_container_width=True)

        with coluna_gradcam:
            st.subheader("Mapa de atenção Grad-CAM")
            if gerar_mapa:
                try:
                    imagem_gradcam = gerar_gradcam(
                        model=model,
                        input_tensor=input_tensor,
                        imagem=imagem,
                        checkpoint=checkpoint,
                        class_idx=class_idx,
                    )
                    st.image(imagem_gradcam, use_container_width=True)
                    st.caption(
                        "O Grad-CAM destaca regiões que influenciaram a decisão "
                        "do modelo. Essa visualização é interpretativa e não "
                        "representa diagnóstico clínico."
                    )
                except Exception as erro:
                    st.error(f"Não foi possível gerar o Grad-CAM: {erro}")
            else:
                st.info("Grad-CAM desativado para economizar processamento.")

        st.subheader("Resultado da classificação")
        st.metric("Classe prevista", classe_prevista)
        st.metric(
            "Confiança estimada pelo modelo",
            f"{confianca * 100:.2f}%",
        )
        st.caption(
            "A confiança exibida representa a probabilidade estimada pelo modelo "
            "entre as classes treinadas. Ela não corresponde a uma certeza clínica "
            "e deve ser interpretada apenas como apoio educacional."
        )
        mostrar_alerta_confianca(confianca)
        mostrar_top3(top3)

    mostrar_descricao_classes()


if __name__ == "__main__":
    main()
