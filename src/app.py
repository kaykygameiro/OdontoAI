from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet50


# Caminho do checkpoint final treinado localmente.
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "modelo_doencas.pth"

# Formatos aceitos no upload da interface.
TIPOS_IMAGEM = ["jpg", "jpeg", "png", "bmp", "webp"]

# Metricas finais do experimento consolidado para o TCC.
METRICAS_FINAIS = {
    "dataset_final": "3.507",
    "treino": "2.455",
    "validacao": "528",
    "total_teste": "524",
    "acuracia_teste": "96,18%",
    "f1_macro": "90,78%",
    "f1_ponderado": "96,06%",
}

# Nomes amigaveis em portugues. As classes internas do checkpoint nao sao alteradas.
NOMES_PT = {
    "Calculus": "Tártaro / cálculo dentário",
    "Caries": "Cárie",
    "Gingivitis": "Gengivite",
    "Hypodontia": "Hipodontia",
    "Mouth_Ulcer": "Úlcera bucal",
    "Tooth_Discoloration": "Descoloração dentária",
}

DESCRICAO_CLASSES = {
    "Calculus": "Acúmulo mineralizado aderido aos dentes.",
    "Caries": "Lesão associada à perda mineral da estrutura dentária.",
    "Gingivitis": "Inflamação gengival observável em imagens intraorais.",
    "Hypodontia": "Ausência congênita de um ou mais dentes.",
    "Mouth_Ulcer": "Lesão ulcerada na mucosa oral.",
    "Tooth_Discoloration": "Alteração visível de coloração dentária.",
}

AVISO_EDUCACIONAL = (
    "Este sistema é apenas um apoio educacional e não substitui avaliação "
    "profissional de um cirurgião-dentista."
)

EXPLICACAO_PROBABILIDADE = (
    "A probabilidade exibida representa a distribuição produzida pelo modelo "
    "entre as classes conhecidas e não deve ser interpretada como certeza clínica."
)


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

        # Cada canal recebe peso pela média espacial dos gradientes.
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
        """Remove hooks para evitar efeitos colaterais entre inferências."""
        for handle in self.handles:
            handle.remove()


def nome_pt(classe: str) -> str:
    """Retorna o nome em português sem alterar o identificador interno."""
    return NOMES_PT.get(classe, classe)


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
    """Cria o pré-processamento usando os parâmetros salvos no checkpoint."""
    return transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
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
    """Executa inferência e retorna classe prevista, probabilidade e top 3."""
    with torch.no_grad():
        logits = model(input_tensor)
        probabilidades = torch.softmax(logits, dim=1).squeeze(0)

    probabilidade, indice_predito = torch.max(probabilidades, dim=0)
    top_valores, top_indices = torch.topk(probabilidades, k=min(3, len(classes)))

    top3 = [
        (classes[int(indice)], float(valor.item()))
        for valor, indice in zip(top_valores.cpu(), top_indices.cpu())
    ]

    indice_predito_int = int(indice_predito.item())
    return (
        indice_predito_int,
        classes[indice_predito_int],
        float(probabilidade.item()),
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
    """Mostra informações resumidas do modelo e do experimento final."""
    st.sidebar.header("Informações do modelo")
    st.sidebar.write("MVP/protótipo acadêmico educacional")
    st.sidebar.write("Modelo utilizado: ResNet-50 com transfer learning")
    st.sidebar.write(f"Dataset final: {METRICAS_FINAIS['dataset_final']} imagens")
    st.sidebar.write(f"Treino: {METRICAS_FINAIS['treino']} imagens")
    st.sidebar.write(f"Validação: {METRICAS_FINAIS['validacao']} imagens")
    st.sidebar.write(f"Acurácia no teste: {METRICAS_FINAIS['acuracia_teste']}")
    st.sidebar.write(f"Total de imagens no teste: {METRICAS_FINAIS['total_teste']}")
    st.sidebar.write(f"F1-score macro: {METRICAS_FINAIS['f1_macro']}")
    st.sidebar.write(f"F1-score ponderado: {METRICAS_FINAIS['f1_ponderado']}")

    if device is not None:
        st.sidebar.write(f"Device usado: {device}")

    st.sidebar.divider()
    st.sidebar.caption(
        "Dataset final com imagens originais; pastas augmented e YOLO ignoradas; "
        "duplicatas e conflitos removidos por MD5; auditoria sem duplicatas "
        "exatas entre treino, validação e teste. O dataset é desbalanceado; "
        "Cárie teve o menor recall e baixo suporte no teste."
    )


def mostrar_indicador_probabilidade(probabilidade: float) -> None:
    """Mostra a probabilidade estimada sem sugerir certeza clínica."""
    percentual = probabilidade * 100

    if probabilidade >= 0.80:
        st.success(
            "Probabilidade estimada pelo modelo entre as classes treinadas: "
            f"{percentual:.2f}%"
        )
    elif probabilidade >= 0.50:
        st.warning(
            "Probabilidade estimada pelo modelo entre as classes treinadas: "
            f"{percentual:.2f}%"
        )
    else:
        st.error(
            "Probabilidade estimada pelo modelo entre as classes treinadas: "
            f"{percentual:.2f}%"
        )


def mostrar_top3(top3: list[tuple[str, float]]) -> None:
    """Mostra as três classes mais prováveis com barra de progresso."""
    st.write("**Top 3 classes:**")

    for classe, probabilidade in top3:
        st.write(f"{nome_pt(classe)}: {probabilidade * 100:.2f}%")
        st.progress(probabilidade)


def mostrar_descricao_classes() -> None:
    """Exibe uma explicação simples das classes previstas pelo sistema."""
    st.subheader("Classes avaliadas")

    for classe, descricao in DESCRICAO_CLASSES.items():
        st.write(f"**{nome_pt(classe)}:** {descricao}")


def mostrar_metodologia_resumida() -> None:
    """Resume os cuidados metodológicos usados no experimento final."""
    st.subheader("Cuidados metodológicos")
    st.write(
        "O dataset final usou imagens originais disponíveis no Kaggle. Pastas "
        "augmented e o dataset YOLO foram ignorados para evitar vazamento visual "
        "e mistura de tarefas. Duplicatas exatas e conflitos de rótulo foram "
        "removidos por MD5, e a auditoria confirmou ausência de duplicatas "
        "exatas entre treino, validação e teste."
    )
    st.write(
        "Data augmentation foi aplicado apenas ao conjunto de treino. Validação "
        "e teste utilizaram somente redimensionamento, conversão para tensor e "
        "normalização ImageNet."
    )


def main() -> None:
    """Executa a interface Streamlit para demonstração educacional."""
    st.set_page_config(
        page_title="OdontoAI",
        layout="centered",
    )

    st.title("OdontoAI - Classificação de Doenças Odontológicas")
    st.write(
        "MVP/protótipo acadêmico educacional para classificar imagens "
        "odontológicas em seis categorias treinadas."
    )
    st.warning(AVISO_EDUCACIONAL)

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
        class_idx, classe_prevista, probabilidade, top3 = predizer_tensor(
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
        st.metric("Classe prevista", nome_pt(classe_prevista))
        st.metric(
            "Probabilidade estimada pelo modelo entre as classes treinadas",
            f"{probabilidade * 100:.2f}%",
        )
        st.caption(EXPLICACAO_PROBABILIDADE)
        mostrar_indicador_probabilidade(probabilidade)
        mostrar_top3(top3)

    mostrar_descricao_classes()
    mostrar_metodologia_resumida()


if __name__ == "__main__":
    main()
