from pathlib import Path
import json

import pandas as pd
from PIL import Image, UnidentifiedImageError

try:
    import imagehash
except ImportError as erro:
    raise SystemExit(
        "A biblioteca ImageHash nao esta instalada. Instale com:\n"
        "pip install ImageHash"
    ) from erro


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# Splits avaliados e extensoes de imagem aceitas.
SPLITS = ["train", "val", "test"]
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Distancia maxima entre perceptual hashes para marcar par como suspeito.
LIMIAR_DISTANCIA_HASH = 5

# Arquivos de saida da auditoria.
CSV_PATH = REPORTS_DIR / "auditoria_similaridade_visual.csv"
JSON_PATH = REPORTS_DIR / "auditoria_similaridade_visual_resumo.json"


def caminho_relativo_projeto(caminho):
    """Retorna o caminho relativo a raiz do projeto em formato legivel."""
    return str(caminho.relative_to(ROOT_DIR)).replace("\\", "/")


def verificar_pastas():
    """Confere se os tres splits existem antes de iniciar a auditoria."""
    for split in SPLITS:
        pasta_split = PROCESSED_DIR / split
        if not pasta_split.exists():
            raise FileNotFoundError(f"Pasta do split nao encontrada: {pasta_split}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def listar_imagens_split(split):
    """Lista imagens recursivamente dentro de um split."""
    pasta_split = PROCESSED_DIR / split
    return sorted(
        caminho
        for caminho in pasta_split.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
    )


def calcular_phash(caminho_imagem):
    """Calcula o perceptual hash da imagem usando PIL e imagehash.phash."""
    with Image.open(caminho_imagem) as imagem:
        return imagehash.phash(imagem.convert("RGB"))


def montar_registros_imagens():
    """Calcula hashes perceptuais para todas as imagens dos tres splits."""
    registros = []
    imagens_com_erro = []

    for split in SPLITS:
        for caminho_imagem in listar_imagens_split(split):
            try:
                hash_visual = calcular_phash(caminho_imagem)
            except (UnidentifiedImageError, OSError) as erro:
                imagens_com_erro.append(
                    {
                        "split": split,
                        "caminho": caminho_relativo_projeto(caminho_imagem),
                        "erro": str(erro),
                    }
                )
                continue

            registros.append(
                {
                    "split": split,
                    "classe": caminho_imagem.parent.name,
                    "caminho": caminho_imagem,
                    "hash_visual": hash_visual,
                }
            )

    return registros, imagens_com_erro


def comparar_splits(registros, split_1, split_2):
    """Compara todas as imagens de dois splits e retorna pares suspeitos."""
    imagens_1 = [registro for registro in registros if registro["split"] == split_1]
    imagens_2 = [registro for registro in registros if registro["split"] == split_2]
    pares_suspeitos = []

    for imagem_1 in imagens_1:
        for imagem_2 in imagens_2:
            distancia = imagem_1["hash_visual"] - imagem_2["hash_visual"]
            if distancia <= LIMIAR_DISTANCIA_HASH:
                pares_suspeitos.append(
                    {
                        "split_1": split_1,
                        "classe_1": imagem_1["classe"],
                        "caminho_1": caminho_relativo_projeto(imagem_1["caminho"]),
                        "split_2": split_2,
                        "classe_2": imagem_2["classe"],
                        "caminho_2": caminho_relativo_projeto(imagem_2["caminho"]),
                        "distancia_hash": int(distancia),
                        "mesma_classe": imagem_1["classe"] == imagem_2["classe"],
                    }
                )

    return pares_suspeitos


def gerar_pares_suspeitos(registros):
    """Compara train-val, train-test e val-test."""
    pares = []
    pares.extend(comparar_splits(registros, "train", "val"))
    pares.extend(comparar_splits(registros, "train", "test"))
    pares.extend(comparar_splits(registros, "val", "test"))
    return pares


def gerar_resumo(pares_suspeitos, imagens_com_erro):
    """Gera resumo JSON com totais por comparacao e por relacao de classe."""
    total_mesma_classe = sum(1 for par in pares_suspeitos if par["mesma_classe"])
    total_classes_diferentes = len(pares_suspeitos) - total_mesma_classe

    return {
        "total_pares_suspeitos_train_val": sum(
            1
            for par in pares_suspeitos
            if par["split_1"] == "train" and par["split_2"] == "val"
        ),
        "total_pares_suspeitos_train_test": sum(
            1
            for par in pares_suspeitos
            if par["split_1"] == "train" and par["split_2"] == "test"
        ),
        "total_pares_suspeitos_val_test": sum(
            1
            for par in pares_suspeitos
            if par["split_1"] == "val" and par["split_2"] == "test"
        ),
        "total_mesma_classe": total_mesma_classe,
        "total_classes_diferentes": total_classes_diferentes,
        "total_imagens_com_erro": len(imagens_com_erro),
        "imagens_com_erro": imagens_com_erro,
    }


def salvar_relatorios(pares_suspeitos, resumo):
    """Salva o CSV detalhado e o JSON resumido da auditoria."""
    colunas = [
        "split_1",
        "classe_1",
        "caminho_1",
        "split_2",
        "classe_2",
        "caminho_2",
        "distancia_hash",
        "mesma_classe",
    ]

    df = pd.DataFrame(pares_suspeitos, columns=colunas)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    JSON_PATH.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def imprimir_resumo(resumo):
    """Imprime um resumo claro dos pares visualmente suspeitos."""
    total_geral = (
        resumo["total_pares_suspeitos_train_val"]
        + resumo["total_pares_suspeitos_train_test"]
        + resumo["total_pares_suspeitos_val_test"]
    )

    print("Auditoria de similaridade visual por perceptual hash")
    print("-" * 55)
    print(f"Limiar usado: distancia de phash <= {LIMIAR_DISTANCIA_HASH}")
    print(f"Pares suspeitos train-val: {resumo['total_pares_suspeitos_train_val']}")
    print(f"Pares suspeitos train-test: {resumo['total_pares_suspeitos_train_test']}")
    print(f"Pares suspeitos val-test: {resumo['total_pares_suspeitos_val_test']}")
    print(f"Total geral de pares suspeitos: {total_geral}")
    print(f"Pares na mesma classe: {resumo['total_mesma_classe']}")
    print(f"Pares em classes diferentes: {resumo['total_classes_diferentes']}")

    if resumo["total_imagens_com_erro"] > 0:
        print(f"Imagens ignoradas por erro de leitura: {resumo['total_imagens_com_erro']}")

    print()
    print("Relatorios gerados:")
    print(CSV_PATH)
    print(JSON_PATH)


def main():
    """Executa apenas a auditoria visual, sem mover, apagar, splitar ou treinar."""
    verificar_pastas()
    registros, imagens_com_erro = montar_registros_imagens()
    pares_suspeitos = gerar_pares_suspeitos(registros)
    resumo = gerar_resumo(pares_suspeitos, imagens_com_erro)
    salvar_relatorios(pares_suspeitos, resumo)
    imprimir_resumo(resumo)


if __name__ == "__main__":
    main()
