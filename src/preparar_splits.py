# Script legado de split simples por imagem.
#
# Não foi utilizado no experimento final.
#
# O pipeline oficial utiliza preparar_splits_por_grupos.py,
#
# que agrupa imagens visualmente semelhantes por pHash antes do split.

from pathlib import Path
import json
import random
import shutil

import pandas as pd


# Seed fixa para garantir reprodutibilidade do split.
SEED = 42

# Proporções desejadas para cada conjunto.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Extensões de imagem aceitas.
EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classes esperadas no dataset limpo.
CLASSES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth_Ulcer",
    "Tooth_Discoloration",
]

# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT_DIR / "data" / "clean" / "diseases"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"


def listar_imagens(pasta_classe: Path) -> list[Path]:
    """Lista imagens recursivamente dentro da pasta de uma classe limpa."""
    return sorted(
        arquivo
        for arquivo in pasta_classe.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS
    )


def pasta_tem_arquivos(pasta: Path) -> bool:
    """Verifica se a pasta existe e contém algum arquivo."""
    return pasta.exists() and any(item.is_file() for item in pasta.rglob("*"))


def confirmar_recriacao_processed() -> None:
    """Pergunta antes de apagar data/processed quando há arquivos."""
    if not pasta_tem_arquivos(PROCESSED_DIR):
        return

    resposta = input(
        f"A pasta '{PROCESSED_DIR}' já existe e contém arquivos. "
        "Deseja apagar e recriar? [s/N]: "
    ).strip().lower()

    if resposta not in {"s", "sim", "y", "yes"}:
        raise SystemExit("Operação cancelada. Nenhum arquivo foi apagado.")

    shutil.rmtree(PROCESSED_DIR)


def criar_estrutura_splits() -> None:
    """Cria as pastas train, val e test para todas as classes."""
    for split in ["train", "val", "test"]:
        for classe in CLASSES:
            (PROCESSED_DIR / split / classe).mkdir(parents=True, exist_ok=True)


def dividir_imagens(imagens: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Divide as imagens de uma classe em treino, validação e teste."""
    imagens_embaralhadas = imagens.copy()
    random.shuffle(imagens_embaralhadas)

    total = len(imagens_embaralhadas)
    qtd_train = int(total * TRAIN_RATIO)
    qtd_val = int(total * VAL_RATIO)

    imagens_train = imagens_embaralhadas[:qtd_train]
    imagens_val = imagens_embaralhadas[qtd_train : qtd_train + qtd_val]
    imagens_test = imagens_embaralhadas[qtd_train + qtd_val :]

    return imagens_train, imagens_val, imagens_test


def destino_sem_sobrescrever(destino_classe: Path, imagem: Path) -> Path:
    """Gera um caminho de destino sem sobrescrever arquivos de mesmo nome."""
    destino = destino_classe / imagem.name
    if not destino.exists():
        return destino

    contador = 2
    while True:
        destino = destino_classe / f"{imagem.stem}_{contador}{imagem.suffix}"
        if not destino.exists():
            return destino
        contador += 1


def copiar_imagens(imagens: list[Path], split: str, classe: str) -> None:
    """Copia as imagens para o split correspondente em data/processed."""
    destino_classe = PROCESSED_DIR / split / classe

    for imagem in imagens:
        destino = destino_sem_sobrescrever(destino_classe, imagem)
        shutil.copy2(imagem, destino)


def salvar_metadados_classes() -> tuple[Path, Path]:
    """Salva classes.txt e class_to_idx.json para o pipeline final."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    caminho_classes = REPORTS_DIR / "classes.txt"
    caminho_class_to_idx = REPORTS_DIR / "class_to_idx.json"
    class_to_idx = {classe: indice for indice, classe in enumerate(CLASSES)}

    caminho_classes.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    caminho_class_to_idx.write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return caminho_classes, caminho_class_to_idx


def main() -> None:
    """Cria splits estratificados usando apenas o dataset limpo."""
    random.seed(SEED)

    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de imagens limpas não encontrada: {CLEAN_DIR}\n"
            "Execute primeiro o script src/limpar_dataset.py."
        )

    confirmar_recriacao_processed()
    criar_estrutura_splits()

    linhas_relatorio = []
    total_geral = 0
    total_train = 0
    total_val = 0
    total_test = 0

    for classe in CLASSES:
        pasta_classe = CLEAN_DIR / classe

        if not pasta_classe.exists():
            raise FileNotFoundError(f"Pasta da classe não encontrada: {pasta_classe}")

        imagens = listar_imagens(pasta_classe)
        imagens_train, imagens_val, imagens_test = dividir_imagens(imagens)

        copiar_imagens(imagens_train, "train", classe)
        copiar_imagens(imagens_val, "val", classe)
        copiar_imagens(imagens_test, "test", classe)

        qtd_train = len(imagens_train)
        qtd_val = len(imagens_val)
        qtd_test = len(imagens_test)
        qtd_total = len(imagens)

        total_geral += qtd_total
        total_train += qtd_train
        total_val += qtd_val
        total_test += qtd_test

        linhas_relatorio.append(
            {
                "classe": classe,
                "treino": qtd_train,
                "validacao": qtd_val,
                "teste": qtd_test,
                "total": qtd_total,
            }
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_relatorio = pd.DataFrame(linhas_relatorio)
    caminho_csv = REPORTS_DIR / "distribuicao_splits.csv"
    df_relatorio.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    caminho_classes, caminho_class_to_idx = salvar_metadados_classes()

    print("\nDistribuição dos splits por classe:")
    print(df_relatorio.to_string(index=False))

    print("\nTotais gerais:")
    print(f"Total geral de imagens encontradas: {total_geral}")
    print(f"Total em treino: {total_train}")
    print(f"Total em validação: {total_val}")
    print(f"Total em teste: {total_test}")

    print("\nArquivos salvos:")
    print(caminho_csv)
    print(caminho_classes)
    print(caminho_class_to_idx)


if __name__ == "__main__":
    main()
