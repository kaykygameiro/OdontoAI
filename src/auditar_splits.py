from pathlib import Path
import hashlib
import json

import pandas as pd


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# Splits avaliados e extensões de imagem aceitas.
SPLITS = ["train", "val", "test"]
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calcular_md5(caminho_arquivo, tamanho_bloco=1024 * 1024):
    """Calcula o hash MD5 do arquivo em blocos para evitar alto uso de memória."""
    md5 = hashlib.md5()
    with caminho_arquivo.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(tamanho_bloco), b""):
            md5.update(bloco)
    return md5.hexdigest()


def listar_imagens(split):
    """Lista imagens recursivamente em um split específico."""
    pasta_split = DATA_DIR / split
    if not pasta_split.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta_split}")

    imagens = [
        caminho
        for caminho in pasta_split.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
    ]
    return sorted(imagens)


def montar_registros():
    """Cria os registros básicos com caminho, split, nome de arquivo e hash MD5."""
    registros = []

    for split in SPLITS:
        for caminho in listar_imagens(split):
            registros.append(
                {
                    "split": split,
                    "classe": caminho.parent.name,
                    "arquivo": caminho.name,
                    "caminho": str(caminho.relative_to(ROOT_DIR)),
                    "md5": calcular_md5(caminho),
                }
            )

    return pd.DataFrame(registros)


def contar_duplicatas_internas(df, split, coluna):
    """Conta grupos duplicados dentro de um split para hash ou nome de arquivo."""
    dados_split = df[df["split"] == split]
    contagens = dados_split.groupby(coluna).size()
    return int((contagens > 1).sum())


def contar_duplicatas_entre_splits(df, split_a, split_b, coluna):
    """Conta valores repetidos entre dois splits para hash ou nome de arquivo."""
    valores_a = set(df.loc[df["split"] == split_a, coluna])
    valores_b = set(df.loc[df["split"] == split_b, coluna])
    return len(valores_a.intersection(valores_b))


def adicionar_indicadores(df):
    """Adiciona colunas úteis para identificar duplicatas no CSV detalhado."""
    if df.empty:
        df["duplicata_md5_mesmo_split"] = []
        df["duplicata_md5_entre_splits"] = []
        df["nome_repetido_entre_splits"] = []
        return df

    df = df.copy()
    df["duplicata_md5_mesmo_split"] = (
        df.groupby(["split", "md5"])["md5"].transform("size") > 1
    )
    df["duplicata_md5_entre_splits"] = (
        df.groupby("md5")["split"].transform("nunique") > 1
    )
    df["nome_repetido_entre_splits"] = (
        df.groupby("arquivo")["split"].transform("nunique") > 1
    )
    return df


def gerar_resumo(df):
    """Monta o resumo de totais, duplicatas exatas e nomes repetidos."""
    totais_por_split = {
        split: int((df["split"] == split).sum())
        for split in SPLITS
    }

    resumo = {
        "total_imagens_train": totais_por_split["train"],
        "total_imagens_val": totais_por_split["val"],
        "total_imagens_test": totais_por_split["test"],
        "total_geral": int(len(df)),
        "duplicatas_md5": {
            "dentro_train": contar_duplicatas_internas(df, "train", "md5"),
            "dentro_val": contar_duplicatas_internas(df, "val", "md5"),
            "dentro_test": contar_duplicatas_internas(df, "test", "md5"),
            "entre_train_val": contar_duplicatas_entre_splits(df, "train", "val", "md5"),
            "entre_train_test": contar_duplicatas_entre_splits(df, "train", "test", "md5"),
            "entre_val_test": contar_duplicatas_entre_splits(df, "val", "test", "md5"),
        },
        "nomes_repetidos": {
            "dentro_train": contar_duplicatas_internas(df, "train", "arquivo"),
            "dentro_val": contar_duplicatas_internas(df, "val", "arquivo"),
            "dentro_test": contar_duplicatas_internas(df, "test", "arquivo"),
            "entre_train_val": contar_duplicatas_entre_splits(
                df, "train", "val", "arquivo"
            ),
            "entre_train_test": contar_duplicatas_entre_splits(
                df, "train", "test", "arquivo"
            ),
            "entre_val_test": contar_duplicatas_entre_splits(
                df, "val", "test", "arquivo"
            ),
        },
    }

    return resumo


def salvar_relatorios(df, resumo):
    """Salva o CSV detalhado e o JSON de resumo da auditoria."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    caminho_csv = REPORTS_DIR / "auditoria_splits_detalhada.csv"
    caminho_json = REPORTS_DIR / "auditoria_splits_resumo.json"

    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    caminho_json.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return caminho_csv, caminho_json


def imprimir_resumo(resumo, caminho_csv, caminho_json):
    """Imprime um resumo claro dos resultados encontrados."""
    duplicatas = resumo["duplicatas_md5"]
    nomes = resumo["nomes_repetidos"]
    total_duplicatas_entre_splits = (
        duplicatas["entre_train_val"]
        + duplicatas["entre_train_test"]
        + duplicatas["entre_val_test"]
    )

    print("Resumo da auditoria dos splits")
    print("-" * 35)
    print(f"Total de imagens em train: {resumo['total_imagens_train']}")
    print(f"Total de imagens em val: {resumo['total_imagens_val']}")
    print(f"Total de imagens em test: {resumo['total_imagens_test']}")
    print(f"Total geral: {resumo['total_geral']}")
    print()
    print("Duplicatas exatas por MD5:")
    print(f"  Dentro de train: {duplicatas['dentro_train']}")
    print(f"  Dentro de val: {duplicatas['dentro_val']}")
    print(f"  Dentro de test: {duplicatas['dentro_test']}")
    print(f"  Entre train e val: {duplicatas['entre_train_val']}")
    print(f"  Entre train e test: {duplicatas['entre_train_test']}")
    print(f"  Entre val e test: {duplicatas['entre_val_test']}")
    print()
    print("Nomes de arquivos repetidos:")
    print(f"  Dentro de train: {nomes['dentro_train']}")
    print(f"  Dentro de val: {nomes['dentro_val']}")
    print(f"  Dentro de test: {nomes['dentro_test']}")
    print(f"  Entre train e val: {nomes['entre_train_val']}")
    print(f"  Entre train e test: {nomes['entre_train_test']}")
    print(f"  Entre val e test: {nomes['entre_val_test']}")
    print()

    if duplicatas["entre_train_test"] > 0:
        print("ATENÇÃO: existem imagens idênticas entre treino e teste.")

    if total_duplicatas_entre_splits == 0:
        print("Nenhuma duplicata exata encontrada entre train, val e test.")

    print()
    print("Relatórios gerados:")
    print(caminho_csv)
    print(caminho_json)


def main():
    """Executa a auditoria sem mover, apagar, alterar arquivos ou treinar modelos."""
    df = montar_registros()
    df = adicionar_indicadores(df)
    resumo = gerar_resumo(df)
    caminho_csv, caminho_json = salvar_relatorios(df, resumo)
    imprimir_resumo(resumo, caminho_csv, caminho_json)


if __name__ == "__main__":
    main()
