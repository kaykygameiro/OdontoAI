from pathlib import Path
import csv
import json
import shutil


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = ROOT_DIR / "data" / "original_kaggle"
RAW_DIR = ROOT_DIR / "data" / "raw" / "diseases"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# Extensoes de imagem aceitas para montar o dataset oficial.
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classes finais usadas no problema de classificacao.
CLASSES_FINAIS = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth_Ulcer",
    "Tooth_Discoloration",
]

# Mapeamento explicito das fontes oficiais. As pastas augmented e YOLO ficam fora.
FONTES_OFICIAIS = {
    "Calculus": ORIGINAL_DIR / "Calculus" / "Calculus",
    "Caries": (
        ORIGINAL_DIR
        / "Data caries"
        / "Data caries"
        / "caries orignal data set"
        / "done"
    ),
    "Gingivitis": ORIGINAL_DIR / "Gingivitis" / "Gingivitis",
    "Hypodontia": ORIGINAL_DIR / "hypodontia" / "hypodontia",
    "Mouth_Ulcer": (
        ORIGINAL_DIR
        / "Mouth Ulcer"
        / "Mouth Ulcer"
        / "ulcer original dataset"
        / "ulcer original dataset"
    ),
    "Tooth_Discoloration": (
        ORIGINAL_DIR
        / "Tooth Discoloration"
        / "Tooth Discoloration"
        / "tooth discoloration original dataset"
        / "tooth discoloration original dataset"
    ),
}

# Caminhos conhecidos que nao devem entrar no dataset oficial.
CAMINHOS_IGNORADOS = [
    ORIGINAL_DIR
    / "Data caries"
    / "Data caries"
    / "caries augmented data set"
    / "preview",
    ORIGINAL_DIR
    / "Mouth Ulcer"
    / "Mouth Ulcer"
    / "Mouth_Ulcer_augmented_DataSet"
    / "preview",
    ORIGINAL_DIR
    / "Tooth Discoloration"
    / "Tooth Discoloration"
    / "Tooth_discoloration_augmented_dataser"
    / "preview",
    ORIGINAL_DIR
    / "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset",
]


def caminho_relativo_projeto(caminho):
    """Retorna um caminho relativo a raiz do projeto para relatorios legiveis."""
    return str(caminho.relative_to(ROOT_DIR)).replace("\\", "/")


def listar_imagens(pasta):
    """Lista imagens recursivamente dentro de uma pasta de origem."""
    return sorted(
        caminho
        for caminho in pasta.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
    )


def confirmar_recriacao_raw():
    """Pergunta antes de apagar data/raw/diseases quando ela ja existe."""
    if not RAW_DIR.exists():
        return

    resposta = input(
        f"A pasta {RAW_DIR} ja existe. Apagar e recriar? [s/N]: "
    ).strip().lower()

    if resposta not in {"s", "sim", "y", "yes"}:
        raise SystemExit("Operacao cancelada. Nenhum arquivo foi alterado.")

    shutil.rmtree(RAW_DIR)


def validar_fontes():
    """Confere se o dataset original e todas as fontes oficiais existem."""
    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"Dataset original nao encontrado: {ORIGINAL_DIR}")

    fontes_faltando = [
        f"{classe}: {fonte}"
        for classe, fonte in FONTES_OFICIAIS.items()
        if not fonte.exists()
    ]

    if fontes_faltando:
        mensagem = "\n".join(fontes_faltando)
        raise FileNotFoundError(f"Fontes oficiais nao encontradas:\n{mensagem}")


def preparar_pastas_saida():
    """Cria a estrutura data/raw/diseases/<classe> e a pasta de relatorios."""
    confirmar_recriacao_raw()

    for classe in CLASSES_FINAIS:
        (RAW_DIR / classe).mkdir(parents=True, exist_ok=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def caminho_destino_unico(destino_dir, imagem_origem, nomes_usados):
    """Evita sobrescrever arquivos quando nomes repetidos aparecem na origem."""
    destino = destino_dir / imagem_origem.name

    if destino.name not in nomes_usados:
        nomes_usados.add(destino.name)
        return destino

    contador = 2
    while True:
        destino = destino_dir / f"{imagem_origem.stem}_{contador}{imagem_origem.suffix}"
        if destino.name not in nomes_usados:
            nomes_usados.add(destino.name)
            return destino
        contador += 1


def copiar_classe(classe, fonte):
    """Copia imagens de uma fonte oficial para a pasta final da classe."""
    destino_dir = RAW_DIR / classe
    nomes_usados = set()
    linhas = []

    for imagem_origem in listar_imagens(fonte):
        destino = caminho_destino_unico(destino_dir, imagem_origem, nomes_usados)
        shutil.copy2(imagem_origem, destino)

        linhas.append(
            {
                "classe": classe,
                "arquivo": destino.name,
                "caminho_origem": caminho_relativo_projeto(imagem_origem),
                "caminho_destino": caminho_relativo_projeto(destino),
                "status": "copiada",
                "observacao": "Imagem copiada da fonte oficial original.",
            }
        )

    return linhas


def montar_dataset():
    """Executa a montagem do dataset oficial sem alterar o Kaggle bruto."""
    validar_fontes()
    preparar_pastas_saida()

    linhas_detalhadas = []
    total_por_classe = {}

    for classe in CLASSES_FINAIS:
        fonte = FONTES_OFICIAIS[classe]
        linhas_classe = copiar_classe(classe, fonte)
        linhas_detalhadas.extend(linhas_classe)
        total_por_classe[classe] = len(linhas_classe)
        print(f"{classe}: {len(linhas_classe)} imagens copiadas")

    return linhas_detalhadas, total_por_classe


def montar_resumo(total_por_classe):
    """Cria o resumo JSON com totais, fontes usadas e caminhos ignorados."""
    fontes_usadas = {
        classe: caminho_relativo_projeto(fonte)
        for classe, fonte in FONTES_OFICIAIS.items()
    }
    caminhos_ignorados = [
        caminho_relativo_projeto(caminho)
        for caminho in CAMINHOS_IGNORADOS
    ]

    return {
        "total_por_classe": total_por_classe,
        "total_geral": sum(total_por_classe.values()),
        "caminhos_origem_usados": fontes_usadas,
        "caminhos_ignorados_augmented_ou_yolo": caminhos_ignorados,
    }


def salvar_relatorios(linhas_detalhadas, resumo):
    """Salva o relatorio detalhado em CSV e o resumo em JSON."""
    caminho_csv = REPORTS_DIR / "dataset_oficial_detalhado.csv"
    caminho_json = REPORTS_DIR / "dataset_oficial_resumo.json"

    colunas = [
        "classe",
        "arquivo",
        "caminho_origem",
        "caminho_destino",
        "status",
        "observacao",
    ]

    with caminho_csv.open("w", newline="", encoding="utf-8-sig") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas_detalhadas)

    caminho_json.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return caminho_csv, caminho_json


def imprimir_resumo(resumo, caminho_csv, caminho_json):
    """Imprime os totais finais e os relatorios gerados."""
    print()
    print("Dataset oficial montado")
    print("-" * 24)
    for classe, total in resumo["total_por_classe"].items():
        print(f"{classe}: {total} imagens")
    print(f"Total geral: {resumo['total_geral']} imagens")
    print()
    print("Relatorios gerados:")
    print(caminho_csv)
    print(caminho_json)


def main():
    """Monta apenas data/raw/diseases, sem mover arquivos, splitar ou treinar."""
    linhas_detalhadas, total_por_classe = montar_dataset()
    resumo = montar_resumo(total_por_classe)
    caminho_csv, caminho_json = salvar_relatorios(linhas_detalhadas, resumo)
    imprimir_resumo(resumo, caminho_csv, caminho_json)


if __name__ == "__main__":
    main()
