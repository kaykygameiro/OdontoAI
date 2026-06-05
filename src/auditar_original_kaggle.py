from pathlib import Path
import csv
import json


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "original_kaggle"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# Extensoes de imagem consideradas na auditoria.
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Palavras usadas para reconhecer pastas com possiveis imagens aumentadas.
PALAVRAS_AUGMENTATION = ("augmented", "augment", "preview", "generated")

# Palavras extras que ajudam na observacao e na decisao manual posterior.
PALAVRAS_MONITORADAS = (
    "augmented",
    "augment",
    "preview",
    "generated",
    "data set",
    "dataset",
    "original",
)


def normalizar_caminho_para_busca(caminho):
    """Normaliza o caminho para facilitar a busca por palavras-chave."""
    return caminho.lower().replace("\\", "/").replace("_", " ").replace("-", " ")


def listar_palavras_encontradas(caminho_relativo):
    """Lista palavras monitoradas encontradas no caminho relativo."""
    texto = normalizar_caminho_para_busca(caminho_relativo)
    return [palavra for palavra in PALAVRAS_MONITORADAS if palavra in texto]


def caminho_parece_augmented(caminho_relativo):
    """Marca como augmented quando ha palavras fortes de augmentation no caminho."""
    texto = normalizar_caminho_para_busca(caminho_relativo)
    return any(palavra in texto for palavra in PALAVRAS_AUGMENTATION)


def caminho_parece_original(caminho_relativo, parece_augmented):
    """Aplica a heuristica de originalidade definida para a auditoria."""
    texto = normalizar_caminho_para_busca(caminho_relativo)
    return "original" in texto or not parece_augmented


def contar_imagens_diretas(pasta):
    """Conta apenas as imagens diretamente presentes em uma pasta."""
    return sum(
        1
        for caminho in pasta.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
    )


def montar_observacao(palavras_encontradas, parece_augmented, parece_original):
    """Cria uma observacao curta para orientar a revisao manual."""
    if palavras_encontradas:
        palavras = ", ".join(palavras_encontradas)
    else:
        palavras = "nenhuma palavra monitorada"

    if parece_augmented:
        classificacao = "possivel pasta augmented"
    elif parece_original:
        classificacao = "pasta aparentemente original"
    else:
        classificacao = "classificacao incerta"

    return f"{classificacao}; palavras encontradas: {palavras}."


def montar_registros():
    """Percorre o dataset original e registra somente pastas que contem imagens."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "Pasta do dataset original nao encontrada. "
            f"Esperado em: {DATA_DIR}"
        )

    registros = []

    for pasta in sorted(caminho for caminho in DATA_DIR.rglob("*") if caminho.is_dir()):
        total_imagens = contar_imagens_diretas(pasta)
        if total_imagens == 0:
            continue

        caminho_relativo = pasta.relative_to(DATA_DIR).as_posix()
        caminho_absoluto = str(pasta.resolve())
        profundidade = len(Path(caminho_relativo).parts)
        parece_augmented = caminho_parece_augmented(caminho_relativo)
        parece_original = caminho_parece_original(caminho_relativo, parece_augmented)
        palavras_encontradas = listar_palavras_encontradas(caminho_relativo)

        registros.append(
            {
                "caminho_relativo": caminho_relativo,
                "caminho_absoluto": caminho_absoluto,
                "total_imagens": total_imagens,
                "profundidade": profundidade,
                "parece_augmented": parece_augmented,
                "parece_original": parece_original,
                "observacao": montar_observacao(
                    palavras_encontradas,
                    parece_augmented,
                    parece_original,
                ),
            }
        )

    return registros


def gerar_resumo(registros):
    """Gera totais gerais da auditoria do dataset original."""
    total_imagens = sum(registro["total_imagens"] for registro in registros)
    total_augmented = sum(
        registro["total_imagens"]
        for registro in registros
        if registro["parece_augmented"]
    )
    total_originais = sum(
        registro["total_imagens"]
        for registro in registros
        if registro["parece_original"]
    )

    return {
        "total_pastas_com_imagens": len(registros),
        "total_imagens_encontradas": total_imagens,
        "total_imagens_em_pastas_parecem_augmented": total_augmented,
        "total_imagens_em_pastas_parecem_originais": total_originais,
    }


def salvar_relatorios(registros, resumo):
    """Salva o CSV detalhado e o JSON resumido em outputs/reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    caminho_csv = REPORTS_DIR / "auditoria_original_kaggle.csv"
    caminho_json = REPORTS_DIR / "auditoria_original_kaggle_resumo.json"

    colunas = [
        "caminho_relativo",
        "caminho_absoluto",
        "total_imagens",
        "profundidade",
        "parece_augmented",
        "parece_original",
        "observacao",
    ]

    with caminho_csv.open("w", newline="", encoding="utf-8-sig") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(registros)

    caminho_json.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return caminho_csv, caminho_json


def imprimir_arvore(registros):
    """Imprime uma arvore resumida das pastas que contem imagens."""
    print("Arvore resumida do dataset original do Kaggle")
    print("-" * 46)

    if not registros:
        print("Nenhuma pasta com imagens foi encontrada.")
        return

    for registro in registros:
        indentacao = "  " * max(registro["profundidade"] - 1, 0)
        marcador = "[AUGMENTED]" if registro["parece_augmented"] else "[ORIGINAL?]"
        print(
            f"{indentacao}{registro['caminho_relativo']} - "
            f"{registro['total_imagens']} imagens {marcador}"
        )


def imprimir_resumo(resumo, caminho_csv, caminho_json):
    """Mostra os totais principais e os relatorios gerados."""
    print()
    print("Resumo")
    print("-" * 6)
    print(f"Total de pastas com imagens: {resumo['total_pastas_com_imagens']}")
    print(f"Total de imagens encontradas: {resumo['total_imagens_encontradas']}")
    print(
        "Total em pastas que parecem augmented: "
        f"{resumo['total_imagens_em_pastas_parecem_augmented']}"
    )
    print(
        "Total em pastas que parecem originais: "
        f"{resumo['total_imagens_em_pastas_parecem_originais']}"
    )
    print()
    print("Relatorios gerados:")
    print(caminho_csv)
    print(caminho_json)


def main():
    """Executa apenas auditoria, sem mover, apagar, dividir ou treinar arquivos."""
    registros = montar_registros()
    resumo = gerar_resumo(registros)
    caminho_csv, caminho_json = salvar_relatorios(registros, resumo)
    imprimir_arvore(registros)
    imprimir_resumo(resumo, caminho_csv, caminho_json)


if __name__ == "__main__":
    main()
