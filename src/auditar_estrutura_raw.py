from pathlib import Path
import csv
import json


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw" / "diseases"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# Extensoes de imagem aceitas na auditoria.
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Palavras-chave usadas para marcar caminhos que parecem conter dados aumentados.
PALAVRAS_CHAVE_AUGMENTED = (
    "augmented",
    "augment",
    "preview",
    "data set",
    "dataset",
    "generated",
)


def caminho_parece_augmented(caminho_relativo):
    """Verifica se o caminho contem indicios de imagens aumentadas."""
    texto = caminho_relativo.lower().replace("\\", "/").replace("_", " ")
    return any(palavra in texto for palavra in PALAVRAS_CHAVE_AUGMENTED)


def listar_pastas():
    """Lista todas as pastas dentro do dataset raw, incluindo as classes raiz."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {DATA_DIR}")

    pastas = [caminho for caminho in DATA_DIR.rglob("*") if caminho.is_dir()]
    return sorted(pastas)


def contar_imagens_diretas(pasta):
    """Conta apenas imagens diretamente dentro da pasta informada."""
    return sum(
        1
        for caminho in pasta.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
    )


def montar_observacao(parece_augmented, total_imagens):
    """Cria uma observacao simples para facilitar a leitura do CSV."""
    if parece_augmented:
        return "Possivel pasta com imagens aumentadas; nao usar antes do split."

    if total_imagens == 0:
        return "Pasta sem imagens diretamente dentro dela."

    return "Pasta aparentemente original."


def montar_registros():
    """Monta uma linha de auditoria para cada pasta encontrada no dataset raw."""
    registros = []

    for pasta in listar_pastas():
        relativo = pasta.relative_to(DATA_DIR)
        partes = relativo.parts
        classe_raiz = partes[0] if partes else ""
        caminho_relativo = relativo.as_posix()
        total_imagens = contar_imagens_diretas(pasta)
        parece_augmented = caminho_parece_augmented(caminho_relativo)
        profundidade = len(partes)

        registros.append(
            {
                "classe_raiz": classe_raiz,
                "caminho_relativo": caminho_relativo,
                "total_imagens": total_imagens,
                "parece_augmented": parece_augmented,
                "profundidade": profundidade,
                "observacao": montar_observacao(parece_augmented, total_imagens),
            }
        )

    return registros


def somar_por_classe(registros, apenas_augmented=None):
    """Soma imagens por classe, opcionalmente filtrando pastas augmented/originais."""
    totais = {}

    for registro in registros:
        if apenas_augmented is not None and registro["parece_augmented"] != apenas_augmented:
            continue

        classe = registro["classe_raiz"]
        totais[classe] = totais.get(classe, 0) + registro["total_imagens"]

    return dict(sorted(totais.items()))


def gerar_resumo(registros):
    """Gera os totais gerais e por classe para o resumo em JSON."""
    total_geral = sum(registro["total_imagens"] for registro in registros)
    total_augmented = sum(
        registro["total_imagens"]
        for registro in registros
        if registro["parece_augmented"]
    )
    total_original = total_geral - total_augmented

    return {
        "total_imagens_encontradas": total_geral,
        "total_imagens_em_pastas_aparentemente_originais": total_original,
        "total_imagens_em_pastas_parecem_augmented": total_augmented,
        "total_por_classe": somar_por_classe(registros),
        "total_por_classe_pastas_augmented": somar_por_classe(
            registros,
            apenas_augmented=True,
        ),
        "total_por_classe_pastas_aparentemente_originais": somar_por_classe(
            registros,
            apenas_augmented=False,
        ),
    }


def salvar_relatorios(registros, resumo):
    """Salva o CSV detalhado e o JSON de resumo da auditoria."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    caminho_csv = REPORTS_DIR / "auditoria_estrutura_raw.csv"
    caminho_json = REPORTS_DIR / "auditoria_estrutura_raw_resumo.json"

    colunas = [
        "classe_raiz",
        "caminho_relativo",
        "total_imagens",
        "parece_augmented",
        "profundidade",
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
    """Imprime uma arvore resumida com classe, subpastas, total e classificacao."""
    print("Arvore resumida do dataset raw")
    print("-" * 34)

    classe_atual = None
    for registro in registros:
        classe = registro["classe_raiz"]
        if classe != classe_atual:
            classe_atual = classe
            print(classe)

        partes = Path(registro["caminho_relativo"]).parts
        nome_pasta = partes[-1]
        nivel = max(registro["profundidade"] - 1, 0)
        indentacao = "  " * (nivel + 1)
        tipo = "augmented" if registro["parece_augmented"] else "original"

        print(f"{indentacao}{nome_pasta}: {registro['total_imagens']} imagens ({tipo})")


def imprimir_resumo(resumo, caminho_csv, caminho_json):
    """Imprime os totais principais e os caminhos dos relatorios gerados."""
    print()
    print("Resumo")
    print("-" * 6)
    print(f"Total de imagens encontradas: {resumo['total_imagens_encontradas']}")
    print(
        "Total em pastas aparentemente originais: "
        f"{resumo['total_imagens_em_pastas_aparentemente_originais']}"
    )
    print(
        "Total em pastas que parecem augmented: "
        f"{resumo['total_imagens_em_pastas_parecem_augmented']}"
    )
    print()
    print("Relatorios gerados:")
    print(caminho_csv)
    print(caminho_json)


def main():
    """Executa apenas a auditoria, sem mover, apagar, dividir ou treinar dados."""
    registros = montar_registros()
    resumo = gerar_resumo(registros)
    caminho_csv, caminho_json = salvar_relatorios(registros, resumo)
    imprimir_arvore(registros)
    imprimir_resumo(resumo, caminho_csv, caminho_json)


if __name__ == "__main__":
    main()
