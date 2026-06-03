from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd


# Caminhos principais do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw" / "diseases"
CLEAN_ROOT_DIR = ROOT_DIR / "data" / "clean"
CLEAN_DIR = CLEAN_ROOT_DIR / "diseases"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

# Classes esperadas no dataset.
CLASSES_ESPERADAS = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth_Ulcer",
    "Tooth_Discoloration",
]

# Extensões de imagem aceitas.
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calcular_md5(caminho_arquivo, tamanho_bloco=1024 * 1024):
    """Calcula o hash MD5 do arquivo em blocos para poupar memória."""
    md5 = hashlib.md5()
    with caminho_arquivo.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(tamanho_bloco), b""):
            md5.update(bloco)
    return md5.hexdigest()


def confirmar_recriacao_clean():
    """Pergunta antes de apagar data/clean quando ela já existe."""
    if not CLEAN_ROOT_DIR.exists():
        return

    resposta = input(
        f"A pasta {CLEAN_ROOT_DIR} já existe. Apagar e recriar? [s/N]: "
    ).strip().lower()

    if resposta not in {"s", "sim", "y", "yes"}:
        raise SystemExit("Operação cancelada. Nenhum arquivo foi alterado.")

    shutil.rmtree(CLEAN_ROOT_DIR)


def preparar_pastas_saida():
    """Cria a estrutura data/clean/diseases/<classe> e a pasta de relatórios."""
    confirmar_recriacao_clean()

    for classe in CLASSES_ESPERADAS:
        (CLEAN_DIR / classe).mkdir(parents=True, exist_ok=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def listar_imagens_raw():
    """Procura imagens recursivamente dentro de data/raw/diseases/<classe>."""
    registros = []

    for classe in CLASSES_ESPERADAS:
        pasta_classe = RAW_DIR / classe
        if not pasta_classe.exists():
            raise FileNotFoundError(f"Pasta de classe não encontrada: {pasta_classe}")

        imagens = sorted(
            caminho
            for caminho in pasta_classe.rglob("*")
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
        )

        for caminho in imagens:
            registros.append(
                {
                    "classe_original": classe,
                    "caminho_original": caminho,
                    "md5": calcular_md5(caminho),
                }
            )

    return registros


def caminho_destino_unico(caminho_original, classe, nomes_usados_por_classe):
    """Define um nome de destino sem sobrescrever arquivos já copiados."""
    destino_dir = CLEAN_DIR / classe
    destino = destino_dir / caminho_original.name

    if destino.name not in nomes_usados_por_classe[classe]:
        nomes_usados_por_classe[classe].add(destino.name)
        return destino

    contador = 2
    while True:
        destino = destino_dir / f"{caminho_original.stem}_{contador}{caminho_original.suffix}"
        if destino.name not in nomes_usados_por_classe[classe]:
            nomes_usados_por_classe[classe].add(destino.name)
            return destino
        contador += 1


def limpar_por_md5(registros):
    """Agrupa imagens por MD5, copia únicas válidas e registra remoções."""
    grupos_por_md5 = {}
    for registro in registros:
        grupos_por_md5.setdefault(registro["md5"], []).append(registro)

    linhas_detalhadas = []
    conflitos_rotulo = []
    duplicatas_removidas = []
    nomes_usados_por_classe = {classe: set() for classe in CLASSES_ESPERADAS}
    total_por_classe = {classe: 0 for classe in CLASSES_ESPERADAS}

    for md5, ocorrencias in sorted(grupos_por_md5.items()):
        classes_do_md5 = sorted({item["classe_original"] for item in ocorrencias})

        if len(classes_do_md5) > 1:
            observacao = "Conflito de rótulo: mesmo MD5 encontrado em mais de uma classe."
            for item in ocorrencias:
                linha = {
                    "md5": md5,
                    "classe_original": item["classe_original"],
                    "caminho_original": str(item["caminho_original"].relative_to(ROOT_DIR)),
                    "status": "removida_conflito_rotulo",
                    "caminho_copiado_se_houver": "",
                    "observacao": observacao,
                }
                linhas_detalhadas.append(linha)
                conflitos_rotulo.append(linha)
            continue

        # Quando o MD5 pertence a uma única classe, mantemos a primeira ocorrência.
        ocorrencias_ordenadas = sorted(
            ocorrencias,
            key=lambda item: str(item["caminho_original"]).lower(),
        )
        mantida = ocorrencias_ordenadas[0]
        classe = mantida["classe_original"]
        destino = caminho_destino_unico(
            mantida["caminho_original"],
            classe,
            nomes_usados_por_classe,
        )
        shutil.copy2(mantida["caminho_original"], destino)
        total_por_classe[classe] += 1

        linhas_detalhadas.append(
            {
                "md5": md5,
                "classe_original": classe,
                "caminho_original": str(mantida["caminho_original"].relative_to(ROOT_DIR)),
                "status": "mantida",
                "caminho_copiado_se_houver": str(destino.relative_to(ROOT_DIR)),
                "observacao": "Imagem única mantida no dataset limpo.",
            }
        )

        for duplicata in ocorrencias_ordenadas[1:]:
            linha = {
                "md5": md5,
                "classe_original": duplicata["classe_original"],
                "caminho_original": str(duplicata["caminho_original"].relative_to(ROOT_DIR)),
                "status": "removida_duplicata_mesma_classe",
                "caminho_copiado_se_houver": "",
                "observacao": "Duplicata exata dentro da mesma classe; uma ocorrência foi mantida.",
            }
            linhas_detalhadas.append(linha)
            duplicatas_removidas.append(linha)

    return {
        "linhas_detalhadas": linhas_detalhadas,
        "conflitos_rotulo": conflitos_rotulo,
        "duplicatas_removidas": duplicatas_removidas,
        "total_por_classe": total_por_classe,
    }


def salvar_relatorios(total_original, resultado):
    """Salva os relatórios JSON e CSV da limpeza."""
    total_por_classe = resultado["total_por_classe"]
    total_mantido = sum(total_por_classe.values())
    total_duplicatas = len(resultado["duplicatas_removidas"])
    total_conflitos = len(resultado["conflitos_rotulo"])

    resumo = {
        "total_imagens_originais": total_original,
        "total_imagens_unicas_mantidas": total_mantido,
        "total_duplicatas_removidas_dentro_mesma_classe": total_duplicatas,
        "total_imagens_removidas_por_conflito_rotulo": total_conflitos,
        "total_final_por_classe": total_por_classe,
    }

    detalhada = pd.DataFrame(
        resultado["linhas_detalhadas"],
        columns=[
            "md5",
            "classe_original",
            "caminho_original",
            "status",
            "caminho_copiado_se_houver",
            "observacao",
        ],
    )
    conflitos = pd.DataFrame(resultado["conflitos_rotulo"], columns=detalhada.columns)
    duplicatas = pd.DataFrame(resultado["duplicatas_removidas"], columns=detalhada.columns)
    distribuicao = pd.DataFrame(
        [
            {"classe": classe, "total": total_por_classe[classe]}
            for classe in CLASSES_ESPERADAS
        ]
    )

    caminho_resumo = REPORTS_DIR / "limpeza_dataset_resumo.json"
    caminho_detalhada = REPORTS_DIR / "limpeza_dataset_detalhada.csv"
    caminho_conflitos = REPORTS_DIR / "conflitos_rotulo_md5.csv"
    caminho_duplicatas = REPORTS_DIR / "duplicatas_removidas.csv"
    caminho_distribuicao = REPORTS_DIR / "distribuicao_dataset_limpo.csv"

    caminho_resumo.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    detalhada.to_csv(caminho_detalhada, index=False, encoding="utf-8-sig")
    conflitos.to_csv(caminho_conflitos, index=False, encoding="utf-8-sig")
    duplicatas.to_csv(caminho_duplicatas, index=False, encoding="utf-8-sig")
    distribuicao.to_csv(caminho_distribuicao, index=False, encoding="utf-8-sig")

    return resumo


def imprimir_resumo(resumo):
    """Imprime os totais principais após a limpeza."""
    print("Resumo da limpeza do dataset")
    print("-" * 31)
    print(f"Total original: {resumo['total_imagens_originais']}")
    print(f"Total mantido: {resumo['total_imagens_unicas_mantidas']}")
    print(
        "Total removido por duplicata: "
        f"{resumo['total_duplicatas_removidas_dentro_mesma_classe']}"
    )
    print(
        "Total removido por conflito: "
        f"{resumo['total_imagens_removidas_por_conflito_rotulo']}"
    )
    print()
    print("Distribuição final por classe:")
    for classe, total in resumo["total_final_por_classe"].items():
        print(f"  {classe}: {total}")
    print()
    print("Relatórios salvos em:")
    print(REPORTS_DIR)


def main():
    """Executa a limpeza sem alterar data/raw, data/processed ou treinar modelos."""
    preparar_pastas_saida()
    registros = listar_imagens_raw()
    resultado = limpar_por_md5(registros)
    resumo = salvar_relatorios(len(registros), resultado)
    imprimir_resumo(resumo)


if __name__ == "__main__":
    main()
