from pathlib import Path
import json
import random
import shutil

import pandas as pd
from PIL import Image, UnidentifiedImageError

try:
    import imagehash
except ImportError as erro:
    raise SystemExit(
        "A biblioteca ImageHash nao esta instalada. Instale com:\n"
        "pip install ImageHash"
    ) from erro


# Seed fixa para reprodutibilidade do split agrupado.
SEED = 42

# Proporcoes desejadas para os conjuntos finais.
RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Distancia maxima de pHash para unir imagens no mesmo grupo visual.
LIMIAR_DISTANCIA_HASH = 5

# Extensoes de imagem aceitas.
EXTENSOES_ACEITAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

# Arquivos de relatorio.
DISTRIBUICAO_CSV = REPORTS_DIR / "distribuicao_splits.csv"
CLASSES_TXT = REPORTS_DIR / "classes.txt"
CLASS_TO_IDX_JSON = REPORTS_DIR / "class_to_idx.json"
GRUPOS_CSV = REPORTS_DIR / "grupos_similaridade_visual.csv"
RESUMO_JSON = REPORTS_DIR / "resumo_grupos_similaridade_visual.json"


def caminho_relativo_projeto(caminho):
    """Retorna caminho relativo a raiz do projeto em formato legivel."""
    return str(caminho.relative_to(ROOT_DIR)).replace("\\", "/")


def listar_imagens(pasta_classe):
    """Lista imagens recursivamente dentro da pasta limpa de uma classe."""
    return sorted(
        caminho
        for caminho in pasta_classe.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
    )


def pasta_tem_arquivos(pasta):
    """Verifica se uma pasta existe e contem algum arquivo."""
    return pasta.exists() and any(item.is_file() for item in pasta.rglob("*"))


def confirmar_recriacao_processed():
    """Pergunta antes de apagar data/processed quando ela ja tem arquivos."""
    if not pasta_tem_arquivos(PROCESSED_DIR):
        return

    resposta = input(
        f"A pasta '{PROCESSED_DIR}' ja existe e contem arquivos. "
        "Deseja apagar e recriar? [s/N]: "
    ).strip().lower()

    if resposta not in {"s", "sim", "y", "yes"}:
        raise SystemExit("Operacao cancelada. Nenhum arquivo foi apagado.")

    shutil.rmtree(PROCESSED_DIR)


def verificar_entrada():
    """Confere se o dataset limpo existe e contem as classes esperadas."""
    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de imagens limpas nao encontrada: {CLEAN_DIR}\n"
            "Execute primeiro o script src/limpar_dataset.py."
        )

    for classe in CLASSES:
        pasta_classe = CLEAN_DIR / classe
        if not pasta_classe.exists():
            raise FileNotFoundError(f"Pasta da classe nao encontrada: {pasta_classe}")


def criar_estrutura_saida():
    """Cria data/processed/<split>/<classe> e a pasta de relatorios."""
    confirmar_recriacao_processed()

    for split in RATIOS:
        for classe in CLASSES:
            (PROCESSED_DIR / split / classe).mkdir(parents=True, exist_ok=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calcular_phash(caminho_imagem):
    """Calcula perceptual hash da imagem usando imagehash.phash."""
    with Image.open(caminho_imagem) as imagem:
        return imagehash.phash(imagem.convert("RGB"))


def calcular_hashes_classe(imagens):
    """Calcula pHash para todas as imagens validas de uma classe."""
    registros = []
    imagens_com_erro = []

    for imagem in imagens:
        try:
            hash_visual = calcular_phash(imagem)
        except (UnidentifiedImageError, OSError) as erro:
            imagens_com_erro.append(
                {
                    "caminho": caminho_relativo_projeto(imagem),
                    "erro": str(erro),
                }
            )
            continue

        registros.append(
            {
                "caminho": imagem,
                "phash": hash_visual,
            }
        )

    return registros, imagens_com_erro


def encontrar_raiz(pais, indice):
    """Encontra a raiz de um componente no Union-Find."""
    while pais[indice] != indice:
        pais[indice] = pais[pais[indice]]
        indice = pais[indice]
    return indice


def unir_componentes(pais, tamanhos, indice_a, indice_b):
    """Une dois componentes no Union-Find."""
    raiz_a = encontrar_raiz(pais, indice_a)
    raiz_b = encontrar_raiz(pais, indice_b)

    if raiz_a == raiz_b:
        return

    if tamanhos[raiz_a] < tamanhos[raiz_b]:
        raiz_a, raiz_b = raiz_b, raiz_a

    pais[raiz_b] = raiz_a
    tamanhos[raiz_a] += tamanhos[raiz_b]


def criar_grupos_similares(registros):
    """Cria componentes conectados usando distancia pHash menor ou igual ao limiar."""
    total = len(registros)
    pais = list(range(total))
    tamanhos = [1] * total

    for indice_a in range(total):
        hash_a = registros[indice_a]["phash"]
        for indice_b in range(indice_a + 1, total):
            distancia = hash_a - registros[indice_b]["phash"]
            if distancia <= LIMIAR_DISTANCIA_HASH:
                unir_componentes(pais, tamanhos, indice_a, indice_b)

    grupos_por_raiz = {}
    for indice, registro in enumerate(registros):
        raiz = encontrar_raiz(pais, indice)
        grupos_por_raiz.setdefault(raiz, []).append(registro)

    grupos = list(grupos_por_raiz.values())
    grupos.sort(
        key=lambda grupo: (
            -len(grupo),
            str(grupo[0]["caminho"]).lower(),
        )
    )
    return grupos


def escolher_split_para_grupo(totais_split, metas_split, tamanho_grupo):
    """Escolhe o split que fica mais perto da meta ao receber o grupo."""
    melhor_split = None
    melhor_pontuacao = None

    for split in RATIOS:
        total_atual = totais_split[split]
        meta = metas_split[split]

        # Penaliza ultrapassar a meta, mas permite quando for necessario.
        excesso_antes = max(0, total_atual - meta)
        excesso_depois = max(0, total_atual + tamanho_grupo - meta)
        distancia_depois = abs((total_atual + tamanho_grupo) - meta)
        pontuacao = (excesso_depois - excesso_antes, distancia_depois, total_atual)

        if melhor_pontuacao is None or pontuacao < melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_split = split

    return melhor_split


def atribuir_grupos_a_splits(grupos, total_imagens):
    """Divide grupos inteiros em train, val e test tentando respeitar 70/15/15."""
    metas_split = {
        split: total_imagens * proporcao
        for split, proporcao in RATIOS.items()
    }
    totais_split = {split: 0 for split in RATIOS}
    atribuicoes = {}

    grupos_embaralhados = grupos.copy()
    random.shuffle(grupos_embaralhados)
    grupos_embaralhados.sort(key=len, reverse=True)

    for grupo in grupos_embaralhados:
        split = escolher_split_para_grupo(
            totais_split=totais_split,
            metas_split=metas_split,
            tamanho_grupo=len(grupo),
        )
        atribuicoes[id(grupo)] = split
        totais_split[split] += len(grupo)

    return atribuicoes


def destino_sem_sobrescrever(destino_classe, imagem):
    """Gera caminho de destino sem sobrescrever arquivos de mesmo nome."""
    destino = destino_classe / imagem.name
    if not destino.exists():
        return destino

    contador = 2
    while True:
        destino = destino_classe / f"{imagem.stem}_{contador}{imagem.suffix}"
        if not destino.exists():
            return destino
        contador += 1


def copiar_grupo(grupo, split, classe):
    """Copia todas as imagens de um grupo para o mesmo split."""
    destino_classe = PROCESSED_DIR / split / classe

    for registro in grupo:
        origem = registro["caminho"]
        destino = destino_sem_sobrescrever(destino_classe, origem)
        shutil.copy2(origem, destino)


def salvar_metadados_classes():
    """Salva classes.txt e class_to_idx.json para compatibilidade do pipeline."""
    class_to_idx = {classe: indice for indice, classe in enumerate(CLASSES)}

    CLASSES_TXT.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    CLASS_TO_IDX_JSON.write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def processar_classe(classe):
    """Cria grupos visuais de uma classe, atribui splits e copia imagens."""
    imagens = listar_imagens(CLEAN_DIR / classe)
    registros, imagens_com_erro = calcular_hashes_classe(imagens)
    grupos = criar_grupos_similares(registros)
    atribuicoes = atribuir_grupos_a_splits(grupos, len(registros))

    linhas_grupos = []
    totais_split = {split: 0 for split in RATIOS}

    for grupo_id, grupo in enumerate(grupos, start=1):
        split = atribuicoes[id(grupo)]
        totais_split[split] += len(grupo)
        copiar_grupo(grupo, split, classe)

        for registro in grupo:
            linhas_grupos.append(
                {
                    "classe": classe,
                    "grupo_id": grupo_id,
                    "caminho_imagem": caminho_relativo_projeto(registro["caminho"]),
                    "phash": str(registro["phash"]),
                    "split_atribuido": split,
                }
            )

    tamanhos_grupos = [len(grupo) for grupo in grupos]
    resumo_classe = {
        "total_imagens": len(registros),
        "total_grupos": len(grupos),
        "maior_grupo": max(tamanhos_grupos, default=0),
        "grupos_com_mais_de_uma_imagem": sum(
            1 for tamanho in tamanhos_grupos if tamanho > 1
        ),
        "imagens_com_erro": imagens_com_erro,
        "total_por_split": totais_split,
    }

    return linhas_grupos, resumo_classe


def salvar_relatorios(linhas_distribuicao, linhas_grupos, resumo):
    """Salva CSVs e JSONs gerados pelo split agrupado."""
    pd.DataFrame(linhas_distribuicao).to_csv(
        DISTRIBUICAO_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(linhas_grupos).to_csv(
        GRUPOS_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    RESUMO_JSON.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    salvar_metadados_classes()


def imprimir_resumo(linhas_distribuicao, resumo):
    """Imprime distribuicao por classe, estatisticas dos grupos e totais finais."""
    df_distribuicao = pd.DataFrame(linhas_distribuicao)

    print("\nDistribuicao dos splits por classe:")
    print(df_distribuicao.to_string(index=False))

    print("\nGrupos por classe:")
    for classe in CLASSES:
        resumo_classe = resumo["classes"][classe]
        print(
            f"{classe}: {resumo_classe['total_grupos']} grupos | "
            f"maior grupo: {resumo_classe['maior_grupo']} imagens | "
            "grupos com mais de uma imagem: "
            f"{resumo_classe['grupos_com_mais_de_uma_imagem']}"
        )

    totais = resumo["total_final_por_split"]
    print("\nTotal de imagens por split:")
    print(f"Treino: {totais['train']}")
    print(f"Validacao: {totais['val']}")
    print(f"Teste: {totais['test']}")

    print("\nRelatorios salvos:")
    print(DISTRIBUICAO_CSV)
    print(CLASSES_TXT)
    print(CLASS_TO_IDX_JSON)
    print(GRUPOS_CSV)
    print(RESUMO_JSON)


def main():
    """Cria splits por grupos visuais sem usar raw/original e sem treinar nada."""
    random.seed(SEED)
    verificar_entrada()
    criar_estrutura_saida()

    linhas_distribuicao = []
    linhas_grupos = []
    resumo_classes = {}
    totais_finais = {split: 0 for split in RATIOS}

    for classe in CLASSES:
        linhas_classe, resumo_classe = processar_classe(classe)
        linhas_grupos.extend(linhas_classe)
        resumo_classes[classe] = resumo_classe

        total_train = resumo_classe["total_por_split"]["train"]
        total_val = resumo_classe["total_por_split"]["val"]
        total_test = resumo_classe["total_por_split"]["test"]

        totais_finais["train"] += total_train
        totais_finais["val"] += total_val
        totais_finais["test"] += total_test

        linhas_distribuicao.append(
            {
                "classe": classe,
                "treino": total_train,
                "validacao": total_val,
                "teste": total_test,
                "total": resumo_classe["total_imagens"],
            }
        )

    resumo = {
        "total_imagens_por_classe": {
            classe: dados["total_imagens"]
            for classe, dados in resumo_classes.items()
        },
        "total_grupos_por_classe": {
            classe: dados["total_grupos"]
            for classe, dados in resumo_classes.items()
        },
        "maior_grupo_por_classe": {
            classe: dados["maior_grupo"]
            for classe, dados in resumo_classes.items()
        },
        "quantidade_grupos_com_mais_de_uma_imagem": {
            classe: dados["grupos_com_mais_de_uma_imagem"]
            for classe, dados in resumo_classes.items()
        },
        "total_grupos_com_mais_de_uma_imagem": sum(
            dados["grupos_com_mais_de_uma_imagem"]
            for dados in resumo_classes.values()
        ),
        "total_final_por_split": totais_finais,
        "classes": resumo_classes,
        "seed": SEED,
        "limiar_distancia_phash": LIMIAR_DISTANCIA_HASH,
        "ratios_desejados": RATIOS,
    }

    salvar_relatorios(linhas_distribuicao, linhas_grupos, resumo)
    imprimir_resumo(linhas_distribuicao, resumo)


if __name__ == "__main__":
    main()
