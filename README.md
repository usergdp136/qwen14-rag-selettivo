---
license: apache-2.0
base_model: mlx-community/Qwen3-14B-8bit
library_name: mlx
tags: [rag, bm25, italian, mlx, qwen3, evaluation]
language: [it, en]
---
# Qwen3-14B + RAG selettivo (qwen14_rag_v1)

**Solo codice, nessun peso.** Il modello è [`mlx-community/Qwen3-14B-8bit`](https://huggingface.co/mlx-community/Qwen3-14B-8bit) (Apache-2.0), non-thinking, decoding greedy. Indici BM25: [`odelpa/qwen14-rag-selettivo-index`](https://huggingface.co/datasets/odelpa/qwen14-rag-selettivo-index) (CC BY-SA 4.0). Risultati grezzi: [`odelpa/qwen14-rag-v1-risultati`](https://huggingface.co/datasets/odelpa/qwen14-rag-v1-risultati).

## In sintesi / TL;DR
- **IT:** con il gate selettivo il sistema **non è mai peggiore del modello base; piccolo guadagno, non significativo, su ARC-it**.
- **EN:** with the selective gate the system is **never worse than the base model; small, non-significant gain on ARC-it**.

## Gate per benchmark / Per-benchmark gate
| Benchmark | RAG | Regola / Rule |
|---|---|---|
| ARC-it | **ON** (fonte `wiki_math`) | retrieval score (BM25 top normalizzato) ≥ 0.3 **e** confidenza no-RAG < 0.9 |
| GSM8K | OFF | dev +1/60, CI [0,5] troppo incerto; sul test il gated sarebbe stato −1.33 pt |
| IFEval | OFF | dev net 0 |

Retrieval: BM25 top-3, brani troncati a 700 caratteri, prompt "usali SOLO se utili".

## Metodo / Method
- **IT:** gate scelto **solo sul dev** (ARC-it validation 286, disgiunto dal test; GSM8K train 60; IFEval 60 id non nel test). Test eseguito **una volta sola** (file `TEST_RUN_ONCE`). Significatività calcolata offline sui file grezzi (McNemar esatto, bootstrap 10k).
- **EN:** gate chosen **on dev only** (disjoint splits); test run **once**; significance computed offline from raw outputs (exact McNemar, 10k bootstrap).

## Risultati test / Test results
| Set | n | base | RAG | fix/break | McNemar p | bootstrap 95% (pt) |
|---|---|---|---|---|---|---|
| ARC-it, gated (selettivo) | 1132 | 0.9046 | **0.9090** | 5 / 0 | 0.0625 (n.s.) | [+0.09, +0.88] |
| ARC-it, RAG always-on wiki_math | 1132 | 0.9046 | 0.8843 | 19 / 42 | 0.0044 (peggio / worse) | [−3.36, −0.71] |
| GSM8K, gated (→ OFF) | 150 | 0.9333 | 0.920 | 1 / 3 | 0.625 (n.s.) | [−4.0, +1.33] |
| IFEval (OFF) | 100 | 0.83 | 0.83 | – | – | – |

Configurazione finale selettiva: ARC-it 0.9090, GSM8K 0.9333 (=base), IFEval 0.83 (=base).
Nota: il p di McNemar (0.0625) non è significativo a 0.05; il CI bootstrap esclude lo zero ma va letto come indicativo (pochi casi discordanti).

## Come eseguire / How to run (Apple Silicon, MLX)
```bash
pip install mlx-lm datasets lm-eval huggingface_hub
huggingface-cli download mlx-community/Qwen3-14B-8bit
huggingface-cli download odelpa/qwen14-rag-selettivo-index --repo-type dataset --local-dir index
# adattare in qrag.py: T0, M (path del modello) e i path dei chunks (SRC) -> index/wiki_it_chunks.jsonl, index/wiki_math_chunks.jsonl
python qrag.py dev     # sceglie il gate -> gate.json
python qrag.py test    # una volta sola
python significance.py # McNemar + bootstrap (richiede anche baseline_raw.json del baseline Qwen3-14B)
```
`qrag.py` usa anche `baseline_raw.json` (run baseline separato) per le righe no-RAG di GSM8K/IFEval. I path assoluti nel codice sono quelli della macchina originale.

## Articolo scientifico
Del Papa G. (2026). *RAG selettivo per benchmark su Qwen3-14B: mai peggio del modello base, guadagno piccolo e non significativo su ARC-it*. Preprint (IT), Zenodo. DOI: [10.5281/zenodo.23674921](https://doi.org/10.5281/zenodo.23674921) (tutte le versioni; v1.1: [10.5281/zenodo.23674951](https://doi.org/10.5281/zenodo.23674951)) — licenza testo CC BY 4.0.
- PDF articolo: [paper/qwen14_rag_selettivo_preprint.pdf](paper/qwen14_rag_selettivo_preprint.pdf)
- Versione divulgativa: [paper/divulgativa.pdf](paper/divulgativa.pdf)
- Codice: [Hugging Face](https://huggingface.co/odelpa/qwen14-rag-selettivo) · [GitHub](https://github.com/usergdp136/qwen14-rag-selettivo)

## File
`qrag.py` (bench + RAG), `run_q.sh`, `gate.json` (gate dal dev), `config.json` (config selettiva), `significance.py`, `build_corpus.py` (wiki_math), `build_chunks.py` + `dl_wiki.py` (wiki_it).

## Licenza
Codice Apache-2.0. Modello: Apache-2.0 (Qwen). Indici: CC BY-SA 4.0 (Wikipedia).
