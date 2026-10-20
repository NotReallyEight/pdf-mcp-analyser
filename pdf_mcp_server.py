import sys
import os
import io
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pymupdf as fitz
import pytesseract
from PIL import Image
import ollama
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

mcp = MCPServer("PDF Range Analyzer")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"

CHUNK_SIZE = 4            # pagine per chunk
STRIDE = 2                # passo della finestra mobile
PIECE_CHARS = 2000        # caratteri per pezzo di embedding (contesto effettivo di nomic ~2048 token)
MAX_PIECES_PER_PAGE = 4   # una pagina OCR dense non supera ~8000 caratteri utili
EMBED_BATCH = 32          # pezzi per chiamata a ollama.embed
IGNORED_PREFIXES = ("logistica",)
MAX_CHAPTER_PAGES = 90    # un "capitolo" più lungo (in pagine PDF) è quasi certamente un falso positivo

# Cartelle da cui è consentito leggere i PDF (separate da ":"). Default: home dell'utente.
# Per un disco esterno: PDF_BASE_DIRS="~/Downloads:/Volumes/Wang-SSD-2"
BASE_DIRS = [
    os.path.realpath(os.path.expanduser(d))
    for d in os.environ.get("PDF_BASE_DIRS", "~").split(os.pathsep) if d.strip()
]


def log(msg: str):
    # su stderr: con il trasporto stdio stdout è riservato al protocollo
    print(msg, file=sys.stderr, flush=True)


class AnalysisError(Exception):
    def __init__(self, code: str, message: str, hint: str = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


def _error(code: str, message: str, hint: str = None) -> str:
    payload = {"status": "error", "error_code": code, "message": message}
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Validazione input
# ---------------------------------------------------------------------------
def validate_pdf_path(pdf_path: str) -> str:
    if not pdf_path or not pdf_path.strip():
        raise AnalysisError("PERCORSO_VUOTO", "Il percorso del PDF è vuoto.",
                            "Passa il percorso assoluto del file PDF sul Mac.")
    path = os.path.realpath(os.path.expanduser(pdf_path.strip().strip("\"'")))
    if not path.lower().endswith(".pdf"):
        raise AnalysisError("NON_UN_PDF", f"Il file non ha estensione .pdf: {path}")
    if not any(path == b or path.startswith(b + os.sep) for b in BASE_DIRS):
        raise AnalysisError("PERCORSO_NON_CONSENTITO",
                            "Il percorso è fuori dalle cartelle autorizzate sul server.",
                            "Avvia il server con PDF_BASE_DIRS impostata sulla cartella che contiene il libro.")
    if not os.path.isfile(path):
        raise AnalysisError("FILE_NON_TROVATO", f"File non trovato: {path}",
                            "Controlla il percorso assoluto (attenzione a spazi e maiuscole).")
    return path


# ---------------------------------------------------------------------------
# Testo per pagina (nativo + OCR) con cache
# ---------------------------------------------------------------------------
def _ocr_page(pdf_path: str, page_idx: int):
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        text = page.get_text("text").strip()
        if len(text) < 50:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="ita+eng").strip()
    finally:
        doc.close()
    return {"page": page_idx + 1, "text": text}


def load_pages(pdf_path: str):
    cache_file = pdf_path + ".cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()
    except Exception as e:
        raise AnalysisError("PDF_ILLEGGIBILE", f"Impossibile aprire il PDF: {e}")

    log(f"OCR su {total} pagine...")
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            pages = list(ex.map(lambda i: _ocr_page(pdf_path, i), range(total)))
    except pytesseract.TesseractNotFoundError:
        raise AnalysisError("TESSERACT_MANCANTE", "Tesseract OCR non è installato o non è nel PATH.",
                            "Installa con `brew install tesseract tesseract-lang`.")
    log(f"OCR completato in {time.time() - t0:.1f}s")

    if sum(len(p["text"]) for p in pages) < 200:
        raise AnalysisError("PDF_SENZA_TESTO", "Dal PDF non è stato estratto praticamente nessun testo.",
                            "Il file potrebbe essere protetto o l'OCR non ha riconosciuto le pagine.")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)
    return pages


# ---------------------------------------------------------------------------
# Embedding: batch + cache su disco
# ---------------------------------------------------------------------------
def normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def embed_batch(texts):
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = ollama.embed(model=EMBED_MODEL, input=texts[i:i + EMBED_BATCH])
        out.extend(resp["embeddings"])
    return np.array(out, dtype=np.float32)


def page_embeddings(pdf_path: str, pages):
    """Un vettore (normalizzato) per ogni pagina, calcolato una sola volta e salvato su .npy."""
    cache = pdf_path + f".emb.{EMBED_MODEL.replace(':', '_')}.npy"
    if os.path.exists(cache):
        arr = np.load(cache)
        if arr.shape[0] == len(pages):
            return arr

    pieces, owner = [], []
    for idx, p in enumerate(pages):
        text = (p["text"] or "").strip() or "(pagina vuota)"
        for k in range(0, min(len(text), PIECE_CHARS * MAX_PIECES_PER_PAGE), PIECE_CHARS):
            pieces.append("search_document: " + text[k:k + PIECE_CHARS])
            owner.append(idx)

    log(f"Embedding di {len(pages)} pagine ({len(pieces)} pezzi) in batch da {EMBED_BATCH}...")
    t0 = time.time()
    vecs = embed_batch(pieces)
    arr = np.zeros((len(pages), vecs.shape[1]), dtype=np.float32)
    np.add.at(arr, np.array(owner), vecs)
    arr = normalize(arr)
    np.save(cache, arr)
    log(f"Embedding completato in {time.time() - t0:.1f}s (salvato in cache)")
    return arr


# ---------------------------------------------------------------------------
# Chunk, riassunto, punteggi
# ---------------------------------------------------------------------------
def build_chunks(pages, page_vecs):
    chunks = []
    for i in range(0, len(pages), STRIDE):
        sl = slice(i, i + CHUNK_SIZE)
        cp = pages[sl]
        text = "\n\n".join(p["text"] for p in cp if p["text"])
        chunks.append({
            "start_page": cp[0]["page"],
            "end_page": cp[-1]["page"],
            "text": text if text.strip() else "(vuoto)",
            "vector": page_vecs[sl].mean(axis=0),
        })
    return chunks


def split_summary(summary: str):
    parts = [p.strip() for p in re.split(r"\n#{1,6}\s*", "\n" + summary) if p.strip()]
    if len(parts) <= 1:  # riassunto senza titoli: spezza per paragrafi
        parts = [p.strip() for p in re.split(r"\n\s*\n", summary) if p.strip()]
    return [p for p in parts if not p.lower().startswith(IGNORED_PREFIXES) and len(p) >= 30]


def zscore_cols(sims: np.ndarray) -> np.ndarray:
    std = sims.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (sims - sims.mean(axis=0)) / std


def grow_region(scores: np.ndarray, keep: float = 0.5):
    best = int(np.argmax(scores))
    thr = scores.mean() + keep * (scores[best] - scores.mean())
    lo = hi = best
    while lo > 0 and scores[lo - 1] >= thr:
        lo -= 1
    while hi < len(scores) - 1 and scores[hi + 1] >= thr:
        hi += 1
    return lo, hi


def grow_region_pages(page_scores: np.ndarray, smooth: int = 5, keep: float = 0.25):
    """Regione contigua sui punteggi di PAGINA smussati (media mobile).
    Soglia = mediana del libro + keep * (picco - mediana): robusta a singole pagine "fredde" nel capitolo."""
    pad = smooth // 2
    sm = np.convolve(np.pad(page_scores, pad, mode="edge"), np.ones(smooth) / smooth, mode="valid")
    base = float(np.median(sm))
    peak = int(np.argmax(sm))
    thr = base + keep * (sm[peak] - base)
    lo = hi = peak
    while lo > 0 and sm[lo - 1] >= thr:
        lo -= 1
    while hi < len(sm) - 1 and sm[hi + 1] >= thr:
        hi += 1
    return lo, hi


# ---------------------------------------------------------------------------
# Capitoli reali: TOC del PDF, altrimenti regex "Capitolo N"
# ---------------------------------------------------------------------------
# Solo righe che sono intestazioni: "Capitolo 5", "Capitolo 5 - Titolo", "Capitolo 5 Titolo".
# Scarta i rimandi nel testo ("Capitolo 3: se un thread invoca...", "capitolo 3 descrive...").
CHAPTER_RE = re.compile(
    r"^[ \t]*(?i:capitolo|chapter)[ \t]+(\d{1,2})[ \t]*(?:[-–—.][ \t]*)?(?:[A-ZÀ-Ý][^:\n]{2,69})?$",
    re.M,
)


def _finalize_chapters(entries, n_pages):
    chapters = []
    for i, (title, start) in enumerate(entries):
        end = entries[i + 1][1] - 1 if i + 1 < len(entries) else n_pages
        chapters.append({"title": title, "start": start, "end": max(start, end)})
    return chapters


SCANNER_NAME_RE = re.compile(r"^[A-Za-z]{1,6}[_\-]?\d{5,}")


def _toc_is_usable(entries) -> bool:
    """Scarta TOC generati dallo scanner (es. 'SKM_75822092708350'): sono lotti di scansione, non capitoli."""
    good = sum(
        1 for title, _ in entries
        if re.search(r"[A-Za-zÀ-ÿ]{3,}", title) and not SCANNER_NAME_RE.match(title)
    )
    return good / len(entries) >= 0.7


def detect_chapters(pdf_path: str, pages):
    n = len(pages)

    # 1) segnalibri del PDF (livello 1)
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        doc.close()
        entries = sorted({(t.strip(), p) for lvl, t, p in toc if lvl == 1 and 1 <= p <= n}, key=lambda e: e[1])
        if len(entries) >= 3 and _toc_is_usable(entries):
            return _finalize_chapters(entries, n), "toc"
    except Exception:
        pass

    # 2) regex sul testo: per ogni numero di capitolo, prima pagina utile dopo l'inizio del capitolo precedente
    occ = {}
    for p in pages:
        matches = list(CHAPTER_RE.finditer(p["text"] or ""))
        if len({int(m.group(1)) for m in matches}) >= 3:
            continue  # pagina dell'indice: elenca molti capitoli insieme
        for m in matches:
            occ.setdefault(int(m.group(1)), []).append((p["page"], m.group(0).strip()[:80]))

    entries, last = [], 0
    for num in sorted(occ):
        nxt = next(((pg, title) for pg, title in occ[num] if pg > last), None)
        if nxt:
            entries.append((nxt[1], nxt[0]))
            last = nxt[0]
    if len(entries) >= 3:
        return _finalize_chapters(entries, n), "regex"

    return [], "nessuno"


def snap_to_chapter(chapters, page_scores, min_page, max_page):
    """Se la regione trovata cade quasi tutta in un capitolo, estende ai confini del capitolo."""
    region_len = max_page - min_page + 1
    best = None
    for ch in chapters:
        overlap = min(ch["end"], max_page) - max(ch["start"], min_page) + 1
        if overlap <= 0:
            continue
        mean_score = float(page_scores[ch["start"] - 1:ch["end"]].mean())
        if best is None or mean_score > best[0]:
            best = (mean_score, ch, overlap)
    if best is None:
        return None
    _, ch, overlap = best
    plausible = (ch["end"] - ch["start"] + 1) <= MAX_CHAPTER_PAGES
    return {"chapter": ch, "snapped": overlap / region_len >= 0.5 and plausible}


# ---------------------------------------------------------------------------
# Spiegazione LLM
# ---------------------------------------------------------------------------
def explain_chunk(section: str, chunk: dict) -> str:
    prompt = f"""Sei un assistente universitario. Devi spiegare perché un estratto di un libro è pertinente a un argomento di una lezione.

ARGOMENTO DELLA LEZIONE:
{section[:1200]}

ESTRATTO DEL LIBRO (pagine PDF {chunk['start_page']}-{chunk['end_page']}, testo da OCR, può contenere errori):
{chunk['text'][:1500]}

Rispondi in italiano con al massimo 2 frasi, citando concetti concreti presenti nell'estratto.
Se l'estratto non è realmente pertinente all'argomento, dillo esplicitamente."""
    resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    text = resp["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def analyze(pdf_path: str, lesson_summary: str, spiega: bool) -> dict:
    t0 = time.time()
    path = validate_pdf_path(pdf_path)

    sections = split_summary(lesson_summary or "")
    if not sections:
        raise AnalysisError("RIASSUNTO_NON_VALIDO",
                            "Dal riassunto non è stata ricavata nessuna sezione didattica utile.",
                            "Passa il testo completo del riassunto (almeno qualche frase per argomento).")

    pages = load_pages(path)
    page_vecs = page_embeddings(path, pages)
    chunks = build_chunks(pages, page_vecs)

    Q = normalize(embed_batch(["search_query: " + s[:PIECE_CHARS] for s in sections]))

    page_scores = zscore_cols(page_vecs @ Q.T).max(axis=1)
    chunk_vecs = normalize(np.array([c["vector"] for c in chunks]))
    chunk_sims = zscore_cols(chunk_vecs @ Q.T)
    chunk_scores = chunk_sims.max(axis=1)
    chunk_best_section = chunk_sims.argmax(axis=1)

    lo_p, hi_p = grow_region_pages(page_scores)
    est_min, est_max = pages[lo_p]["page"], pages[hi_p]["page"]
    min_page, max_page = est_min, est_max

    chapters, method = detect_chapters(path, pages)
    anchor = {"metodo": method, "range_stimato_clustering": f"{est_min}-{est_max}"}
    if chapters:
        snap = snap_to_chapter(chapters, page_scores, est_min, est_max)
        if snap:
            ch = snap["chapter"]
            anchor["capitolo"] = ch["title"]
            anchor["range_capitolo"] = f"{ch['start']}-{ch['end']}"
            anchor["allineato_al_capitolo"] = snap["snapped"]
            if snap["snapped"]:
                min_page, max_page = ch["start"], ch["end"]

    details = []
    if spiega:
        in_region = [k for k, c in enumerate(chunks) if c["end_page"] >= est_min and c["start_page"] <= est_max]
        top = sorted(in_region, key=lambda k: chunk_scores[k], reverse=True)[:3]
        for k in sorted(top):
            details.append({
                "pagine": f"{chunks[k]['start_page']}-{chunks[k]['end_page']}",
                "z_score": round(float(chunk_scores[k]), 3),
                "argomento_lezione": sections[chunk_best_section[k]].splitlines()[0][:120],
                "motivazione": explain_chunk(sections[chunk_best_section[k]], chunks[k]),
            })

    return {
        "status": "success",
        "range_pagine": f"Da pagina {min_page} a pagina {max_page}",
        "pagina_inizio": min_page,
        "pagina_fine": max_page,
        "ancoraggio_capitoli": anchor,
        "nota_pagine": "Numeri di pagina del file PDF, non del libro stampato: se la scansione è a doppia "
                       "pagina, ogni pagina PDF contiene due pagine del libro.",
        "dettagli_sezioni": details,
        "tempo_secondi": round(time.time() - t0, 1),
    }


@mcp.tool()
def analyze_lesson_ranges(pdf_path: str, lesson_summary: str, spiega: bool = True) -> str:
    """
    Trova in quali pagine di un libro PDF locale è trattato l'argomento di una lezione universitaria.

    Quando usarlo: l'utente fornisce il percorso di un libro in PDF (anche scansionato) e il riassunto
    o gli appunti di una lezione, e vuole sapere quali pagine studiare.

    Parametri:
    - pdf_path: percorso ASSOLUTO del PDF sul Mac dell'utente (es. /Users/nome/Downloads/libro.pdf).
    - lesson_summary: testo COMPLETO del riassunto della lezione, non una sintesi tua. Se ha titoli
      di sezione (###), mantienili: ogni sezione viene confrontata separatamente col libro.
    - spiega: se True (default) aggiunge una breve motivazione per i passaggi più rilevanti.
      Metti False se serve solo l'intervallo di pagine ed è più veloce.

    Risultato (JSON):
    - status "success": usa `range_pagine` come risposta principale. Se `ancoraggio_capitoli` indica
      un capitolo, riportalo. Riassumi `dettagli_sezioni` in modo naturale. Ricorda all'utente che i
      numeri sono pagine del PDF (vedi `nota_pagine`), che possono differire da quelle stampate.
    - status "error": spiega all'utente il problema con `message` e, se presente, `hint`.
      NON inventare pagine e NON ritentare con gli stessi parametri.

    Nota: la prima analisi di un libro scansionato può richiedere diversi minuti (OCR ed embedding);
    le successive sullo stesso file sono molto più veloci perché i risultati vengono salvati in cache.
    """
    try:
        result = analyze(pdf_path, lesson_summary, spiega)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except AnalysisError as e:
        return _error(e.code, e.message, e.hint)
    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404:
            return _error("MODELLO_MANCANTE", f"Modello Ollama non trovato: {e.error}",
                          f"Esegui `ollama pull {EMBED_MODEL}` e `ollama pull {LLM_MODEL}`.")
        return _error("OLLAMA_ERRORE", f"Errore da Ollama: {e.error}")
    except ConnectionError:
        return _error("OLLAMA_NON_RAGGIUNGIBILE", "Impossibile connettersi a Ollama.",
                      "Avvia Ollama (app o `ollama serve`) e riprova.")
    except Exception as e:
        log(f"Errore imprevisto: {type(e).__name__}: {e}")
        return _error("ERRORE_INTERNO", f"Errore imprevisto durante l'analisi: {type(e).__name__}: {e}")


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    if transport == "stdio":
        mcp.run()  # client locali che parlano stdio
    else:
        kwargs = {}
        tunnel_host = os.environ.get("MCP_ALLOWED_HOST")  # es. xxxx.trycloudflare.com
        if tunnel_host:
            kwargs["transport_security"] = TransportSecuritySettings(
                allowed_hosts=[tunnel_host, "127.0.0.1:*", "localhost:*"],
                allowed_origins=["https://chatgpt.com", "http://127.0.0.1:*", "http://localhost:*"],
            )
        mcp.run(
            transport="streamable-http",
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8000")),
            **kwargs,
        )
