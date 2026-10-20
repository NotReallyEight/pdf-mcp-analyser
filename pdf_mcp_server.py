import sys
import os
import io
import re
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pymupdf as fitz
import pytesseract
from PIL import Image
import ollama
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
import tempfile

pytesseract.pytesseract.DEFAULT_ENCODING = "latin-1"

mcp = MCPServer("PDF Range Analyzer")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"

CHUNK_SIZE = 4  # pagine per chunk
STRIDE = 2  # passo della finestra mobile
PIECE_CHARS = 2000  # caratteri per pezzo di embedding
MAX_PIECES_PER_PAGE = 4  # massimo pezzi di embedding per pagina
EMBED_BATCH = 32  # pezzi per chiamata a ollama.embed
IGNORED_PREFIXES = ("logistica",)
MAX_CHAPTER_PAGES = (
    90  # oltre questa lunghezza, il capitolo non viene usato per allineare l'intervallo
)

# Cartelle da cui è consentito leggere i PDF (separate da os.pathsep).
# Esempio macOS: PDF_BASE_DIRS="~/Downloads:/Volumes/Wang-SSD-2"
BASE_DIRS = [
    os.path.realpath(os.path.expanduser(d))
    for d in os.environ.get("PDF_BASE_DIRS", "~").split(os.pathsep)
    if d.strip()
]


def log(msg: str):
    # Con il trasporto stdio stdout è riservato al protocollo MCP.
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
        raise AnalysisError(
            "PERCORSO_VUOTO",
            "Il percorso del PDF è vuoto.",
            "Passa il percorso assoluto del file PDF sul Mac.",
        )

    path = os.path.realpath(os.path.expanduser(pdf_path.strip().strip("\"'")))

    if not path.lower().endswith(".pdf"):
        raise AnalysisError("NON_UN_PDF", f"Il file non ha estensione .pdf: {path}")

    if not any(path == base or path.startswith(base + os.sep) for base in BASE_DIRS):
        raise AnalysisError(
            "PERCORSO_NON_CONSENTITO",
            "Il percorso è fuori dalle cartelle autorizzate sul server.",
            "Avvia il server con PDF_BASE_DIRS impostata sulla cartella che contiene il libro.",
        )

    if not os.path.isfile(path):
        raise AnalysisError(
            "FILE_NON_TROVATO",
            f"File non trovato: {path}",
            "Controlla il percorso assoluto (attenzione a spazi e maiuscole).",
        )

    return path


# ---------------------------------------------------------------------------
# Cache del testo estratto e OCR
# ---------------------------------------------------------------------------
def _ocr_page(pdf_path: str, page_idx: int):
    """Elabora una singola pagina in un thread separato in modo thread-safe."""
    doc = fitz.open(pdf_path)

    try:
        page = doc[page_idx]
        text = page.get_text("text").strip()

        if len(text) < 50:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as temp_file:
                    temp_path = temp_file.name

                img.save(temp_path, format="PNG")
                text = pytesseract.image_to_string(temp_path, lang="ita+eng").strip()

            except Exception:
                log(f"Errore OCR alla pagina PDF {page_idx + 1}")
                raise

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        return {"page": page_idx + 1, "text": text}

    finally:
        doc.close()


def _read_valid_page_cache(cache_file: str):
    """Carica la cache solo se è JSON valido con la struttura attesa."""
    with open(cache_file, "r", encoding="utf-8") as f:
        cached_pages = json.load(f)

    if not isinstance(cached_pages, list):
        raise ValueError("La cache non contiene una lista di pagine.")

    for index, page in enumerate(cached_pages, start=1):
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("page"), int)
            or not isinstance(page.get("text"), str)
        ):
            raise ValueError(f"Struttura non valida nella cache alla voce {index}.")

    return cached_pages


def load_pages(pdf_path: str):
    log(f"Python in uso: {sys.executable}")
    log(
        f"Tesseract configurato in pytesseract: "
        f"{pytesseract.pytesseract.tesseract_cmd}"
    )
    cache_file = pdf_path + ".cache.json"

    if os.path.exists(cache_file):
        try:
            cached_pages = _read_valid_page_cache(cache_file)

            # Verifica anche che la cache corrisponda al numero di pagine del PDF.
            with fitz.open(pdf_path) as doc:
                total_pdf_pages = len(doc)

            if len(cached_pages) == total_pdf_pages:
                log(f"Cache testo valida: {cache_file}")
                return cached_pages

            log(
                f"Cache non coerente con il PDF "
                f"({len(cached_pages)} voci nella cache, {total_pdf_pages} pagine nel PDF). "
                "Verrà rigenerata."
            )

        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            log(f"Cache testo non valida ({type(e).__name__}: {e}). Verrà rigenerata.")
        except OSError as e:
            log(f"Impossibile leggere la cache ({e}). Verrà rigenerata.")

        # La cache è rigenerabile. Se non si riesce a rimuoverla, segnala l'errore.
        try:
            os.remove(cache_file)
        except OSError as e:
            raise AnalysisError(
                "CACHE_NON_LEGGIBILE",
                f"Impossibile rimuovere la cache non valida: {e}",
                "Controlla i permessi del file .cache.json.",
            )

    try:
        with fitz.open(pdf_path) as doc:
            total = len(doc)
    except Exception as e:
        raise AnalysisError("PDF_ILLEGGIBILE", f"Impossibile aprire il PDF: {e}")

    log(f"OCR su {total} pagine...")
    t0 = time.time()

    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            pages = list(ex.map(lambda i: _ocr_page(pdf_path, i), range(total)))
    except pytesseract.TesseractNotFoundError:
        raise AnalysisError(
            "TESSERACT_MANCANTE",
            "Tesseract OCR non è installato o non è nel PATH.",
            "Installa con `brew install tesseract tesseract-lang`.",
        )

    log(f"OCR completato in {time.time() - t0:.1f}s")

    if sum(len(p["text"]) for p in pages) < 200:
        raise AnalysisError(
            "PDF_SENZA_TESTO",
            "Dal PDF è stato estratto praticamente nessun testo.",
            "Il file potrebbe essere protetto oppure l'OCR potrebbe non aver riconosciuto le pagine.",
        )

    # Scrittura atomica: evita di lasciare una cache parziale se la scrittura viene interrotta.
    temp_cache = cache_file + ".tmp"
    try:
        with open(temp_cache, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False)
        os.replace(temp_cache, cache_file)
    finally:
        if os.path.exists(temp_cache):
            try:
                os.remove(temp_cache)
            except OSError:
                pass

    return pages


# ---------------------------------------------------------------------------
# Embedding: batch + cache su disco
# ---------------------------------------------------------------------------
def normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def embed_batch(texts):
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = ollama.embed(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        out.extend(resp["embeddings"])
    return np.array(out, dtype=np.float32)


def page_embeddings(pdf_path: str, pages):
    """Un vettore normalizzato per pagina, calcolato una volta e salvato in .npy."""
    cache = pdf_path + f".emb.{EMBED_MODEL.replace(':', '_')}.npy"

    if os.path.exists(cache):
        try:
            arr = np.load(cache, allow_pickle=False)
            if arr.ndim == 2 and arr.shape[0] == len(pages):
                return arr
            log(
                "Cache embedding non coerente con il numero di pagine: verrà rigenerata."
            )
        except (OSError, ValueError) as e:
            log(f"Cache embedding non valida ({e}): verrà rigenerata.")

    pieces, owner = [], []

    for idx, page in enumerate(pages):
        text = (page["text"] or "").strip() or "(pagina vuota)"
        limit = min(len(text), PIECE_CHARS * MAX_PIECES_PER_PAGE)

        for k in range(0, limit, PIECE_CHARS):
            pieces.append("search_document: " + text[k : k + PIECE_CHARS])
            owner.append(idx)

    if not pieces:
        raise AnalysisError(
            "PDF_SENZA_TESTO",
            "Non ci sono contenuti testuali da trasformare in embedding.",
        )

    log(
        f"Embedding di {len(pages)} pagine ({len(pieces)} pezzi) in batch da {EMBED_BATCH}..."
    )
    t0 = time.time()
    vecs = embed_batch(pieces)

    arr = np.zeros((len(pages), vecs.shape[1]), dtype=np.float32)
    np.add.at(arr, np.array(owner), vecs)
    arr = normalize(arr)

    temp_cache = cache + ".tmp"
    try:
        with open(temp_cache, "wb") as f:
            np.save(f, arr)
        os.replace(temp_cache, cache)
    finally:
        if os.path.exists(temp_cache):
            try:
                os.remove(temp_cache)
            except OSError:
                pass

    log(f"Embedding completati in {time.time() - t0:.1f}s (salvati in cache)")
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

        chunks.append(
            {
                "start_page": cp[0]["page"],
                "end_page": cp[-1]["page"],
                "text": text if text.strip() else "(vuoto)",
                "vector": page_vecs[sl].mean(axis=0),
            }
        )

    return chunks


def split_summary(summary: str):
    parts = [p.strip() for p in re.split(r"\n#{1,6}\s*", "\n" + summary) if p.strip()]

    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\n\s*\n", summary) if p.strip()]

    return [
        p for p in parts if not p.lower().startswith(IGNORED_PREFIXES) and len(p) >= 30
    ]


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
    """Individua una regione contigua sui punteggi di pagina smussati."""
    pad = smooth // 2
    sm = np.convolve(
        np.pad(page_scores, pad, mode="edge"), np.ones(smooth) / smooth, mode="valid"
    )

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
CHAPTER_RE = re.compile(
    r"^[ \t]*(?i:capitolo|chapter)[ \t]+(\d{1,2})[ \t]*"
    r"(?:[-–—.][ \t]*)?(?:[A-ZÀ-Ý][^:\n]{2,69})?$",
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
    """Scarta indici generati dallo scanner anziché capitoli reali."""
    good = sum(
        1
        for title, _ in entries
        if re.search(r"[A-Za-zÀ-ÿ]{3,}", title) and not SCANNER_NAME_RE.match(title)
    )
    return bool(entries) and good / len(entries) >= 0.7


def detect_chapters(pdf_path: str, pages):
    n = len(pages)

    # 1. Segnalibri del PDF (livello 1).
    try:
        with fitz.open(pdf_path) as doc:
            toc = doc.get_toc()

        entries = sorted(
            {
                (title.strip(), page)
                for level, title, page in toc
                if level == 1 and 1 <= page <= n
            },
            key=lambda entry: entry[1],
        )

        if len(entries) >= 3 and _toc_is_usable(entries):
            return _finalize_chapters(entries, n), "toc"
    except Exception as e:
        log(f"Impossibile leggere il TOC del PDF: {e}")

    # 2. Regex sul testo estratto.
    occurrences = {}

    for page in pages:
        matches = list(CHAPTER_RE.finditer(page["text"] or ""))

        if len({int(match.group(1)) for match in matches}) >= 3:
            continue  # Probabile pagina dell'indice.

        for match in matches:
            occurrences.setdefault(int(match.group(1)), []).append(
                (page["page"], match.group(0).strip()[:80])
            )

    entries, last = [], 0

    for number in sorted(occurrences):
        nxt = next(
            ((pg, title) for pg, title in occurrences[number] if pg > last), None
        )
        if nxt:
            entries.append((nxt[1], nxt[0]))
            last = nxt[0]

    if len(entries) >= 3:
        return _finalize_chapters(entries, n), "regex"

    return [], "nessuno"


def snap_to_chapter(chapters, page_scores, min_page, max_page):
    """Allinea l'intervallo ai confini del capitolo quando la sovrapposizione è sufficiente."""
    region_len = max_page - min_page + 1
    best = None

    for chapter in chapters:
        overlap = min(chapter["end"], max_page) - max(chapter["start"], min_page) + 1

        if overlap <= 0:
            continue

        mean_score = float(page_scores[chapter["start"] - 1 : chapter["end"]].mean())

        if best is None or mean_score > best[0]:
            best = (mean_score, chapter, overlap)

    if best is None:
        return None

    _, chapter, overlap = best
    plausible = (chapter["end"] - chapter["start"] + 1) <= MAX_CHAPTER_PAGES

    return {"chapter": chapter, "snapped": overlap / region_len >= 0.5 and plausible}


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
    response_text = resp["message"]["content"]

    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.S).strip()


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def analyze(pdf_path: str, lesson_summary: str, spiega: bool) -> dict:
    t0 = time.time()
    path = validate_pdf_path(pdf_path)

    sections = split_summary(lesson_summary or "")
    if not sections:
        raise AnalysisError(
            "RIASSUNTO_NON_VALIDO",
            "Dal riassunto non è stata ricavata nessuna sezione didattica utile.",
            "Passa il testo completo del riassunto (almeno qualche frase per argomento).",
        )

    pages = load_pages(path)
    page_vecs = page_embeddings(path, pages)
    chunks = build_chunks(pages, page_vecs)

    query_vecs = normalize(
        embed_batch(["search_query: " + section[:PIECE_CHARS] for section in sections])
    )

    page_scores = zscore_cols(page_vecs @ query_vecs.T).max(axis=1)
    chunk_vecs = normalize(np.array([chunk["vector"] for chunk in chunks]))
    chunk_sims = zscore_cols(chunk_vecs @ query_vecs.T)
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
            chapter = snap["chapter"]
            anchor["capitolo"] = chapter["title"]
            anchor["range_capitolo"] = f"{chapter['start']}-{chapter['end']}"
            anchor["allineato_al_capitolo"] = snap["snapped"]

            if snap["snapped"]:
                min_page, max_page = chapter["start"], chapter["end"]

    details = []

    if spiega:
        in_region = [
            index
            for index, chunk in enumerate(chunks)
            if chunk["end_page"] >= est_min and chunk["start_page"] <= est_max
        ]
        top = sorted(in_region, key=lambda index: chunk_scores[index], reverse=True)[:3]

        for index in sorted(top):
            details.append(
                {
                    "pagine": f"{chunks[index]['start_page']}-{chunks[index]['end_page']}",
                    "z_score": round(float(chunk_scores[index]), 3),
                    "argomento_lezione": sections[
                        chunk_best_section[index]
                    ].splitlines()[0][:120],
                    "motivazione": explain_chunk(
                        sections[chunk_best_section[index]], chunks[index]
                    ),
                }
            )

    return {
        "status": "success",
        "range_pagine": f"Da pagina {min_page} a pagina {max_page}",
        "pagina_inizio": min_page,
        "pagina_fine": max_page,
        "ancoraggio_capitoli": anchor,
        "nota_pagine": (
            "Numeri di pagina del file PDF, non del libro stampato: se la scansione "
            "è a doppia pagina, ogni pagina PDF contiene due pagine del libro."
        ),
        "dettagli_sezioni": details,
        "tempo_secondi": round(time.time() - t0, 1),
    }


@mcp.tool()
def analyze_lesson_ranges(
    pdf_path: str, lesson_summary: str, spiega: bool = True
) -> str:
    """
    Trova in quali pagine di un libro PDF locale è trattato l'argomento di una lezione universitaria.

    Parametri:
    - pdf_path: percorso assoluto del PDF sul Mac.
    - lesson_summary: testo completo del riassunto o degli appunti della lezione.
    - spiega: se True, aggiunge una breve motivazione per i passaggi più rilevanti.

    Il risultato è un JSON con status "success" o "error". I numeri di pagina
    sono riferiti al PDF, non necessariamente alla numerazione stampata nel libro.
    """
    try:
        result = analyze(pdf_path, lesson_summary, spiega)
        return json.dumps(result, ensure_ascii=False, indent=2)

    except AnalysisError as e:
        return _error(e.code, e.message, e.hint)

    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404:
            return _error(
                "MODELLO_MANCANTE",
                f"Modello Ollama non trovato: {e.error}",
                f"Esegui `ollama pull {EMBED_MODEL}` e `ollama pull {LLM_MODEL}`.",
            )
        return _error("OLLAMA_ERRORE", f"Errore da Ollama: {e.error}")

    except ConnectionError:
        return _error(
            "OLLAMA_NON_RAGGIUNGIBILE",
            "Impossibile connettersi a Ollama.",
            "Avvia Ollama (app o `ollama serve`) e riprova.",
        )

    except Exception as e:
        tb = traceback.format_exc()

        log(f"Errore imprevisto: {type(e).__name__}: {e}")
        log(tb)

        return json.dumps(
            {
                "status": "error",
                "error_code": "ERRORE_INTERNO",
                "message": f"{type(e).__name__}: {e}",
                "traceback": tb,
            },
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")

    if transport == "stdio":
        mcp.run()
    else:
        kwargs = {}
        tunnel_host = os.environ.get("MCP_ALLOWED_HOST")

        if tunnel_host:
            kwargs["transport_security"] = TransportSecuritySettings(
                allowed_hosts=[tunnel_host, "127.0.0.1:*", "localhost:*"],
                allowed_origins=[
                    "https://chatgpt.com",
                    "http://127.0.0.1:*",
                    "http://localhost:*",
                ],
            )

        mcp.run(
            transport="streamable-http",
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8000")),
            **kwargs,
        )
