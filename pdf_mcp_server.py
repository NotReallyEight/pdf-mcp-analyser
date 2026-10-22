from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Tesseract usa OpenMP: con N processi in parallelo ognuno aprirebbe N thread
# (oversubscription). Va impostato PRIMA di lanciare i sottoprocessi.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import httpx
import numpy as np
import ollama
import pymupdf as fitz
import pytesseract
from PIL import Image
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

# ---------------------------------------------------------------------------
# Configurazione (tutto sovrascrivibile da variabili d'ambiente)
# ---------------------------------------------------------------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")

CHUNK_SIZE = 4  # pagine per chunk
STRIDE = 2  # passo della finestra mobile
PIECE_CHARS = 2000  # caratteri per pezzo di embedding
MAX_PIECES_PER_PAGE = 4  # massimo pezzi di embedding per pagina
EMBED_BATCH = 32  # pezzi per chiamata a ollama.embed
MAX_CHAPTER_PAGES = 90  # oltre questa lunghezza il capitolo non allinea l'intervallo

OCR_MIN_CHARS = 50  # sotto questa soglia la pagina viene considerata scansione
OCR_DPI = int(os.environ.get("OCR_DPI", "150"))
OCR_LANG = os.environ.get("OCR_LANG", "ita+eng")
OCR_WORKERS = int(os.environ.get("OCR_WORKERS", min(8, os.cpu_count() or 4)))
OCR_PAGE_TIMEOUT = 60  # secondi massimi per una singola pagina
OCR_DEBUG = os.environ.get("OCR_DEBUG", "0") == "1"

MAX_PDF_MB = int(os.environ.get("MAX_PDF_MB", "500"))
MAX_PAGES = int(os.environ.get("MAX_PAGES", "3000"))
MAX_SECTIONS = 60
MAX_SUMMARY_CHARS = 100_000
MAX_EXPLAIN = 3  # passaggi da spiegare con l'LLM
EXCERPT_CHARS = 900  # caratteri per pagina nell'estratto passato all'LLM
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
MAX_CONCURRENT_ANALYSES = int(os.environ.get("MAX_CONCURRENT_ANALYSES", "2"))

IGNORED_PREFIXES = tuple(
    p.strip().lower()
    for p in os.environ.get("IGNORED_SECTION_PREFIXES", "logistica").split(",")
    if p.strip()
)

# Cartelle da cui è consentito leggere i PDF (separate da os.pathsep).
# Esempio macOS: PDF_BASE_DIRS="~/Downloads:/Volumes/Eight-SSD"
BASE_DIRS = [
    os.path.realpath(os.path.expanduser(d))
    for d in os.environ.get("PDF_BASE_DIRS", "~").split(os.pathsep)
    if d.strip()
]

# Le cache vivono fuori dalle cartelle dei PDF (che possono essere read-only
# o sincronizzate) e sono legate al contenuto del file (dimensione + mtime).
CACHE_DIR = Path(
    os.environ.get("PDF_CACHE_DIR", "~/.cache/pdf-mcp-analyser")
).expanduser()

# Con il trasporto stdio stdout è riservato al protocollo MCP: log su stderr.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pdf_range_analyzer")


# ---------------------------------------------------------------------------
# Percorso del binario Tesseract
# ---------------------------------------------------------------------------
def _resolve_tesseract() -> str:
    """Il server può partire con un PATH più povero del terminale: se
    `tesseract` non si trova nel PATH, provo i percorsi tipici di Homebrew."""
    explicit = os.environ.get("TESSERACT_CMD")
    if explicit:
        return explicit
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
        if os.path.isfile(candidate):
            return candidate
    return "tesseract"


TESSERACT_CMD = _resolve_tesseract()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

mcp = MCPServer("PDF Range Analyzer")
_ollama = ollama.Client(timeout=OLLAMA_TIMEOUT)

_ANALYSIS_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_ANALYSES)
_locks_guard = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}


def _path_lock(path: str) -> threading.Lock:
    """Un lock per PDF: due richieste sullo stesso libro non rifanno l'OCR due volte."""
    with _locks_guard:
        return _path_locks.setdefault(path, threading.Lock())


# ---------------------------------------------------------------------------
# Errori
# ---------------------------------------------------------------------------
class AnalysisError(Exception):
    def __init__(self, code: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


def _error(code: str, message: str, hint: str | None = None) -> str:
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

    size_mb = os.path.getsize(path) / 1e6
    if size_mb > MAX_PDF_MB:
        raise AnalysisError(
            "PDF_TROPPO_GRANDE",
            f"Il PDF pesa {size_mb:.0f} MB (limite {MAX_PDF_MB} MB).",
            "Alza il limite con la variabile MAX_PDF_MB se necessario.",
        )

    return path


# ---------------------------------------------------------------------------
# Cache su disco
# ---------------------------------------------------------------------------
def _cache_file(pdf_path: str, kind: str) -> Path:
    """Il nome contiene un hash di percorso+dimensione+mtime: se il PDF cambia,
    la cache vecchia non viene più trovata (nessuna cache obsoleta)."""
    st = os.stat(pdf_path)
    digest = hashlib.sha256(
        f"{pdf_path}|{st.st_size}|{st.st_mtime_ns}".encode()
    ).hexdigest()[:20]
    return CACHE_DIR / f"{digest}.{kind}"


def _store_cache(target: Path, data: bytes) -> None:
    """Scrittura atomica; un errore di cache non deve far fallire l'analisi."""
    tmp = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
        tmp = None
    except OSError as e:
        logger.warning("Impossibile salvare la cache %s: %s", target.name, e)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_pages_cache(cache_file: Path, total: int) -> list[str] | None:
    if not cache_file.exists():
        return None
    try:
        pages = json.loads(cache_file.read_text(encoding="utf-8"))
        if (
            isinstance(pages, list)
            and len(pages) == total
            and all(isinstance(p, str) for p in pages)
        ):
            return pages
        logger.info("Cache testo non coerente: verrà rigenerata.")
    except (
        OSError,
        ValueError,
    ) as e:  # UnicodeDecodeError e JSONDecodeError sono ValueError
        logger.warning("Cache testo non valida (%s): verrà rigenerata.", e)
    return None


# ---------------------------------------------------------------------------
# Estrazione testo + OCR
# ---------------------------------------------------------------------------
_ocr_stats_lock = threading.Lock()
_ocr_stats = {"errors": 0, "last": None, "debug_saved": False}
MAX_LOGGED_OCR_ERRORS = 5


def _reset_ocr_stats() -> None:
    with _ocr_stats_lock:
        _ocr_stats["errors"] = 0
        _ocr_stats["last"] = None


def _record_ocr_error(message: str) -> None:
    with _ocr_stats_lock:
        _ocr_stats["errors"] += 1
        _ocr_stats["last"] = message
        n = _ocr_stats["errors"]
    if n <= MAX_LOGGED_OCR_ERRORS:
        logger.error("OCR errore #%d: %s", n, message)


def _maybe_save_debug_image(img: Image.Image) -> None:
    if not OCR_DEBUG:
        return
    with _ocr_stats_lock:
        if _ocr_stats["debug_saved"]:
            return
        _ocr_stats["debug_saved"] = True
    try:
        target = "/tmp/ocr_debug_page.png"
        img.save(target)
        logger.info("OCR_DEBUG: salvata la prima immagine in %s", target)
    except Exception as e:
        logger.warning("OCR_DEBUG: impossibile salvare l'immagine: %s", e)


def _ocr_image(img: Image.Image) -> str:
    """Chiama direttamente il binario tesseract: immagine su stdin, testo su
    stdout. stderr viene decodificato con errors="replace", quindi l'errore
    reale di Tesseract finisce nei log invece di un UnicodeDecodeError."""
    _maybe_save_debug_image(img)

    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)

    try:
        proc = subprocess.run(
            [TESSERACT_CMD, "stdin", "stdout", "-l", OCR_LANG],
            input=buf.getvalue(),
            capture_output=True,
            timeout=OCR_PAGE_TIMEOUT,
        )
    except FileNotFoundError:
        raise pytesseract.TesseractNotFoundError()
    except subprocess.TimeoutExpired:
        _record_ocr_error(f"timeout di {OCR_PAGE_TIMEOUT}s su una pagina")
        return ""
    except Exception as e:  # una pagina illeggibile non deve far fallire il libro
        _record_ocr_error(f"{type(e).__name__}: {e}")
        return ""

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        _record_ocr_error(f"tesseract exit {proc.returncode}: {err[:500]}")
        return ""

    return proc.stdout.decode("utf-8", errors="replace").replace("\ufffd", "").strip()


def _extract_texts(doc, total: int) -> list[str]:
    """PyMuPDF non è thread-safe: lettura e rendering avvengono nel solo thread
    chiamante (documento aperto una volta sola). Solo Tesseract, che gira in
    sottoprocessi, è parallelizzato. Le immagini in coda sono limitate per non
    riempire la RAM."""
    _reset_ocr_stats()
    texts = [""] * total
    pending: deque = deque()
    max_pending = OCR_WORKERS * 2
    ocr_done = 0

    def drain(limit: int):
        nonlocal ocr_done
        while len(pending) > limit:
            idx, fut, fallback = pending.popleft()
            texts[idx] = fut.result() or fallback
            ocr_done += 1
            if ocr_done % 50 == 0:
                logger.info("OCR: %d pagine completate", ocr_done)

    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as ex:
        for idx in range(total):
            page = doc[idx]
            text = page.get_text("text").strip()

            if len(text) >= OCR_MIN_CHARS:
                texts[idx] = text
                continue

            # Scala di grigi: ~3x meno dati e nessuna codifica/decodifica PNG.
            pix = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csGRAY, alpha=False)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            pending.append((idx, ex.submit(_ocr_image, img), text))
            drain(max_pending)

        drain(0)

    return texts


def load_pages(pdf_path: str) -> list[str]:
    """Restituisce il testo di ogni pagina (indice 0 = pagina PDF 1)."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise AnalysisError("PDF_ILLEGGIBILE", f"Impossibile aprire il PDF: {e}")

    with doc:
        if doc.needs_pass:
            raise AnalysisError(
                "PDF_PROTETTO", "Il PDF è protetto da password.", "Rimuovi la password."
            )

        total = len(doc)
        if total == 0:
            raise AnalysisError("PDF_VUOTO", "Il PDF non contiene pagine.")
        if total > MAX_PAGES:
            raise AnalysisError(
                "PDF_TROPPE_PAGINE",
                f"Il PDF ha {total} pagine (limite {MAX_PAGES}).",
                "Alza il limite con la variabile MAX_PAGES se necessario.",
            )

        cache_file = _cache_file(pdf_path, f"pages.{OCR_LANG}.{OCR_DPI}.json")
        cached = _load_pages_cache(cache_file, total)
        if cached is not None:
            logger.info("Cache testo valida: %s", cache_file.name)
            return cached

        logger.info("Estrazione testo/OCR su %d pagine...", total)
        t0 = time.time()
        try:
            texts = _extract_texts(doc, total)
        except pytesseract.TesseractNotFoundError:
            raise AnalysisError(
                "TESSERACT_MANCANTE",
                f"Tesseract OCR non è installato o non è raggiungibile ({TESSERACT_CMD}).",
                "Installa con `brew install tesseract tesseract-lang` oppure imposta TESSERACT_CMD.",
            )
        logger.info("Estrazione completata in %.1fs", time.time() - t0)

    total_chars = sum(len(t) for t in texts)
    empty = sum(1 for t in texts if len(t) < OCR_MIN_CHARS)
    logger.info(
        "Testo estratto: %d caratteri, %d/%d pagine sotto %d caratteri, errori OCR: %d",
        total_chars,
        empty,
        total,
        OCR_MIN_CHARS,
        _ocr_stats["errors"],
    )

    if total_chars < 200:
        last = _ocr_stats["last"]
        detail = (
            f" Errori OCR: {_ocr_stats['errors']}. Ultimo: {last}"
            if _ocr_stats["errors"]
            else " Nessun errore OCR registrato: Tesseract ha restituito testo vuoto."
        )
        raise AnalysisError(
            "PDF_SENZA_TESTO",
            "Dal PDF è stato estratto praticamente nessun testo." + detail,
            "Controlla i log del server (righe 'OCR errore') e prova OCR_DEBUG=1.",
        )

    _store_cache(cache_file, json.dumps(texts, ensure_ascii=False).encode("utf-8"))
    return texts


# ---------------------------------------------------------------------------
# Embedding: batch + cache su disco
# ---------------------------------------------------------------------------
def normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def embed_batch(texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = _ollama.embed(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        out.extend(resp["embeddings"])
    return np.array(out, dtype=np.float32)


def page_embeddings(pdf_path: str, texts: list[str]) -> np.ndarray:
    """Un vettore normalizzato per pagina, calcolato una volta e salvato in .npy."""
    model_tag = re.sub(r"[^\w.-]", "_", EMBED_MODEL)
    # Dipende dal testo, quindi anche dai parametri OCR che lo hanno prodotto.
    cache = _cache_file(
        pdf_path,
        f"emb.{model_tag}.{PIECE_CHARS}x{MAX_PIECES_PER_PAGE}.{OCR_LANG}.{OCR_DPI}.npy",
    )

    if cache.exists():
        try:
            arr = np.load(cache, allow_pickle=False)
            if arr.ndim == 2 and arr.shape[0] == len(texts):
                logger.info("Cache embedding valida: %s", cache.name)
                return arr
            logger.info("Cache embedding non coerente: verrà rigenerata.")
        except (OSError, ValueError) as e:
            logger.warning("Cache embedding non valida (%s): verrà rigenerata.", e)

    pieces: list[str] = []
    starts: list[int] = []  # indice del primo pezzo di ogni pagina

    for text in texts:
        text = text.strip() or "(pagina vuota)"
        limit = min(len(text), PIECE_CHARS * MAX_PIECES_PER_PAGE)
        starts.append(len(pieces))
        for k in range(0, limit, PIECE_CHARS):
            pieces.append("search_document: " + text[k : k + PIECE_CHARS])

    logger.info(
        "Embedding di %d pagine (%d pezzi) in batch da %d...",
        len(texts),
        len(pieces),
        EMBED_BATCH,
    )
    t0 = time.time()
    vecs = embed_batch(pieces)

    # Ogni pagina ha almeno un pezzo e i pezzi sono contigui: reduceat somma
    # i segmenti in un'unica operazione vettoriale (più veloce di np.add.at).
    arr = normalize(np.add.reduceat(vecs, starts, axis=0))

    buf = io.BytesIO()
    np.save(buf, arr)
    _store_cache(cache, buf.getvalue())

    logger.info("Embedding completati in %.1fs (salvati in cache)", time.time() - t0)
    return arr


# ---------------------------------------------------------------------------
# Chunk, punteggi, regione
# ---------------------------------------------------------------------------
def build_chunks(page_vecs: np.ndarray):
    """Finestre mobili sulle pagine. Restituisce confini [start, end) a base 0 e
    il vettore medio di ogni finestra (somme cumulative: O(n), nessuna copia di testo).
    """
    n = len(page_vecs)
    bounds = np.array(
        [(s, min(s + CHUNK_SIZE, n)) for s in range(0, n, STRIDE)], dtype=np.int64
    )
    csum = np.vstack(
        [np.zeros((1, page_vecs.shape[1]), np.float32), np.cumsum(page_vecs, axis=0)]
    )
    sizes = (bounds[:, 1] - bounds[:, 0])[:, None]
    means = (csum[bounds[:, 1]] - csum[bounds[:, 0]]) / sizes
    return bounds, normalize(means)


def split_summary(summary: str) -> list[str]:
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


def grow_region_pages(page_scores: np.ndarray, smooth: int = 5, keep: float = 0.25):
    """Individua una regione contigua sui punteggi di pagina smussati."""
    smooth = min(smooth, len(page_scores))
    if smooth % 2 == 0:  # la finestra deve essere dispari per mantenere la lunghezza
        smooth -= 1
    smooth = max(smooth, 1)

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


def pick_top_chunks(bounds, scores, lo, hi, k):
    """I k chunk migliori nella regione, senza sovrapposizioni (con stride 2 e
    finestra 4 i chunk adiacenti condividono metà pagine: i primi 3 sarebbero
    quasi lo stesso passaggio)."""
    cand = [i for i, (s, e) in enumerate(bounds) if e - 1 >= lo and s <= hi]
    cand.sort(key=lambda i: scores[i], reverse=True)

    chosen: list[int] = []
    for i in cand:
        s, e = bounds[i]
        if all(e <= bounds[j][0] or s >= bounds[j][1] for j in chosen):
            chosen.append(i)
            if len(chosen) == k:
                break
    return sorted(chosen)


# ---------------------------------------------------------------------------
# Capitoli reali: TOC del PDF, altrimenti regex "Capitolo N"
# ---------------------------------------------------------------------------
CHAPTER_RE = re.compile(
    r"^[ \t]*(?i:capitolo|chapter)[ \t]+(\d{1,2})[ \t]*"
    r"(?:[-–—.][ \t]*)?(?:[A-ZÀ-Ý][^:\n]{2,69})?$",
    re.M,
)
SCANNER_NAME_RE = re.compile(r"^[A-Za-z]{1,6}[_\-]?\d{5,}")


def _finalize_chapters(entries, n_pages):
    chapters = []
    for i, (title, start) in enumerate(entries):
        end = entries[i + 1][1] - 1 if i + 1 < len(entries) else n_pages
        chapters.append({"title": title, "start": start, "end": max(start, end)})
    return chapters


def _toc_is_usable(entries) -> bool:
    """Scarta indici generati dallo scanner anziché capitoli reali."""
    good = sum(
        1
        for title, _ in entries
        if re.search(r"[A-Za-zÀ-ÿ]{3,}", title) and not SCANNER_NAME_RE.match(title)
    )
    return bool(entries) and good / len(entries) >= 0.7


def detect_chapters(pdf_path: str, texts: list[str]):
    n = len(texts)

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
        logger.warning("Impossibile leggere il TOC del PDF: %s", e)

    # 2. Regex sul testo estratto.
    occurrences: dict[int, list] = {}

    for page_no, text in enumerate(texts, start=1):
        matches = list(CHAPTER_RE.finditer(text))

        if len({int(m.group(1)) for m in matches}) >= 3:
            continue  # Probabile pagina dell'indice.

        for m in matches:
            occurrences.setdefault(int(m.group(1)), []).append(
                (page_no, m.group(0).strip()[:80])
            )

    entries, last = [], 0
    for number in sorted(occurrences):
        nxt = next(((pg, t) for pg, t in occurrences[number] if pg > last), None)
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
def explain_chunk(section, texts, sec_page_scores, start, end) -> str:
    """Passa all'LLM le 2 pagine più pertinenti del chunk (non solo l'inizio
    del testo concatenato, che spesso taglia fuori proprio la parte rilevante)."""
    ranked = np.argsort(sec_page_scores[start:end])[::-1][:2] + start
    excerpt = "\n\n".join(
        f"[pagina PDF {p + 1}]\n{texts[p][:EXCERPT_CHARS]}" for p in sorted(ranked)
    )

    prompt = f"""Sei un assistente universitario. Devi spiegare perché un estratto di un libro è pertinente a un argomento di una lezione.

ARGOMENTO DELLA LEZIONE:
{section[:1200]}

ESTRATTO DEL LIBRO (pagine PDF {start + 1}-{end}, testo da OCR, può contenere errori):
{excerpt}

Rispondi in italiano con al massimo 2 frasi, citando concetti concreti presenti nell'estratto.
Se l'estratto non è realmente pertinente all'argomento, dillo esplicitamente."""

    resp = _ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False,  # qwen3 altrimenti "ragiona" per centinaia di token inutili
        options={"temperature": 0.1, "num_predict": 250},
    )
    text = resp["message"]["content"]
    # Fallback per versioni di Ollama che ignorano think=False.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _safe_explain(*args) -> str:
    """Se la spiegazione fallisce il risultato principale resta comunque valido."""
    try:
        return explain_chunk(*args)
    except Exception as e:
        logger.warning("Spiegazione LLM fallita: %s: %s", type(e).__name__, e)
        return f"(spiegazione non disponibile: {type(e).__name__})"


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def analyze(pdf_path: str, lesson_summary: str, spiega: bool) -> dict:
    t0 = time.time()
    path = validate_pdf_path(pdf_path)
    lesson_summary = lesson_summary or ""

    if len(lesson_summary) > MAX_SUMMARY_CHARS:
        raise AnalysisError(
            "RIASSUNTO_TROPPO_LUNGO",
            f"Il riassunto supera {MAX_SUMMARY_CHARS} caratteri.",
        )

    sections = split_summary(lesson_summary)
    if not sections:
        raise AnalysisError(
            "RIASSUNTO_NON_VALIDO",
            "Dal riassunto non è stata ricavata nessuna sezione didattica utile.",
            "Passa il testo completo del riassunto (almeno qualche frase per argomento).",
        )
    if len(sections) > MAX_SECTIONS:
        logger.warning(
            "Sezioni %d > %d: uso solo le prime.", len(sections), MAX_SECTIONS
        )
        sections = sections[:MAX_SECTIONS]

    # Solo la fase di cache (OCR + embedding) è serializzata per PDF.
    with _path_lock(path):
        texts = load_pages(path)
        page_vecs = page_embeddings(path, texts)

    bounds, chunk_vecs = build_chunks(page_vecs)

    query_vecs = normalize(
        embed_batch(["search_query: " + s[:PIECE_CHARS] for s in sections])
    )

    page_z = zscore_cols(page_vecs @ query_vecs.T)  # (pagine, sezioni)
    page_scores = page_z.max(axis=1)
    chunk_sims = zscore_cols(chunk_vecs @ query_vecs.T)
    chunk_scores = chunk_sims.max(axis=1)
    chunk_best_section = chunk_sims.argmax(axis=1)

    lo_p, hi_p = grow_region_pages(page_scores)
    est_min, est_max = lo_p + 1, hi_p + 1
    min_page, max_page = est_min, est_max

    chapters, method = detect_chapters(path, texts)
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
        top = pick_top_chunks(bounds, chunk_scores, lo_p, hi_p, MAX_EXPLAIN)

        def job(i: int) -> str:
            start, end = int(bounds[i][0]), int(bounds[i][1])
            sec = int(chunk_best_section[i])
            return _safe_explain(sections[sec], texts, page_z[:, sec], start, end)

        if top:
            # Utile se Ollama ha OLLAMA_NUM_PARALLEL > 1; altrimenti le richieste si accodano.
            with ThreadPoolExecutor(max_workers=len(top)) as ex:
                reasons = list(ex.map(job, top))

            for i, reason in zip(top, reasons):
                sec = int(chunk_best_section[i])
                details.append(
                    {
                        "pagine": f"{int(bounds[i][0]) + 1}-{int(bounds[i][1])}",
                        "z_score": round(float(chunk_scores[i]), 3),
                        "argomento_lezione": sections[sec].splitlines()[0][:120],
                        "motivazione": reason,
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


def _run_analysis(pdf_path: str, lesson_summary: str, spiega: bool) -> str:
    """Esegue l'analisi e traduce ogni eccezione in un JSON di errore."""
    try:
        with _ANALYSIS_SLOTS:
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

    except (httpx.TimeoutException, TimeoutError):
        return _error(
            "OLLAMA_TIMEOUT",
            f"Ollama non ha risposto entro {OLLAMA_TIMEOUT:.0f}s.",
            "Riprova (i modelli freddi impiegano di più) o alza OLLAMA_TIMEOUT.",
        )

    except (ConnectionError, httpx.ConnectError):
        return _error(
            "OLLAMA_NON_RAGGIUNGIBILE",
            "Impossibile connettersi a Ollama.",
            "Avvia Ollama (app o `ollama serve`) e riprova.",
        )

    except Exception as e:
        # Il traceback resta nei log locali: al client (magari via tunnel) non va esposto.
        logger.error(
            "Errore imprevisto: %s: %s\n%s", type(e).__name__, e, traceback.format_exc()
        )
        return _error(
            "ERRORE_INTERNO", f"{type(e).__name__}: {e}", "Controlla i log del server."
        )


@mcp.tool()
async def analyze_lesson_ranges(
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
    # Il lavoro è CPU/IO-bound e dura anche minuti (OCR): in un thread, per non
    # bloccare l'event loop e quindi il server MCP.
    return await asyncio.to_thread(_run_analysis, pdf_path, lesson_summary, spiega)


def _log_startup_diagnostics() -> None:
    logger.info("Python: %s", sys.executable)
    logger.info("Tesseract (usato): %s", TESSERACT_CMD)
    logger.info("which tesseract: %s", shutil.which("tesseract"))
    logger.info("TESSDATA_PREFIX: %s", os.environ.get("TESSDATA_PREFIX"))
    logger.info("PATH: %s", os.environ.get("PATH"))
    logger.info("OCR_LANG=%s OCR_DPI=%s OCR_WORKERS=%s", OCR_LANG, OCR_DPI, OCR_WORKERS)
    try:
        r = subprocess.run(
            [TESSERACT_CMD, "--list-langs"], capture_output=True, timeout=15
        )
        logger.info(
            "tesseract --list-langs rc=%s\nOUT: %s\nERR: %s",
            r.returncode,
            r.stdout.decode("utf-8", "replace").strip(),
            r.stderr.decode("utf-8", "replace").strip(),
        )
    except Exception as e:
        logger.error(
            "Impossibile eseguire tesseract --list-langs: %s: %s", type(e).__name__, e
        )
    logger.info("Cartelle consentite: %s", BASE_DIRS)


if __name__ == "__main__":
    _log_startup_diagnostics()

    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")

    if transport == "stdio":
        mcp.run()
    else:
        kwargs = {}
        tunnel_host = os.environ.get("MCP_ALLOWED_HOST")

        if tunnel_host:
            extra_origins = [
                o.strip()
                for o in os.environ.get("MCP_ALLOWED_ORIGINS", "").split(",")
                if o.strip()
            ]
            kwargs["transport_security"] = TransportSecuritySettings(
                allowed_hosts=[tunnel_host, "127.0.0.1:*", "localhost:*"],
                allowed_origins=[
                    "https://chatgpt.com",
                    "http://127.0.0.1:*",
                    "http://localhost:*",
                    *extra_origins,
                ],
            )

        mcp.run(
            transport="streamable-http",
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8000")),
            **kwargs,
        )
