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

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"

def _ocr_page(pdf_path: str, page_idx: int):
    """Elabora una singola pagina in un thread separato in modo thread-safe."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    text = page.get_text("text").strip()

    if len(text) < 50:
        # DPI 150 per massima velocità mantenendo un'ottima precisione OCR
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="ita+eng").strip()

    doc.close()
    return {"page": page_idx + 1, "text": text}

# FIX 3: chunk più piccoli (4 pagine) con sovrapposizione (stride 2),
# così i bordi non dipendono da una griglia fissa da 10 pagine
def extract_and_chunk_pdf(pdf_path: str, chunk_size: int = 4, stride: int = 2, max_pages: int = None):
    if not os.path.exists(pdf_path):
        print(f"Errore: File '{pdf_path}' non trovato.")
        return []

    # Gestione file di cache pulito
    cache_suffix = f"_p{max_pages}.json" if max_pages else ".json"
    cache_file = pdf_path + f".cache{cache_suffix}"

    if os.path.exists(cache_file):
        print("⚡ Trovata cache locale! Caricamento testo istantaneo...")
        with open(cache_file, "r", encoding="utf-8") as f:
            pages_data = json.load(f)
    else:
        doc = fitz.open(pdf_path)
        total_pages = min(len(doc), max_pages) if max_pages else len(doc)
        doc.close()

        print(f"🔄 Estrazione OCR Parallela su {total_pages} pagine con 8 thread...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=8) as executor:
            args = [(pdf_path, idx) for idx in range(total_pages)]
            pages_data = list(executor.map(lambda p: _ocr_page(*p), args))

        print(f"✅ OCR completato in {time.time() - start_time:.2f} secondi!")

        # Salva la cache per le chiamate successive
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False)

    # Creazione dei blocchi di pagine (sovrapposti)
    chunks = []
    for i in range(0, len(pages_data), stride):
        chunk_pages = pages_data[i:i + chunk_size]
        start_p = chunk_pages[0]["page"]
        end_p = chunk_pages[-1]["page"]
        combined_text = "\n\n".join(p["text"] for p in chunk_pages if p["text"])

        # i chunk vuoti restano nella lista (testo placeholder) per mantenere l'ordine delle pagine
        chunks.append({
            "start_page": start_p,
            "end_page": end_p,
            "text": combined_text if combined_text.strip() else "(vuoto)"
        })

    return chunks

# FIX 1: non tagliare più a 2000 caratteri (vedeva meno di una pagina) + prefissi di nomic-embed-text
def get_embedding(text: str, prefix: str = "") -> np.ndarray:
    # nomic-embed-text in Ollama ha un contesto effettivo di ~2048 token (e l'OCR tokenizza male):
    # il testo lungo viene diviso in pezzi da 2000 caratteri, ognuno embeddato, poi si fa la media
    pieces = [text[i:i + 2000] for i in range(0, min(len(text), 16000), 2000)] or [text]
    response = ollama.embed(model=EMBED_MODEL, input=[prefix + p for p in pieces])
    return np.array(response["embeddings"]).mean(axis=0)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# FIX 4: embedding per sezione del riassunto, scartando la logistica
def split_summary(summary: str):
    parts = [p.strip() for p in re.split(r"\n#{1,6}\s*", "\n" + summary) if p.strip()]
    return [p for p in parts if not p.lower().startswith("logistica")]

def test_semantic_match(pdf_path: str, lesson_summary: str, max_pages: int = None):
    chunks = extract_and_chunk_pdf(pdf_path, chunk_size=4, stride=2, max_pages=max_pages)
    if not chunks:
        return

    print("\n🧠 Calcolo similarità vettoriale con Ollama...")
    start_time = time.time()

    sections = split_summary(lesson_summary)
    section_vectors = [get_embedding(s, "search_query: ") for s in sections]

    # matrice similarità: righe = chunk, colonne = sezioni del riassunto
    sims = np.zeros((len(chunks), len(sections)))
    for i, chunk in enumerate(chunks):
        chunk_vector = get_embedding(chunk["text"], "search_document: ")
        for j, sv in enumerate(section_vectors):
            sims[i, j] = cosine_similarity(sv, chunk_vector)

    # z-score per sezione (i valori grezzi sono tutti simili), poi massimo sulle sezioni
    sims = (sims - sims.mean(axis=0)) / np.where(sims.std(axis=0) < 1e-9, 1.0, sims.std(axis=0))
    scores = sims.max(axis=1)

    # FIX 2: regione contigua attorno al chunk migliore, non min/max di chunk sparsi
    best = int(np.argmax(scores))
    thr = scores.mean() + 0.5 * (scores[best] - scores.mean())
    lo = hi = best
    while lo > 0 and scores[lo - 1] >= thr:
        lo -= 1
    while hi < len(chunks) - 1 and scores[hi + 1] >= thr:
        hi += 1

    print(f"✅ Matching completato in {time.time() - start_time:.2f} secondi!")

    min_page = chunks[lo]["start_page"]
    max_page = chunks[hi]["end_page"]

    print("\n" + "=" * 60)
    print(f"📌 RISULTATO COMPLESSIVO: DA PAGINA {min_page} A PAGINA {max_page}")
    print("=" * 60)

    # i 3 chunk migliori all'interno della regione trovata
    region = sorted(range(lo, hi + 1), key=lambda k: scores[k], reverse=True)[:3]
    top_candidates = [(scores[k], chunks[k]) for k in region]

    for rank, (score, chunk) in enumerate(top_candidates, start=1):
        print(f"\n[Sezione #{rank}] Score (z): {score:.4f} | Pagine {chunk['start_page']}-{chunk['end_page']}")

        prompt = f"""
Sei un assistente universitario. Confronta questo riassunto di una lezione con un estratto di un libro.

RIASSUNTO LEZIONE:
"{lesson_summary}"

ESTRATTO LIBRO (Pagine {chunk['start_page']}-{chunk['end_page']}):
"{chunk['text'][:1500]}"

Spiega in 2 brevi frasi perché queste pagine sono rilevanti per la lezione.
"""
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response["message"]["content"].strip()
        print(f"Analisi {LLM_MODEL}:\n{explanation}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 test_matching.py /percorso/del/pdf [max_pagine]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    # Se specifichi un secondo argomento numerico imposta il limite pagine, altrimenti analizza TUTTO il libro
    pages_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    sample_summary = """Lezione sulla schedulazione della CPU: criteri, algoritmi (FCFS, SJF/SRTF, priorità, round robin, code multiple) e valutazione. Esercizi su diagramma di Gantt e tempo di attesa medio molto probabili alla prova intercorso.

### Concetti fondamentali e criteri

- Con un'unica CPU e multiprogrammazione, lo scheduler a breve termine sceglie dalla coda dei pronti a chi dare la CPU
- Burst di CPU tipicamente tra 0 e 5-6 ms, curva esponenziale
- Schedulazione nei casi 1 e 4 (running→waiting, terminazione) è non preemptive; nei casi 2 e 3 è preemptive
- Il dispatcher esegue context switch e passaggio a modo utente; il suo tempo è la latenza di dispatch (overhead)
- Criteri: utilizzo CPU, throughput, tempo di completamento, attesa e risposta; ci si concentra sul minimo tempo di attesa

### FCFS e SJF

- FCFS con burst 24, 3, 3: attesa media 17; con ordine P2, P3, P1 scende a 3 (effetto convoglio)
- SJF non preemptive sull'esempio (arrivi 0, 2, 4, 5; burst 7, 4, 1, 4) dà attesa media 4; SRTF preemptive dà 3
- Errore classico: il tempo di attesa va calcolato dall'istante di arrivo, non da 0
- SJF è ottimale ma il prossimo burst va stimato con media esponenziale: tau_n+1 = alpha t_n + (1 - alpha) tau_n

### Priorità, round robin e code multiple

- Scheduling per priorità rischia la starvation; soluzione: aging. In Unix intero più basso significa priorità più alta (NICE)
- Round robin con quanto di circa 10 ms; q deve essere ben maggiore del costo del context switch
- Round robin ha completamento medio più alto di SJF ma tempo di risposta migliore
- Code multiple con retroazione: esempio q0 RR 8 ms, q1 RR 16 ms, q2 FCFS, con scelta tra code 70/20/10%

### Valutazione degli algoritmi

- Modelli deterministici e reti di code sono teorici e approssimati; simulazioni più precise ma onerose
- Unico metodo sicuro: implementare l'algoritmo, oggi semplice con macchine virtuali

### Logistica del corso

- Esame indicato il 28 (mercoledì); prossima lezione il martedì dopo la settimana prossima
- Esercizi tipo SJF con diagramma di Gantt e tempo di attesa medio molto probabili alla prova intercorso"""

    print(f"Analisi su file: {pdf_file}")
    if pages_limit:
        print(f"Modalità test: Prime {pages_limit} pagine")
    else:
        print("Modalità completa: INTERO LIBRO")
    print(f"Riassunto lezione: \"{sample_summary}\"\n")

    test_semantic_match(pdf_file, sample_summary, max_pages=pages_limit)
