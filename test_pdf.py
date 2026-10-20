import sys
import os
import pymupdf as fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def inspect_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Errore: Il file '{pdf_path}' non esiste.")
        return

    doc = fitz.open(pdf_path)
    print("=" * 60)
    print(f"ANALISI PDF: {os.path.basename(pdf_path)}")
    print(f"Totale Pagine: {len(doc)}")
    print("=" * 60 + "\n")

    # 1. Estrazione Indice (TOC)
    toc = doc.get_toc()
    print("--- 1. ESTRAZIONE INDICE (TOC) ---")
    if toc:
        print(f"Trovate {len(toc)} voci nell'indice:")
        for item in toc[:15]:  # Mostra le prime 15 voci
            level, title, page = item[0], item[1], item[2]
            indent = "  " * (level - 1)
            print(f"{indent}- {title} (Pagina {page})")
        if len(toc) > 15:
            print(f"  ... e altre {len(toc) - 15} voci.")
    else:
        print("Nessun indice TOC nativo rilevato. Il sistema applicherà la suddivisione dinamica a blocchi.")

    print("\n" + "--- 2. TEST ESTRAZIONE E OCR SULLE PRIME 5 PAGINE ---")
    
    # 2. Test estrazione e OCR sulle prime pagine di esempio
    sample_limit = min(5, len(doc))
    for page_idx in range(sample_limit):
        page = doc[page_idx]
        native_text = page.get_text("text").strip()
        
        used_ocr = False
        final_text = native_text

        # Soglia: se il testo nativo è inferiore a 50 caratteri, attiva Tesseract
        if len(native_text) < 50:
            used_ocr = True
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            final_text = pytesseract.image_to_string(img, lang="ita+eng").strip()

        mode_str = "OCR (Tesseract)" if used_ocr else "Nativo (PyMuPDF)"
        preview = final_text[:120].replace('\n', ' ') if final_text else "[PAGINA VUOTA O NON LEGGIBILE]"
        
        print(f"Pagina {page_idx + 1}: Modalità [{mode_str}] | Caratteri estraibili: {len(final_text)}")
        print(f"  Anteprima: \"{preview}...\"\n")

    print("=" * 60)
    print("Test completato con successo.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 test_pdf.py /percorso/del/tuo/file.pdf")
    else:
        inspect_pdf(sys.argv[1])
