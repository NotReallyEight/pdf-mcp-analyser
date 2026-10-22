# PDF MCP Analyser 📚⚡

Un server **Model Context Protocol (MCP)** leggero e attento alla privacy, scritto in Python. Consente ai modelli linguistici locali (LLM) di analizzare in modo intelligente manuali e libri di testo in PDF di centinaia di pagine, mappando i riassunti delle lezioni sull'intervallo esatto di pagine con motivazioni dettagliate.

Progettato per **MacBook Apple Silicon (serie M)** e ottimizzato per un'**elaborazione 100% locale e a costo zero** tramite **Ollama** e **Tesseract OCR**.

## Caratteristiche

* 🎯 **Mappatura precisa delle pagine**: Individua le sezioni del libro di testo rilevanti per i riassunti delle lezioni o per gli argomenti del programma.
* 🔒 **100% Privato e Locale**: Utilizza Ollama per embeddings e sintesi: non sono necessarie chiavi API esterne o crediti cloud.
* 🔍 **OCR Integrato**: Utilizza Tesseract OCR per leggere PDF scansionati e pagine di manuali senza testo selezionabile.
* 🧠 **Sintesi del contesto intelligente**: Genera brevi spiegazioni sul *perché* determinate sezioni sono rilevanti per i tuoi obiettivi di studio.
* 🔌 **Compatibilità MCP Universale**: Supporta i trasporti `stdio` e `streamable-http` per client MCP come **Cline (VS Code)**, **Cursor**, **Claude Desktop**, **Chatbox**, **ChatGPT Desktop**.

## 🛠 Tech Stack

* **Protocollo**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) (trasporto `streamable-http` predefinito, `stdio` disponibile)
* **Runtime**: Python 3.10+
* **Embeddings**: `nomic-embed-text` tramite Ollama
* **Ragionamento locale / Sintesi**: `qwen3:8b` (o `qwen2.5-coder:7b`) tramite Ollama
* **Motore OCR**: Tesseract (invocato tramite `pytesseract` e il binario di sistema)

## 📋 Prerequisiti

1. **Python 3.10+**
2. **Ollama**: Installato e in esecuzione sul tuo sistema ([ollama.com](https://ollama.com)).
3. **Modelli locali**:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen3:8b
   ```
4. **Tesseract OCR** (macOS):
   ```bash
   brew install tesseract tesseract-lang
   ```

   Se il server MCP viene avviato da un client con un `PATH` limitato, il codice
   cerca automaticamente i percorsi Homebrew standard
   (`/opt/homebrew/bin/tesseract` e `/usr/local/bin/tesseract`). In alternativa
   è possibile impostare `TESSERACT_CMD` con il percorso assoluto del binario.

## 🚀 Installazione

1. **Clona il repository**:
   ```bash
   git clone https://github.com/NotReallyEight/pdf-mcp-analyser.git
   cd pdf-mcp-analyser
   ```
2. **Crea e attiva un ambiente virtuale**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configurazione

### Aggiunta ai client MCP (ad es. Cline, Cursor, Claude Desktop)

Aggiungi la seguente configurazione al file di impostazioni MCP del tuo client (ad es. `cline_mcp_settings.json` o `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "pdf-mcp-analyser": {
      "command": "/PERCORSO/ASSOLUTO/A/pdf-mcp-analyzer/venv/bin/python3",
      "args": [
        "/PERCORSO/ASSOLUTO/A/pdf-mcp-analyzer/pdf_mcp_server.py"
      ],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "PDF_BASE_DIRS": "/Users/NOME_UTENTE"
      },
      "timeout": 300,
      "disabled": false,
      "autoApprove": [
        "analyze_lesson_ranges"
      ]
    }
  }
}
```

La configurazione precedente usa `stdio`, necessario per i client che avviano
il server tramite il campo `command`. Per avviare invece il server come endpoint
HTTP autonomo, ometti `MCP_TRANSPORT` (il valore predefinito è
`streamable-http`): il server ascolta su `127.0.0.1:8000`.

```json
"MCP_TRANSPORT": "streamable-http"
```

> **Nota**: Sostituisci `/PERCORSO/ASSOLUTO/A/` e `NOME_UTENTE` con i percorsi reali presenti sul tuo Mac.

### Variabili d'ambiente

Le impostazioni principali sono sovrascrivibili senza modificare il codice:

| Variabile | Predefinito | Descrizione |
| --- | --- | --- |
| `PDF_BASE_DIRS` | `~` | Cartelle autorizzate per la lettura dei PDF, separate da `:` su macOS. |
| `PDF_CACHE_DIR` | `~/.cache/pdf-mcp-analyser` | Cartella per cache OCR ed embeddings. |
| `MCP_TRANSPORT` | `streamable-http` | Impostare a `stdio` per usare stdin/stdout. |
| `MCP_HOST` | `127.0.0.1` | Host del trasporto HTTP. |
| `MCP_PORT` | `8000` | Porta del trasporto HTTP. |
| `MCP_ALLOWED_HOST` | non impostata | Abilita le impostazioni di sicurezza per l'accesso tramite tunnel/proxy. |
| `MCP_ALLOWED_ORIGINS` | vuota | Origini HTTP aggiuntive, separate da virgole, quando è impostato `MCP_ALLOWED_HOST`. |
| `EMBED_MODEL` | `nomic-embed-text` | Modello Ollama per gli embeddings. |
| `LLM_MODEL` | `qwen3:8b` | Modello Ollama per le spiegazioni. |
| `OLLAMA_TIMEOUT` | `180` | Timeout, in secondi, delle richieste a Ollama. |
| `OCR_LANG` | `ita+eng` | Lingue Tesseract da usare. |
| `OCR_DPI` | `150` | Risoluzione del rendering delle pagine per l'OCR. |
| `OCR_WORKERS` | fino a `8` | Numero massimo di pagine OCR elaborate in parallelo. |
| `OCR_DEBUG` | `0` | Impostare a `1` per salvare `/tmp/ocr_debug_page.png`. |
| `TESSERACT_CMD` | rilevato automaticamente | Percorso assoluto del binario Tesseract. |
| `MAX_PDF_MB` | `500` | Dimensione massima del PDF in MB. |
| `MAX_PAGES` | `3000` | Numero massimo di pagine analizzabili. |
| `IGNORED_SECTION_PREFIXES` | `logistica` | Prefissi delle sezioni da ignorare, separati da virgole. |

Per esempio:

```json
"env": {
  "MCP_TRANSPORT": "streamable-http",
  "PDF_BASE_DIRS": "/Users/NOME_UTENTE/Downloads:/Volumes/Eight-SSD",
  "OCR_LANG": "ita+eng",
  "PDF_CACHE_DIR": "/Users/NOME_UTENTE/.cache/pdf-mcp-analyser"
}
```

## 🧰 Strumenti MCP Disponibili

### `analyze_lesson_ranges`

Analizza un documento PDF locale confrontandolo con il riassunto di una lezione per individuare gli intervalli di pagine corrispondenti e le relative spiegazioni.

| Parametro        | Tipo      | Obbligatorio | Descrizione                                                                               | 
| ---------------- | --------- | ------------ | ----------------------------------------------------------------------------------------- | 
| `pdf_path`.      | `string`  | **Sì**       | Percorso assoluto del file PDF sul tuo sistema.                                           | 
| `lesson_summary` | `string`  | **Sì**       | Il testo grezzo o gli argomenti della lezione da analizzare.                              | 
| `spiega`         | `boolean` | No           | Imposta su `true` (predefinito) per generare le spiegazioni degli intervalli selezionati. | 

## 💬 Esempio di Utilizzo

Incolla nel prompt di chat del tuo client MCP:

```text
SYSTEM INSTRUCTION: Ignora il workspace e i file di sistema. Usa direttamente il tool MCP pdf-mcp-analyser.

Esegui la funzione analyze_lesson_ranges:
PDF: "/Users/nomeutente/Downloads/Sistemi-Operativi.pdf"
Lesson summary: "Lezione sulla schedulazione della CPU: criteri, algoritmi FCFS, SJF, Round Robin e valutazione."
```

## 📜 Licenza

Questo progetto è distribuito sotto Licenza MIT - vedi il file [LICENSE](LICENSE) per ulteriori dettagli.
