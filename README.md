# PDF MCP Analyser 📚⚡

Un server **Model Context Protocol (MCP)** leggero e attento alla privacy, scritto in Python. Consente ai modelli linguistici locali (LLM) di analizzare in modo intelligente manuali e libri di testo in PDF di centinaia di pagine, mappando i riassunti delle lezioni sull'intervallo esatto di pagine con motivazioni dettagliate.

Progettato per **MacBook Apple Silicon (serie M)** e ottimizzato per un'**elaborazione 100% locale e a costo zero** tramite **Ollama** e **Tesseract OCR**.

## Caratteristiche

* 🎯 **Mappatura precisa delle pagine**: Individua le sezioni del libro di testo rilevanti per i riassunti delle lezioni o per gli argomenti del programma.
* 🔒 **100% Privato e Locale**: Utilizza Ollama per embeddings e sintesi: non sono necessarie chiavi API esterne o crediti cloud.
* 🔍 **OCR Integrato**: Utilizza Tesseract OCR per leggere PDF scansionati e pagine di manuali senza testo selezionabile.
* 🧠 **Sintesi del contesto intelligente**: Genera brevi spiegazioni sul *perché* determinate sezioni sono rilevanti per i tuoi obiettivi di studio.
* 🔌 **Compatibilità MCP Universale**: Si connette senza problemi tramite trasporto `stdio` ai client MCP come **Cline (VS Code)**, **Cursor**, **Claude Desktop**, **Chatbox**, **ChatGPT Desktop**.

## 🛠 Tech Stack

* **Protocollo**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) (trasporto `stdio`)
* **Runtime**: Python 3.10+
* **Embeddings**: `nomic-embed-text` tramite Ollama
* **Ragionamento locale / Sintesi**: `qwen3:8b` (o `qwen2.5-coder:7b`) tramite Ollama
* **Motore OCR**: Tesseract (`pytesseract`)

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

> **Nota**: Sostituisci `/PERCORSO/ASSOLUTO/A/` e `NOME_UTENTE` con i percorsi reali presenti sul tuo Mac.

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
