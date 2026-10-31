# Warhammer 40k RAG

A chatbot that answers questions on the Warhammer 40,000 Core Rules (PDF) and unit
datasheets (SQLite database), using Gemini through Google AI Studio.

## Authors

- **Weronika Biegowska**: Gemini client, rulebook parsing, chunking, retrieval, evaluation, monitoring
- **Ala Karamkesh**: text-to-SQL, router and conversation memory, prompts, chat interface

## Setup & Quickstart

Follow these steps to set up and run the project locally:

### Step 1: Install `uv`
Install the `uv` package manager (if you haven't already):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Install Project Dependencies
Clone the repository, navigate into the project directory, and sync the virtual environment:
```bash
git clone https://github.com/wbiegowska/RAG-project.git
cd RAG-project
uv sync
```
*(Alternatively, if `make` is installed on your system, you can run `make install`)*

### Step 3: Add Required Data Files
Obtain and place the following two binary/large data files inside the `data/` directory (these are excluded from git repository control):
- `Core Rules.pdf`
- `wahadb.sqlite`

### Step 4: Configure Environment Variables
Create your local environment configuration file from the template:
```bash
cp .env.example .env
```
Open `.env` in your text editor and add your Google AI Studio API key:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### Step 5: Build Index & Start the Chat Interface
1. Generate the vector search index for the Core Rules PDF (run once prior to launching the chatbot):
   ```bash
   make index
   ```
2. Launch the Streamlit chat application:
   ```bash
   make chat
   ```
   *(Or launch directly using Streamlit: `uv run streamlit run scripts/chat.py`)*

---

## Commands Reference

| Command | Description |
|---|---|
| `make chat` | Start the Streamlit chat interface |
| `make index` | Build/update the rulebook vector search index |
| `make test` | Run the PyTest unit test suite |
| `make eval` | Run the evaluation pipeline |
| `make parse` | Re-parse the PDF (slow; parsed rulebook is already in `data/processed/`) |
| `make lint` / `make format` | Check or auto-fix code formatting with Ruff |

## Documentation

- [Architecture](docs/architecture.md)
- [Technical choices and justifications](docs/technical_choices.md)
