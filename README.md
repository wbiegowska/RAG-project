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

## Example Queries & Validation

The system successfully routes and resolves single-source and multi-source queries:

### 1. Rules / Mechanics (`rules` route)
* **Query:** *"What are the rules for throwing a grenade?"*
* **Result:** Successfully retrieves Chunk 0148 (Page 42) detailing the Shooting phase timing, 8" range restriction, visible target condition, and rolling six D6 for mortal wounds on 4+.

### 2. Unit Statistics & Costs (`units` route)
* **Query:** *"What is the Toughness stat of an Intercessor Squad?"*
* **Result:** Successfully queries `wahadb.sqlite` via Text-to-SQL and returns a Toughness of 4 for Intercessor and Assault Intercessor squads (and 6 for Heavy Intercessors).

### 3. Combined / Hybrid Query (`both` route)
* **Query:** *"Can an Intercessor Squad use the Grenade stratagem, and what does it do?"*
* **Result:** Combines database keyword checking (verifying the unit's eligibility) with vector search over the rulebook to explain the full Grenade stratagem mechanics.

## Documentation

- [Architecture](docs/architecture.md)
- [Technical choices and justifications](docs/technical_choices.md)
