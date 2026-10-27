# Warhammer 40k RAG

A chatbot that answers questions on the Warhammer 40,000 Core Rules (PDF) and unit
datasheets (SQLite database), using Gemini through Google AI Studio.

## Authors

- **Weronika Biegowska**: Gemini client, rulebook parsing, chunking, retrieval, evaluation, monitoring
- **Ala Karamkesh**: text-to-SQL, router and conversation memory, prompts, chat interface

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install the project: `make install`
3. Put `Core Rules.pdf` and `wahadb.sqlite` in `data/` (not in git because of their size).
4. Copy `.env.example` to `.env` and paste your Google AI Studio API key.

## Commands

| Command | What it does |
|---|---|
| `make chat` | Start the chat |
| `make index` | Build the rulebook search index |
| `make test` | Run the unit tests |
| `make eval` | Run the evaluation |
| `make parse` | Re-parse the PDF (slow; the parsed rulebook is already in `data/processed/`) |
| `make lint` / `make format` | Check / fix code style |

## Documentation

- [Architecture](docs/architecture.md)
- [Technical choices and justifications](docs/technical_choices.md)
