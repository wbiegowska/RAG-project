# Technical choices

This file explains the main choices we made and why. Parsing, retrieval, the Gemini client,
evaluation and monitoring were done by Weronika, text-to-SQL, the router, the prompts and the
chat interface by Ala.

## Project setup

- **uv** for the dependencies, with two extra groups: `parse` (docling, only needed to parse
  the PDF again) and `eval` (ragas), so running the chat doesn't need them. `make install`
  installs everything.
- **ruff** for linting and formatting, run by a pre-commit git hook so the code stays
  consistent between the two authors.
- **pydantic-settings**: every setting (model names, paths, retrieval parameters) is in
  `src/rag/settings.py` and can be overridden in `.env`. The API key is only in `.env`, which
  is not in git: copy `.env.example` to `.env` and paste your key.
* **Base / Impl Modular Architecture**: Core abstract base classes and dataclasses in `src/rag/base` (`Chunk`, `BaseRetriever`, `LLMClient`, `SQLResult`, and `BasePipeline`) were established up front to define strict interface contracts. This decoupled component implementation from system integration, allowing concurrent, parallel development across database, retrieval, and pipeline modules without blocking dependencies.
- The PDF and the SQLite database are not in git (too big). The parsed rulebook is, since
  parsing it again takes several minutes.

## Gemini client

- Google AI Studio, with `gemini-3.8-flash` (the stable flash model of the free tier when we
  did the project).
- Every LLM call goes through `get_llm()`, which always returns the same client. That way all
  the calls share one **throttle** (10 requests per minute by default), otherwise each part
  of the pipeline would count on its own and would go over the free tier limit.
- Retries on 429 and 5xx errors are done by the google-genai SDK. While testing we got a lot
  of `503 UNAVAILABLE` ("high demand"), so after 3 attempts the client switches to fallback
  models (`gemini-3.6-flash`, then `gemini-3.5-flash-lite`) and prints a warning. Other
  errors (bad request, wrong key) are raised directly, another model wouldnt fix them.
- The temperature is left at the default, google recommends it for the gemini 3 models.
- For the evaluation we will have to turn the fallback off or record which model answered,
  otherwise the scores could mix two models.

## Parsing

We use **docling** with the TableFormer table model in accurate mode and the heading
hierarchy option (it guesses the heading levels from the font styles). The rulebook has
several columns per page, tables and a lot of headings: a plain text extraction (pypdf,
pdftotext) mixes the columns and loses the structure. Docling is slower (a few minutes for
the 60 pages) but we only run it once and commit the result
(`data/processed/Core_rules_parsed.md`).

Two export options: a page break marker between pages, to keep the page numbers, and no
`<!-- image -->` placeholders (there were 148 of them, only noise).

## Cleaning

`make clean-md` writes `data/processed/Core_rules_clean.md` from the docling output.

- **Page 4 is dropped.** It's the "how to read this book" page: the list of chapters with
  their page ranges and short notes about hints and tips, summaries and page numbers. The
  problem is that the PDF also has the text of pages 5-9 behind that panel as a background,
  cut at the edges, and docling reads both layers and mixes them line by line. About half of
  what comes out is this hidden copy (pages 5-9 are parsed fine on their own) and the rest
  contains no rules. We cant separate the two reliably, so the page is dropped and we only
  lose the chapter list and the notes.
- Page numbers, diagram labels and page headers ("CORE RULES | FIGHT PHASE") are removed.
- Control characters and double bullets ("- ■") coming from the PDF fonts are removed, and
  bullet points stuck on one line are split again.
- **Heading levels.** Docling puts the numbered steps ("1. HIT ROLL") at the top level, which
  would give wrong section paths. The title becomes `#`, the other headings move up one
  level and numbered steps go under their section. In the terrain chapter every terrain type
  (WOODS, RUINS...) has the same sub-headings (MOVEMENT, VISIBILITY, BENEFIT OF COVER) at the
  same level as the type, without a fix 7 different chunks were all called
  "TERRAIN FEATURES > BENEFIT OF COVER". AIRCRAFT also ended up under EXAMPLE BATTLEFIELDS.
- Page breaks become `<!-- page: N -->` markers for the chunker.

Limitation: some of these rules (page 4, the terrain types, aircraft) only make sense for
this PDF, another document would need its own checks.

## Chunking

- One section per heading, so a chunk never mixes two rules. The median section is about
  500 characters (one rule, one stratagem, one weapon ability) so most of them stay whole.
- Sections longer than **1200 characters** (about 1 in 7) are split with langchain's
  `RecursiveCharacterTextSplitter`, which cuts on paragraphs, then lines, then sentences,
  with 150 characters of overlap.
- Why 1200: the embedding model only reads the first 512 tokens. 1200 characters is about
  300 tokens, which leaves room for the section path we add in front.
- Each chunk keeps its **section path** (ex `THE BATTLE ROUND > SHOOTING PHASE > MAKING
  ATTACKS > 1. HIT ROLL`) and its **page**. The path is embedded with the text, and both are
  given to the LLM so the answer can cite them.
- We didnt use langchain's `MarkdownHeaderTextSplitter` for the headings because it drops our
  page markers and the chunks would lose their page. Our own split is about 30 lines.
- Result: 233 chunks.

## Embeddings and vector store

- **BAAI/bge-small-en-v1.5**, a small SBERT model (33M parameters) that runs on the laptop
  and does well on the retrieval part of the MTEB leaderboard for its size. Running it
  locally means no API calls: Gemini embeddings would use the same free quota as the chat.
- BGE models are trained with an instruction in front of short queries ("Represent this
  sentence for searching relevant passages: "). Our questions are short and the chunks long
  (asymmetric search), so we add it to the questions only.
- **ChromaDB**, saved on disk (`chroma/`, not in git, built with `make index`), cosine
  distance, HNSW index. We give chroma our own embeddings instead of letting it use its
  default model, so the chunks and the questions are embedded by the same model.

## Hybrid search and reranking

- **BM25** (rank-bm25) on the same chunks, to catch the exact game terms. Words are
  lowercased, a short list of stopwords is removed and the snowball stemmer makes "scores",
  "scoring" and "score" match.
- **Reciprocal rank fusion** (k = 60) merges the vector and BM25 results. It only uses the
  ranks, since cosine similarity and BM25 scores are not on the same scale.
- **Cross-encoder reranker** on the 20 best candidates: implemented but **off by default**.

We compared the setups on 5 questions, 3 from the teacher and 2 of ours
(`make compare-retrieval`, outputs in `eval/results/`). ✓ means the right section is #1:

| Question | vector | hybrid | hybrid + MiniLM reranker |
|---|---|---|---|
| What is the rule for grenades? | ✓ | ✓ | ✓ |
| How can I use a stratagem? | ✓ | ✓ | ✓ |
| What are the different ways to score points? | ✗ army points #1 | ✓ CAPTURE AND CONTROL #1 | ✗ army points #1 |
| Can my unit shoot after falling back? | ✓ | ✓ | ✓ |
| What happens when a unit is below half strength? | ✓ | ✓ | ✓ |

BM25 fixed the hardest question without breaking the others. The reranker
(cross-encoder/ms-marco-MiniLM-L6-v2) put the army points back at #1, and a bigger one
(BAAI/bge-reranker-base) was worse: none of its top 3 for that question was about scoring.
Both are trained on web search data and read "score" and "points" with their usual meaning,
but in the rulebook "scores" is mostly about dice rolls and "points" about the cost of the
army. So the default is hybrid without reranker (`use_reranker` in the settings). 5
questions is not a lot, the RAGAS evaluation will compare the setups again. For this kind of
question, rewriting the query with the LLM ("Victory points") should help more than a
reranker. Also, there isn't much about scoring in this PDF: only 3 passages mention Victory
points.

## Text-to-SQL

We implemented a robust Text-to-SQL pipeline targeting `wahadb.sqlite` to query unit statistics, weapon profiles, stratagems, and abilities.

* **Schema Ingestion & Few-Shot Prompting**: The SQL generation prompt includes the database schema and explicit rules (e.g., distinguishing between unit keywords like Space Marines and actual unit names).
* **Execution Safety**:
  * The database connection is strictly opened in read-only mode (`?mode=ro`).
  * Only single-statement `SELECT` queries are permitted; dangerous operations (`DROP`, `DELETE`, `UPDATE`, multi-statements) are rejected prior to execution.
  * Queries are forced to include a `LIMIT` clause to prevent resource exhaustion.
* **Self-Correction & Fallbacks**: If SQL execution fails due to a syntax error or missing table/column, the exception is caught, and the error context is sent back to Gemini for a single retry attempt before returning a graceful failure in `SQLResult`.

## Router and conversation memory

To handle multi-turn interactions and route queries efficiently, we built an agentic orchestration layer:

* **Query Rewriting**: Multi-turn follow-up questions (e.g., *"And its save?"*) are rewritten using `rewrite.jinja` into clear, self-contained standalone questions using the prior conversation context before reaching the retriever or router.
* **Intent Routing**: A structured router classifies standalone queries into three intent routes:
  * `rules`: Vector/hybrid retrieval over rulebook text.
  * `units`: Text-to-SQL queries over unit statistics and database tables.
  * `both`: Dual execution combining both sources for queries requiring both rule definitions and specific unit characteristics.
* **Pipeline Dataclass (`PipelineResult`)**: Encapsulates state across execution steps, maintaining the original question, standalone rewritten query, active route, SQL results, retrieved chunks, and generated answer.

## Prompts

Prompt engineering was standardized using Jinja2 templates (`src/rag/impl/prompts/`):

* **Modular Template Structure**: Distinct templates for `rewrite.jinja`, `router.jinja`, `sql.jinja`, and `answer.jinja`.
* **Context Formatting & Grounding**:
  * Rulebook chunks are formatted with section headers and page metadata for exact citation.
  * SQL execution results include both the rendered SQL query string and tabular row outputs.
  * System instructions enforce strict factual grounding: the model is instructed to answer strictly using the provided context and respond with *"I don't know based on the provided context"* when information is missing.

## Interface

(Ala)

## Evaluation

(to do)

## Monitoring

(to do)
