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

### Chat Interface & Technical Choices

* **Streamlit Framework**: We used Streamlit to quickly build a clean web UI in pure Python without needing a separate frontend setup.
* **Conversation History (`st.session_state`)**: Chat history is saved across turns using `st.session_state.messages`. This passes past turns to our query rewriter, so follow-up questions (like *"how do I use it?"*) resolve correctly.
* **Sources & Details Expander**: Each response includes a dropdown menu (`st.expander("🔍 View Sources & Details")`) to show how the system worked:
  * **Route**: Shows if the question went to vector search (`rules`) or Text-to-SQL (`units`).
  * **Rewritten Query**: Displays the full standalone question created by the rewriter.
  * **Retrieved Data**: Shows the raw rulebook text chunks (with page numbers and scores) or the generated SQL query and database results.

## Evaluation

We evaluate with RAGAS like in lab 3, only the rulebook part (no text-to-SQL testing, as
asked). Everything is run with `make testset` and `make eval`, the results are in
`eval/results/`.

### Test set

- Generated with ragas from **20 sections** of the cleaned rulebook, 1 to 3 per chapter. We
  didnt use the whole book because of the free tier: building the knowledge graph costs 3
  gemini calls per section (summary, themes, entities), plus the calls for the questions.
  The knowledge graph is saved after every step, so when we hit the quota the next run
  continued where it stopped instead of starting again.
- The sections are short enough to be exactly one of our chunks, so the reference passages
  of the questions are the text of our chunks (checked: 25 out of 25).
- Question types from the course: half single-hop specific, a quarter multi-hop specific and
  a quarter multi-hop abstract. We gave ragas two personas (a new player and an experienced
  player) instead of letting it generate them, it saves calls.
- Ragas generated 16 questions and we **checked all of them by hand**: 2 were removed (one
  was not really a question, the other was two unrelated questions glued together), 1 was
  reworded because the sentence didnt make sense, the typos were kept on purpose (ragas adds
  them to test how robust the system is). We added the **3 rulebook questions of the
  teacher** (grenades, how to use a stratagem, ways to score points) with reference answers
  written from the rulebook.
- Final test set: **17 questions** (8 single-hop, 3 multi-hop specific, 3 multi-hop
  abstract, 3 from the teacher), frozen in `eval/testset.json`. The ragas draft and the
  knowledge graph are kept next to it.

### Retrieval

For each search setup we take the top 5 chunks of every question and compare them with the
reference passages:

- **hit rate**: at least one reference passage is in the top 5
- **MRR**: 1 / rank of the first reference passage (1 if it's first, 0.5 if second...)
- **context recall and precision** from ragas, the non-LLM versions (string similarity), so
  no gemini calls and the results dont depend on a judge

We compare the texts and not chunk ids, as said in the course ragas chunks the documents its
own way, here it works because the reference passages are exactly our chunks.

| setup | hit rate | MRR | context recall | context precision |
|---|---|---|---|---|
| vector | 0.94 | 0.83 | 0.87 | 0.79 |
| **hybrid (default)** | 0.94 | 0.85 | 0.87 | 0.82 |
| hybrid + reranker | 0.94 | 0.85 | 0.81 | 0.86 |

- All setups find a right passage for 16 of the 17 questions.
- Hybrid is as good or better than vector on every metric, so BM25 is worth it.
- The reranker improves the precision but loses recall: on two multi-hop questions it
  pushed one of the two needed passages out of the top 5. Same conclusion as our first test
  on 5 questions, so it stays off by default.
- The only question missed by all setups ("When does the Charge phase occur?") has as
  reference the intro of the shooting phase, which only mentions in passing that the charge
  phase comes next. The search finds the charge phase section instead, which is not wrong.
  It's a good example of why generated questions have to be checked.
- For "ways to score points" the scoring rule is #4 with vector, #1 with hybrid and #2 with
  the reranker, but all setups find only 1 of the 3 passages about Victory points.
- One case where BM25 hurts: "What does a Transport do?" is #1 with vector but #5 with
  hybrid, the other transport sections repeat the word "transport" a lot.

### Answers

Until the full pipeline (router, SQL, memory) is ready, the answers come from a **stand-in
pipeline** (`src/rag/evaluation/baseline.py`): hybrid search, top 5 chunks, and one call to
`gemini-3.1-flash-lite` with a short prompt (answer only from the rules, say you dont know
otherwise, cite section and page). It will also be the baseline to compare the full pipeline
with.

- **Judge: `gemini-3.5-flash-lite`**, different from and newer than the model answering, as
  recommended in the course. On the free tier the flash models only allow 20 requests per
  day, which is not enough for a judge (about 140 calls for one run), so we used the two
  flash-lite models (about 500 requests per day each).
- No fallback models during the evaluation, so every answer and every score comes from the
  same model. When gemini answers 503 ("high demand") we wait 30 s and try again.
- Metrics: **faithfulness** (is everything in the answer supported by the retrieved chunks),
  **answer relevancy** (does it answer the question, with strictness 1 to save calls) and
  **factual correctness** (F1 between the claims of the answer and of the reference).

| questions | n | faithfulness | answer relevancy | factual correctness |
|---|---|---|---|---|
| all | 17 | 0.91 | 0.79 | 0.60 |
| single-hop | 8 | 0.99 | 0.65 | 0.55 |
| multi-hop specific | 3 | 0.50 | 0.93 | 0.56 |
| multi-hop abstract | 3 | 1.00 | 0.93 | 0.80 |
| teacher questions | 3 | 1.00 | 0.91 | 0.59 |

- The answers stick to the retrieved rules (faithfulness 0.91) and cite the sections and
  pages. When the search missed the passage (charge phase question) the model answered "I do
  not know" instead of making something up, which is what we want even if it scores 0 for
  relevancy and correctness.
- Some single scores are clearly noise from the judge: "Can I use the same stratagem twice
  in the shooting phase?" got a correct answer quoting the exact rule but factual
  correctness 0 and faithfulness 0.5, and a good answer about the movement phase got
  relevancy 0. A small judge model and strictness 1 make single scores unreliable, the
  averages are more meaningful.
- "Ways to score points" is the weakest teacher question (factual correctness 0.25) because
  the search only finds one of the three passages about Victory points.

### Limitations

- 17 questions is small (the project asked for about 10): one question changes an average
  by about 0.06, so the differences between the search setups are trends, not proof.
- Nothing about terrain, aircraft, reserves or datasheets in the generated questions, ragas
  chose the topics from the links it found between the 20 sections.
- The judge is a small model (see above) and every run costs a good part of the daily free
  quota, so we couldnt run it many times.
- ragas 0.4.3 doesnt import with the latest langchain-community (a module it uses was
  removed in 0.4.2), so `langchain-community<0.4.2` is pinned in the eval group.

## Monitoring

We implemented a lightweight, non-intrusive tracing and monitoring system in `src/rag/monitoring/logger.py` to record live execution metrics for every turn of the chat pipeline.

- **`TraceLogger` Context Manager**: Wrapped directly around `run_pipeline` in `src/rag/pipeline.py`. It tracks pipeline steps without modifying function signatures or return types, preserving full backward compatibility with tests and evaluation scripts.
- **Turn-Level Telemetry**: Every execution turn appends a structured JSON line to `eval/results/chat_traces.jsonl`. Captured fields include:
  - **Query Context**: The original user question and the rewritten standalone question.
  - **Routing & Execution State**: The intent route classified by the router (`rules`, `units`, or `both`).
  - **Source Attribution**: Retain IDs/metadata for all retrieved rulebook chunks and the exact generated SQL query string.
  - **Latencies & Timing**: Total execution duration as well as per-step timing breakdown (routing, vector retrieval, Text-to-SQL execution, and final LLM synthesis).
  - **Token Consumption**: Prompt tokens, completion tokens, and total Gemini API usage per turn.
  - **Error Auditability**: Caught exceptions and error messages across any step in the pipeline.
- **Zero Framework Overhead**: Implemented in native Python without relying on heavy external tracking platforms (like LangChain, LangSmith, or LangFuse), keeping the execution fast, local, and completely self-contained.
