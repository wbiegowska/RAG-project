.PHONY: install parse clean-md index search compare-retrieval chat test lint format testset eval

install:  ## install everything (app, parsing, evaluation, dev tools) + git hooks
	uv sync --all-groups
	uv run pre-commit install

parse:  ## PDF -> data/processed/Core_rules_parsed.md (slow, result is committed)
	uv run --group parse python scripts/parse.py

clean-md:  ## clean the parsed rulebook -> data/processed/Core_rules_clean.md
	uv run python scripts/clean.py

index:  ## build the ChromaDB index from the cleaned rulebook
	uv run python scripts/index.py

search:  ## try the rulebook search, e.g. make search q="What is the rule for grenades?" mode=vector
	uv run python scripts/search.py "$(q)" $(if $(mode),--mode $(mode))

compare-retrieval:  ## top 3 of a few test questions with vector / hybrid / full search
	uv run python scripts/compare_retrieval.py

chat:  ## start the chat interface
	uv run streamlit run scripts/chat.py

test:  ## unit tests, with the eval group so the ragas tests are not skipped
	uv run --group eval pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

testset:  ## generate the RAGAS test set draft (many LLM calls), options size=20 rebuild=1
	RAGAS_DO_NOT_TRACK=true uv run --group eval python scripts/generate_testset.py $(if $(size),--size $(size)) $(if $(rebuild),--rebuild-graph)

eval:  ## run the RAGAS evaluation, options part=retrieval|answers|summary pipeline=baseline
	RAGAS_DO_NOT_TRACK=true uv run --group eval python scripts/evaluate.py $(if $(part),--part $(part)) $(if $(pipeline),--pipeline $(pipeline))
