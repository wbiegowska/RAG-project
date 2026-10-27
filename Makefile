.PHONY: install parse clean-md index search chat test lint format testset eval

install:  ## install everything (app, parsing, evaluation, dev tools) + git hooks
	uv sync --all-groups
	uv run pre-commit install

parse:  ## PDF -> data/processed/Core_rules_parsed.md (slow, result is committed)
	uv run --group parse python scripts/parse.py

clean-md:  ## clean the parsed rulebook -> data/processed/Core_rules_clean.md
	uv run python scripts/clean.py

index:  ## build the ChromaDB index from the cleaned rulebook
	uv run python scripts/index.py

search:  ## try the rulebook search, e.g. make search q="What is the rule for grenades?"
	uv run python scripts/search.py "$(q)"

chat:  ## start the chat interface
	uv run streamlit run scripts/chat.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

testset:  ## generate the RAGAS test set (many LLM calls)
	uv run --group eval python scripts/generate_testset.py

eval:  ## run the RAGAS evaluation
	uv run --group eval python scripts/evaluate.py
