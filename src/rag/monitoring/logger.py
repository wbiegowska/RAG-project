import json
import time
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path

LOG_FILE = Path("eval/results/chat_traces.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class TraceLogger:
    """Context manager and helper for tracking a chat turn execution across pipeline steps."""

    def __init__(self, question: str):
        self.question = question
        self.start_time = time.time()
        self.route: str | None = None
        self.retrieved_chunk_ids: list[str] = []
        self.generated_sql: str | None = None
        self.step_times: dict[str, float] = {}
        self.token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.errors: list[str] = []
        self.answer: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_time = round(time.time() - self.start_time, 4)
        if exc_val:
            self.errors.append(str(exc_val))

        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": self.question,
            "route": self.route,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "generated_sql": self.generated_sql,
            "total_duration_seconds": total_time,
            "step_times_seconds": self.step_times,
            "token_usage": self.token_usage,
            "errors": self.errors,
            "answer": self.answer,
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_entry) + "\n")

    def record_step(self, step_name: str, duration: float):
        """Record latency for an individual pipeline step."""
        self.step_times[step_name] = round(duration, 4)

    def add_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Accumulate token counts across LLM calls."""
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += prompt_tokens + completion_tokens


def trace_step(step_name: str):
    """Decorator to automatically time and log errors for individual pipeline steps."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            tracer = kwargs.get("tracer", None)
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if tracer:
                    tracer.errors.append(f"Error in {step_name}: {str(e)}")
                raise e
            finally:
                duration = time.time() - start
                if tracer and isinstance(tracer, TraceLogger):
                    tracer.record_step(step_name, duration)

        return wrapper

    return decorator
