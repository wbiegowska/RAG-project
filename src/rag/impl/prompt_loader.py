from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))


def render_prompt(template_name: str, **kwargs) -> str:
    template = _env.get_template(template_name)
    return template.render(**kwargs)
