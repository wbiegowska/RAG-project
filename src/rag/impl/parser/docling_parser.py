"""Convert the rulebook PDF to markdown with Docling."""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from rag.impl.parser.markers import PAGE_BREAK


def build_converter() -> DocumentConverter:
    """Docling converter for the rulebook: accurate tables and inferred heading levels."""
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    # Heading levels are inferred from font styles, which needs the parsed pages.
    pipeline_options.generate_parsed_pages = True
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
    pipeline_options.table_structure_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def parse_document(document_path: str | Path) -> str:
    """Parse a PDF into markdown, with page break markers and without image placeholders."""
    result = build_converter().convert(document_path)
    return result.document.export_to_markdown(
        image_placeholder="",
        page_break_placeholder=PAGE_BREAK,
    )
