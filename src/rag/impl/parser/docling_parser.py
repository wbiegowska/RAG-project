"""Converts the rulebook PDF to markdown with docling."""

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
    """Docling converter with accurate tables and heading levels."""
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    # needed for the heading levels, docling guesses them from the font styles
    pipeline_options.generate_parsed_pages = True
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
    pipeline_options.table_structure_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def parse_document(document_path: str | Path) -> str:
    """Returns the markdown of the PDF, with page break markers and no image placeholders."""
    result = build_converter().convert(document_path)
    return result.document.export_to_markdown(
        image_placeholder="",
        page_break_placeholder=PAGE_BREAK,
    )
