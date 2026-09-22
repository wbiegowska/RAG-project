from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    HeadingHierarchyOptions,
    TableStructureOptions,
    TableFormerMode
)

def parse_document_basic(file_path):
    """Parse a document and convert it to markdown format."""

    "Args: document_path (str): The path to the document file to be parsed."
    
    "Returns: str: The converted document in markdown format."
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.export_to_markdown()


def parse_document(document_path: str) -> str:
    """
    Parse a document and convert it to markdown format.
    
    Args:
        document_path (str): The path to the document file to be parsed.
        
    Returns:
        str: The converted document in markdown format.
    """
    # 1. Initialize PDF pipeline options
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    
    # 2. Configure heading hierarchy and tables
    pipeline_options.generate_parsed_pages = True
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
    pipeline_options.table_structure_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    
    # 3. Create the converter with the specified format options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # 4. Convert the document and return markdown
    result = converter.convert(document_path)
    return result.document.export_to_markdown()