from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat

opts = PdfPipelineOptions()
opts.document_timeout = 600
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

def parse_doc(src):
    try:
        result = converter.convert(src)

    except Exception as e:
        raise RuntimeError(f"Failed to parse document '{src}'") from e

    return result