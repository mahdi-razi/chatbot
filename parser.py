from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat


class Parser:
    def __init__(self, timeout=600):
        opts = PdfPipelineOptions()
        opts.document_timeout = timeout
        self.converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


    def parse_doc(self, src):
        try:
            return self.converter.convert(src)

        except Exception as e:
            raise RuntimeError(f"Failed to parse document '{src}'") from e