import io
import PyPDF2
import pdfplumber
from docx import Document as DocxDocument
from openpyxl import load_workbook
from pptx import Presentation
from bs4 import BeautifulSoup
import os
import tempfile
from pathlib import Path

class DocumentProcessor:
    """
    Multi-format document parser
    
    Supported formats:
    - PDF: PyPDF2 + pdfplumber (Vietnamese support)
    - DOCX/DOC: python-docx
    - XLSX: openpyxl
    - PPTX: python-pptx
    - HTML: BeautifulSoup
    - TXT: Plain text
    
    Strategy:
    1. Detect file type từ extension/mime-type
    2. Parse theo format tương ứng
    3. Extract text + metadata
    4. Clean text (remove extra spaces, Vietnamese chars)
    5. Return normalized text
    """
    
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.html', '.txt'}
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.ms-powerpoint',
        'text/html',
        'text/plain'
    }
    
    @staticmethod
    def validate_file(filename: str, mime_type: str) -> tuple[bool, str]:
        """
        Validate file format
        
        Returns: (is_valid, error_message)
        """
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in DocumentProcessor.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type: {ext}. Allowed: {', '.join(DocumentProcessor.ALLOWED_EXTENSIONS)}"
        
        # Check MIME type
        if mime_type not in DocumentProcessor.ALLOWED_MIME_TYPES:
            return False, f"Invalid MIME type: {mime_type}"
        
        return True, ""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes, filename: str) -> str:
        """
        Extract text từ PDF
        
        Strategy:
        1. Try pdfplumber (better Vietnamese support)
        2. Fallback to PyPDF2
        3. Preserve text structure (paragraphs, lists)
        
        Args:
            file_content: Raw PDF bytes
            filename: Filename (for logging)
        
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            # Write to temp file (pdfplumber cần file path)
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            try:
                # Try pdfplumber (best for Vietnamese)
                with pdfplumber.open(tmp_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
            except Exception as e:
                print(f"pdfplumber failed: {e}, trying PyPDF2...")
                
                # Fallback to PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
            
            finally:
                # Clean temp file
                os.unlink(tmp_path)
        
        except Exception as e:
            raise Exception(f"Failed to extract PDF: {str(e)}")
        
        return text if text else "Unable to extract text from PDF"
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """
        Extract text từ DOCX/DOC
        
        Preserves:
        - Paragraphs
        - Headings
        - Lists
        - Tables
        - Vietnamese characters
        
        Args:
            file_content: Raw DOCX bytes
        
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            import io
            docx_file = DocxDocument(io.BytesIO(file_content))
            
            # Extract paragraphs
            for para in docx_file.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            
            # Extract tables
            for table in docx_file.tables:
                text += "\n--- TABLE ---\n"
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    text += row_text + "\n"
        
        except Exception as e:
            raise Exception(f"Failed to extract DOCX: {str(e)}")
        
        return text if text else "Unable to extract text from DOCX"
    
    @staticmethod
    def extract_text_from_xlsx(file_content: bytes) -> str:
        """
        Extract text từ Excel
        
        Includes:
        - All sheets
        - Cell values
        - Preserves structure
        
        Args:
            file_content: Raw XLSX bytes
        
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            import io
            workbook = load_workbook(io.BytesIO(file_content))
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"\n=== SHEET: {sheet_name} ===\n"
                
                for row in sheet.iter_rows():
                    row_text = " | ".join([
                        str(cell.value) if cell.value is not None else ""
                        for cell in row
                    ])
                    if row_text.strip():
                        text += row_text + "\n"
        
        except Exception as e:
            raise Exception(f"Failed to extract XLSX: {str(e)}")
        
        return text if text else "Unable to extract text from XLSX"
    
    @staticmethod
    def extract_text_from_pptx(file_content: bytes) -> str:
        """
        Extract text từ PowerPoint
        
        Includes:
        - Slide titles
        - Text boxes
        - Notes
        
        Args:
            file_content: Raw PPTX bytes
        
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            import io
            prs = Presentation(io.BytesIO(file_content))
            
            for slide_num, slide in enumerate(prs.slides):
                text += f"\n--- SLIDE {slide_num + 1} ---\n"
                
                # Extract from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text += shape.text + "\n"
                    
                    # Extract from tables in slide
                    if shape.has_table:
                        for row in shape.table.rows:
                            row_text = " | ".join([cell.text for cell in row.cells])
                            if row_text.strip():
                                text += row_text + "\n"
        
        except Exception as e:
            raise Exception(f"Failed to extract PPTX: {str(e)}")
        
        return text if text else "Unable to extract text from PPTX"
    
    @staticmethod
    def extract_text_from_html(file_content: bytes) -> str:
        """
        Extract text từ HTML
        
        Removes:
        - Scripts, styles
        - HTML tags
        - Extra whitespace
        
        Args:
            file_content: Raw HTML bytes
        
        Returns:
            Extracted text
        """
        text = ""
        
        try:
            html_str = file_content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_str, 'html.parser')
            
            # Remove script and style
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up: remove multiple spaces/newlines
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
        
        except Exception as e:
            raise Exception(f"Failed to extract HTML: {str(e)}")
        
        return text if text else "Unable to extract text from HTML"
    
    @staticmethod
    def extract_text(file_content: bytes, filename: str, mime_type: str) -> str:
        """
        Main extraction function - route to specific handler
        
        Args:
            file_content: Raw file bytes
            filename: Filename
            mime_type: MIME type
        
        Returns:
            Extracted text (cleaned)
        """
        
        # Validate
        is_valid, error_msg = DocumentProcessor.validate_file(filename, mime_type)
        if not is_valid:
            raise ValueError(error_msg)
        
        ext = Path(filename).suffix.lower()
        
        # Route to appropriate handler
        if ext == '.pdf':
            text = DocumentProcessor.extract_text_from_pdf(file_content, filename)
        elif ext in ['.docx', '.doc']:
            text = DocumentProcessor.extract_text_from_docx(file_content)
        elif ext in ['.xlsx', '.xls']:
            text = DocumentProcessor.extract_text_from_xlsx(file_content)
        elif ext == '.pptx':
            text = DocumentProcessor.extract_text_from_pptx(file_content)
        elif ext == '.html':
            text = DocumentProcessor.extract_text_from_html(file_content)
        elif ext == '.txt':
            text = file_content.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Clean text
        text = DocumentProcessor.clean_text(text)
        
        return text
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text
        
        Remove:
        - Extra whitespace
        - Weird characters
        - Preserve Vietnamese accents
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        import re
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Trim each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

documentProcessor = DocumentProcessor()