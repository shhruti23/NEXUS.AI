from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # Recursive text splitting with semantic boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_file(self, filename: str, content: bytes) -> List[Dict[str, Any]]:
        chunks_with_metadata = []

        if filename.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                splits = self.splitter.split_text(text)
                for split_idx, split in enumerate(splits):
                    chunks_with_metadata.append({
                        "content": split,
                        "metadata": {
                            "source": filename,
                            "page": page_idx + 1,
                            "chunk_id": f"{filename}_p{page_idx+1}_c{split_idx}"
                        }
                    })
        elif filename.lower().endswith((".txt", ".md")):
            text = content.decode("utf-8", errors="ignore")
            splits = self.splitter.split_text(text)
            for split_idx, split in enumerate(splits):
                chunks_with_metadata.append({
                    "content": split,
                    "metadata": {
                        "source": filename,
                        "page": 1,
                        "chunk_id": f"{filename}_c{split_idx}"
                    }
                })
        else:
            raise ValueError(f"Unsupported file format: {filename}. Allowed formats: .pdf, .txt, .md")

        return chunks_with_metadata