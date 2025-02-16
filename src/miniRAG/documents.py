from typing import Dict


class Document:
    def __init__(self, page_content: str, metadata: Dict[str, str] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata else {}

    def __repr__(self):
        return f"Document(page_content={self.page_content[:30]}..., metadata={self.metadata})"
