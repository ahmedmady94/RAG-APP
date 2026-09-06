"""Utility functions for document loading."""

import logging
import pathlib
from typing import Any
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.epub import  UnstructuredEPubLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from streamlit.logger import get_logger


logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO
)

LOGGER = get_logger(__name__)


# ---------------------------------------------------
# Custom EPUB Loader
# ---------------------------------------------------

class EpubReader(UnstructuredEPubLoader):

    def __init__(
        self,
        file_path: str | list[str],
        **unstructured_kwargs: Any
    ):

        super().__init__(
            file_path,
            **unstructured_kwargs,
            mode="elements",
            strategy="fast"
        )


# ---------------------------------------------------
# Exceptions
# ---------------------------------------------------

class DocumentLoaderException(Exception):
    pass


# ---------------------------------------------------
# JSON Loader Factory
# ---------------------------------------------------

def create_json_loader(file_path: str):

    return JSONLoader(
        file_path=file_path,

        # Extract full JSON object
        jq_schema=".[]",

        # Which field becomes page_content
        content_key="content",

        # Keep text content raw
        text_content=False
    )


# ---------------------------------------------------
# Supported Loaders
# ---------------------------------------------------

class DocumentLoader:
    """Loads supported document types."""

    supported_extensions = {

        ".pdf": PyPDFLoader,

        ".txt": TextLoader,

        ".epub": EpubReader,

        ".docx": UnstructuredWordDocumentLoader,

        ".doc": UnstructuredWordDocumentLoader,

        ".json": create_json_loader,
    }


# ---------------------------------------------------
# Main Loader Function
# ---------------------------------------------------

def load_document(file_path: str) -> list[Document]:
    """
    Load a document and return LangChain Documents.
    """

    ext = pathlib.Path(file_path).suffix.lower()

    loader_factory = DocumentLoader.supported_extensions.get(ext)

    if not loader_factory:

        raise DocumentLoaderException(
            f"Unsupported file extension: {ext}"
        )

    loader = loader_factory(file_path)

    docs = loader.load()

    logging.info(f"Loaded {len(docs)} documents from {file_path}")

    return docs