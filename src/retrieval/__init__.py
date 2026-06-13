"""Retrieval module for mRAG pipeline: BiomedCLIP encoder + FAISS retriever."""

from src.retrieval.biomedclip_encoder import BiomedCLIPEncoder
from src.retrieval.faiss_retriever import FAISSRetriever

__all__ = ["BiomedCLIPEncoder", "FAISSRetriever"]
