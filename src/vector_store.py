import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src import config


def get_embedding_model():
    """
    Retorna el modelo de embeddings. Usa HuggingFace sentence-transformers/all-MiniLM-L6-v2
    por defecto, que es 100% gratuito, ultra rápido y funciona en CPU sin requerir VRAM.
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=config.DEFAULT_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        print(f"[WARN] Error inicializando HuggingFaceEmbeddings: {e}. Reintentando...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class VectorStoreManager:
    """
    Gestor de la base de datos vectorial ChromaDB para indexación, chunking y búsqueda semántica.
    """

    def __init__(self, persist_directory: str | Path = None):
        self.persist_directory = str(persist_directory or config.VECTOR_DB_DIR)
        self.embeddings = get_embedding_model()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> Chroma:
        """Inicializa la instancia de Chroma vector store."""
        return Chroma(
            collection_name="empresa_docs",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Divide los documentos en fragmentos (chunks) y los indexa en ChromaDB.
        Retorna la cantidad de chunks creados.
        """
        if not documents:
            return 0

        chunks = self.text_splitter.split_documents(documents)
        if chunks:
            self.vector_store.add_documents(chunks)
        return len(chunks)

    def search_similarity(
        self,
        query: str,
        k: int = 4,
        category_filter: Optional[str] = None
    ) -> List[Document]:
        """
        Realiza búsqueda semántica por coseno. Permite filtrar por metadato 'category'.
        """
        filter_dict = None
        if category_filter and category_filter != "Todas las áreas":
            filter_dict = {"category": category_filter}

        try:
            if filter_dict:
                results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"[ERROR] Error en búsqueda vectorial: {e}")
            return self.vector_store.similarity_search(query, k=k)

    def get_indexed_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de los fragmentos indexados en la base de datos."""
        try:
            col = self.vector_store._collection
            total_chunks = col.count()
            
            # Obtener desglose básico si hay elementos
            categories_count = {}
            if total_chunks > 0:
                metadatas = col.get(include=["metadatas"])["metadatas"]
                for meta in metadatas:
                    cat = meta.get("category", "Desconocido")
                    categories_count[cat] = categories_count.get(cat, 0) + 1

            return {
                "total_chunks": total_chunks,
                "categories": categories_count
            }
        except Exception as e:
            return {"total_chunks": 0, "categories": {}, "error": str(e)}

    def clear_vector_store(self):
        """Elimina todos los datos indexados y reinicia la colección."""
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass
        
        if Path(self.persist_directory).exists():
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.vector_store = self._init_vector_store()
