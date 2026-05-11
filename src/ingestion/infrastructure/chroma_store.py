"""ChromaDB vector store implementation with enhanced AST entity support.

This module implements IVectorStore using ChromaDB for persistent storage
of code entities extracted via AST parsing.
"""

from pathlib import Path

import chromadb

from src.ingestion.domain.entities import CodeChunk, CodeEntity
from src.ingestion.domain.interfaces import IVectorStore


class ChromaStore(IVectorStore):
    """ChromaDB-based vector store for code entities.

    This implementation supports both legacy CodeChunk format and modern
    CodeEntity format with full AST metadata. All entities are stored
    with rich metadata to enable sophisticated filtering and ranking.
    """

    def __init__(self, collection_name: str = "agentic_source_repo", db_path: str = "./chroma_db"):
        """Initialize ChromaDB connection.

        Args:
            collection_name: Name of the collection to use
            db_path: Path for persistent storage
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.collection_name = collection_name

    async def save_chunks(self, chunks: list[CodeChunk]) -> None:
        """Persist legacy code chunks (backward compatibility).

        Args:
            chunks: List of CodeChunk objects
        """
        if not chunks:
            return

        # Convert to entities for unified storage
        entities = [chunk.to_entity() for chunk in chunks]
        await self.save_entities(entities)

    async def save_entities(self, entities: list[CodeEntity]) -> None:
        """Persist AST-extracted entities to ChromaDB.

        Each entity is stored with:
        - Document: Formatted code with context header for semantic search
        - Metadata: Structured metadata for filtering
        - ID: Unique identifier based on file path and location

        Args:
            entities: List of CodeEntity objects

        Raises:
            ValueError: If entity list is empty
            RuntimeError: If storage operation fails
        """
        if not entities:
            return

        documents = []
        metadatas = []
        ids = []

        for entity in entities:
            documents.append(entity.to_chroma_document())
            metadatas.append(entity.to_metadata())
            ids.append(self._generate_entity_id(entity))

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save entities to ChromaDB: {e}") from e

    async def similarity_search(
        self,
        query: str,
        limit: int = 5,
        filter_dict: dict | None = None,
    ) -> list[CodeEntity]:
        """Search for semantically similar code entities.

        Performs vector similarity search and reconstructs CodeEntity
        objects from the results.

        Args:
            query: The search query text
            limit: Maximum number of results
            filter_dict: Optional ChromaDB metadata filters
                Example: {"entity_type": "FUNCTION"}

        Returns:
            List of CodeEntity objects ranked by relevance

        Raises:
            RuntimeError: If search operation fails
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise RuntimeError(f"Search query failed: {e}") from e

        entities: list[CodeEntity] = []

        if not results["metadatas"] or not results["metadatas"][0]:
            return entities

        for metadata, document, distance in zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0],
        ):
            # Parse entity type
            entity_type_name = metadata.get("entity_type", "STANDALONE")
            from src.ingestion.domain.entities import EntityType

            entity_type = EntityType[entity_type_name]

            # Build entity
            entity = CodeEntity(
                entity_type=entity_type,
                name=metadata.get("name", "unknown"),
                content=self._extract_code_from_document(document),
                file_path=Path(metadata.get("file_path", "unknown.py")),
                start_line=metadata.get("start_line", 1),
                end_line=metadata.get("end_line", 1),
                parent=metadata.get("parent") or None,
                docstring=metadata.get("docstring") or None,
                signature=metadata.get("signature") or None,
                imports=metadata.get("imports", []),
                metadata={
                    "similarity_score": 1 - distance,  # Convert distance to score
                    "entity_type": entity_type_name,
                    **{k: v for k, v in metadata.items() if k not in [
                        "entity_type", "name", "file_path", "start_line",
                        "end_line", "parent", "docstring", "signature", "imports"
                    ]},
                },
            )
            entities.append(entity)

        return entities

    async def get_collection_stats(self) -> dict:
        """Get statistics about the stored collection.

        Returns:
            Dictionary with:
            - total_entities: Total number of entities
            - collection_name: Name of the collection
            - entity_type_counts: Breakdown by entity type
        """
        count = self.collection.count()

        # Get entity type distribution
        # Note: This requires a full scan, so it's done sparingly
        entity_type_counts = {}
        if count > 0:
            all_data = self.collection.get()
            if all_data and all_data.get("metadatas"):
                for metadata in all_data["metadatas"]:
                    if metadata:
                        et = metadata.get("entity_type", "UNKNOWN")
                        entity_type_counts[et] = entity_type_counts.get(et, 0) + 1

        return {
            "total_entities": count,
            "collection_name": self.collection_name,
            "entity_type_counts": entity_type_counts,
        }

    def _generate_entity_id(self, entity: CodeEntity) -> str:
        """Generate unique ID for an entity.

        Format: filepath:start_line:end_line:name
        """
        return f"{entity.file_path}:{entity.start_line}:{entity.end_line}:{entity.name}"

    def _extract_code_from_document(self, document: str) -> str:
        """Extract code content from stored document.

        Documents are stored with a header, so we extract the code block.
        """
        # Look for code block markers
        if "```python" in document:
            parts = document.split("```python")
            if len(parts) > 1:
                code = parts[1]
                if "```" in code:
                    code = code.split("```")[0]
                return code.strip()

        # Fallback: return the document as-is
        return document
