"""Application service for code ingestion pipeline.

This service orchestrates the ingestion of code from a file source into a vector store.
It uses the new AST-based entity extraction for semantic understanding of code structure.
"""

import time

from src.ingestion.domain.entities import CodeEntity
from src.ingestion.domain.interfaces import IFileSystemLoader, IVectorStore
from loguru import logger


class IngestCodeService:
    """Orchestrates the code ingestion pipeline."""

    def __init__(
        self,
        file_source: IFileSystemLoader,
        vector_store: IVectorStore,
    ):
        """Initialize the ingestion service with dependencies.

        Args:
            file_source: File system loader for reading source files
            vector_store: Vector database for persistent storage
        """
        self.file_source = file_source
        self.vector_store = vector_store

    def execute(self, directory_path: str) -> dict:
        """Execute the full ingestion pipeline.

        Pipeline steps:
        1. Extract entities from files using AST-based parsing
        2. Filter and validate entities
        3. Persist to vector store
        4. Return statistics

        Args:
            directory_path: Path to directory containing source files

        Returns:
            Dictionary with ingestion statistics:
            - total_files: Number of files processed
            - total_entities: Total entities extracted
            - entity_breakdown: Count by entity type
            - duration_seconds: Processing time
        """
        logger.info(f"Starting ingestion for: {directory_path}")
        start_time = time.perf_counter()

        # 1. Extraction: Get entities from the filesystem
        logger.info("Parsing source files with AST...")
        try:
            entities = self.file_source.load_entities(directory_path)
        except FileNotFoundError:
            logger.error(f"Directory not found: {directory_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load files: {e}")
            raise

        if not entities:
            logger.info("No code entities found. Ingestion aborted.")
            return {
                "total_files": 0,
                "total_entities": 0,
                "entity_breakdown": {},
                "duration_seconds": 0.0,
            }

        # 2. Analyze and report
        entity_breakdown = self._analyze_entities(entities)
        unique_files = len(set(e.file_path for e in entities))

        logger.info(f"Extracted {len(entities)} entities from {unique_files} files")
        logger.info("Breakdown:")
        for entity_type, count in sorted(entity_breakdown.items()):
            logger.info(f"{entity_type}: {count}")

        # 3. Persistence: Save to vector store (synchronous)
        logger.info(f"Indexing to vector store...")
        try:
            self.vector_store.save_entities(entities)
        except Exception as e:
            logger.error(f"Error during vector storage: {e}")
            raise

        end_time = time.perf_counter()
        duration = end_time - start_time

        logger.info(f"Successfully indexed {len(entities)} entities in {duration:.2f} seconds")

        return {
            "total_files": unique_files,
            "total_entities": len(entities),
            "entity_breakdown": entity_breakdown,
            "duration_seconds": duration,
        }

    def _analyze_entities(self, entities: list[CodeEntity]) -> dict[str, int]:
        """Analyze entity distribution by type.

        Args:
            entities: List of extracted entities

        Returns:
            Dictionary mapping entity type names to counts
        """
        breakdown: dict[str, int] = {}
        for entity in entities:
            type_name = entity.entity_type.name
            breakdown[type_name] = breakdown.get(type_name, 0) + 1
        return breakdown
