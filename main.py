"""Main entry point for AgenticSource.

This module serves as the composition root, wiring together all dependencies
and providing a clean interface to the bug fixing system.
"""

import asyncio
import json
from pathlib import Path
from loguru import logger
from src.agent_orchestration.application.bug_fixer_orchestrator import (
    BugFixerOrchestrator,
    FixResult,
)
from src.agent_orchestration.infrastructure.ollama_client import OllamaClient
from src.agent_orchestration.infrastructure.openai_client import OpenAIClient
from src.config import get_settings
from src.detection.application.bug_detection_service import BugDetectionService
from src.detection.domain.entities import BugSeverity
from src.ingestion.application.ingest_code_service import IngestCodeService
from src.ingestion.infrastructure.chroma_store import ChromaStore
from src.ingestion.infrastructure.local_file_system_loader import (
    LocalFileSystemLoader,
)


def create_llm_client(config):
    """Factory function to create appropriate LLM client based on configuration.

    Args:
        config: Application settings

    Returns:
        Configured ILLMClient instance

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If client cannot be created
    """
    if config.llm_provider == "openai":
        if not config.is_openai_configured:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY in .env file"
            )
        return OpenAIClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.openai_base_url if config.openai_base_url else None,
        )
    elif config.llm_provider == "ollama":
        client = OllamaClient(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
        )
        if not client.is_available():
            raise RuntimeError(
                f"Ollama server not available at {config.ollama_base_url}. "
                f"Make sure Ollama is running and model {config.ollama_model} is pulled."
            )
        return client
    else:
        raise ValueError(f"Unknown LLM provider: {config.llm_provider}")


async def ingest_codebase(directory: str = "./src") -> dict:
    """Ingest code from directory into vector store.

    This sets up the knowledge base for the bug fixing system by:
    1. Scanning the directory for Python files
    2. Parsing with AST to extract semantic entities
    3. Indexing in ChromaDB for retrieval

    Args:
        directory: Directory to ingest (default: ./src)

    Returns:
        Ingestion statistics
    """
    logger.info("=" * 60)
    logger.info("CODEBASE INGESTION")
    logger.info("=" * 60)

    # Load configuration
    config = get_settings()

    # Initialize infrastructure
    file_system = LocalFileSystemLoader(
        supported_extensions=(".py", ".pyw"),
        use_ast_chunking=True,
    )
    vector_db = ChromaStore(
        collection_name=config.chroma_collection,
        db_path=str(config.chroma_db_path),
    )

    # Initialize application service
    ingest_service = IngestCodeService(
        file_source=file_system,
        vector_store=vector_db,
    )

    # Run ingestion
    stats = await ingest_service.execute(directory)

    return stats


async def detect_bugs(directory: str, severity: str | None = None, use_llm: bool = False) -> None:
    """Detect bugs in codebase using static analysis.

    Runs multiple static analysis tools and reports found bugs.

    Args:
        directory: Directory to analyze
        severity: Minimum severity to report (ERROR, WARNING, INFO)
        use_llm: Whether to use LLM-based discovery (slower but more thorough)
    """
    logger.info("=" * 60)
    logger.info("BUG DETECTION")
    logger.info("=" * 60)

    # Load configuration
    config = get_settings()
    if severity is None:
        severity = config.detection_severity_threshold

    # Initialize detection service
    detection_service = BugDetectionService()

    # Get LLM client if needed
    llm_client = None
    if use_llm:
        try:
            llm_client = create_llm_client(config)
            logger.info("LLM discovery enabled")
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}")
            logger.warning("Continuing without LLM discovery...")

    # Run detection
    try:
        result = await detection_service.detect_bugs(
            directory,
            use_llm_discovery=use_llm,
            llm_client=llm_client,
        )
    except FileNotFoundError as e:
        logger.error(f"{e}")
        return

    # Filter by severity
    severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    min_level = severity_order.get(severity.upper(), 0)

    filtered_bugs = [
        bug for bug in result.bugs
        if severity_order.get(bug.severity.name, 3) <= min_level
    ]

    # Display results
    logger.info(f"Detection Results:")
    logger.info(f"Files analyzed: {result.files_analyzed}")
    logger.info(f"Tools used: {', '.join(result.tools_run)}")
    logger.info(f"Duration: {result.duration_seconds:.2f}s")
    logger.info(f"Total bugs: {len(result.bugs)}")
    logger.info(f"Showing: {len(filtered_bugs)} (min severity: {severity})")
    logger.info(f"Errors: {result.error_count}")
    logger.info(f"Warnings: {result.warning_count}")
    logger.info(f"Info: {result.info_count}")

    if not filtered_bugs:
        logger.info("No bugs found!")
        return

    logger.info(f"Found {len(filtered_bugs)} bugs:")

    for i, bug in enumerate(filtered_bugs, 1):
        fixable = " [AUTO-FIXABLE]" if bug.is_auto_fixable else ""
        logger.info(f"{i}. {bug}{fixable}")
        if bug.code_snippet:
            snippet = bug.code_snippet[:100].replace("\n", " ")
            if len(bug.code_snippet) > 100:
                snippet += "..."
            logger.info(f"Context: {snippet}")

    if result.errors:
        logger.info(f"Detection errors:")
        for error in result.errors:
            logger.info(f"{error}")


async def detect_and_fix_bugs(
    directory: str,
    severity: str | None = None,
    interactive: bool = True,
    use_llm: bool = False
) -> None:
    """Detect bugs and offer to fix them.

    Runs detection, then iteratively fixes each bug with confirmation.

    Args:
        directory: Directory to analyze and fix
        severity: Minimum severity to fix
        interactive: Whether to ask for confirmation before each fix
    """
    # Load configuration
    config = get_settings()
    if severity is None:
        severity = config.detection_severity_threshold

    # Detect bugs
    detection_service = BugDetectionService()
    # Get LLM client if needed
    llm_client = None
    if use_llm:
        try:
            llm_client = create_llm_client(config)
            logger.info("LLM discovery enabled")
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}")
            logger.warning("Continuing without LLM discovery...")

    # Run detection
    try:
        result = await detection_service.detect_bugs(
            directory,
            use_llm_discovery=use_llm,
            llm_client=llm_client,
        )
    except FileNotFoundError as e:
        logger.error(f"{e}")
        return

    # Filter by severity
    severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    min_level = severity_order.get(severity.upper(), 0)

    bugs_to_fix = [
        bug for bug in result.bugs
        if severity_order.get(bug.severity.name, 3) <= min_level
    ]

    if not bugs_to_fix:
        logger.info("No bugs to fix!")
        return

    logger.info(f"Will attempt to fix {len(bugs_to_fix)} bugs")

    # Setup fix infrastructure
    llm_client = create_llm_client(config)
    vector_db = ChromaStore(
        collection_name=config.chroma_collection,
        db_path=str(config.chroma_db_path),
    )

    # Check if code is indexed
    stats = await vector_db.get_collection_stats()
    if stats["total_entities"] == 0:
        logger.info("Indexing codebase first...")
        await ingest_codebase(directory)

    # Fix each bug
    fixed_count = 0
    failed_count = 0
    skipped_count = 0

    orchestrator = BugFixerOrchestrator.create_default(
        llm_client=llm_client,
        vector_store=vector_db,
        temperature=0.1,
    )

    for i, bug in enumerate(bugs_to_fix, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Bug {i}/{len(bugs_to_fix)}")
        logger.info(f"{bug}")
        logger.info(f"{'='*60}")

        if interactive:
            response = input("\nFix this bug? [y/n/q/all]: ").strip().lower()
            if response == "q":
                logger.info("Quitting...")
                break
            elif response == "all":
                interactive = False  # Auto-approve remaining
            elif response != "y":
                logger.info("Skipped")
                skipped_count += 1
                continue

        # Fix the bug
        try:
            fix_result = await orchestrator.fix_bug(
                user_goal=bug.to_user_goal(),
                max_retries=config.max_retries,
            )

            if fix_result.success:
                logger.info(f"Fixed successfully!")
                if fix_result.fix:
                    logger.info(f"Proposed fix:")
                    logger.info(f"```python\n{fix_result.fix[:200]}...\n```")
                    save_fix_result_report(fix_result, i, config.result_folder)
                fixed_count += 1
            else:
                logger.info(f"Fix failed after {fix_result.retry_count} retries")
                logger.info(f"Feedback: {fix_result.feedback}")
                failed_count += 1

        except Exception as e:
            logger.error(f"Error during fix: {e}")
            failed_count += 1

    # Summary
    logger.info(f"{'='*60}")
    logger.info("FIX SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"   Total bugs: {len(bugs_to_fix)}")
    logger.info(f"   Fixed: {fixed_count}")
    logger.info(f"   Failed: {failed_count}")
    logger.info(f"   Skipped: {skipped_count}")


async def fix_bug(user_goal: str, max_retries: int = 3) -> None:
    """Fix a bug using the multi-agent system.

    Orchestrates Planner, Coder, and Reviewer agents to:
    1. Analyze the bug and retrieve relevant code
    2. Generate a fix
    3. Review and validate the fix
    4. Retry if rejected (up to max_retries)

    Args:
        user_goal: Description of the bug to fix
        max_retries: Maximum retry attempts if fix is rejected
    """
    logger.info("=" * 60)
    logger.info("BUG FIXING WORKFLOW")
    logger.info("=" * 60)

    # Load configuration
    config = get_settings()

    logger.info(f"Configuration:")
    logger.info(f"LLM Provider: {config.llm_provider}")
    logger.info(f"Model: {config.openai_model if config.llm_provider == 'openai' else config.ollama_model}")
    logger.info(f"Max Retries: {max_retries}")

    # Create LLM client
    llm_client = create_llm_client(config)

    # Initialize vector store
    vector_db = ChromaStore(
        collection_name=config.chroma_collection,
        db_path=str(config.chroma_db_path),
    )

    # Check if we have indexed code
    stats = await vector_db.get_collection_stats()
    logger.info(f"Vector Store Stats:")
    logger.info(f"Total entities: {stats['total_entities']}")
    if stats.get("entity_type_counts"):
        logger.info(f"Breakdown: {stats['entity_type_counts']}")

    if stats["total_entities"] == 0:
        logger.info("No code indexed! Run ingestion first:")
        logger.info("python main.py --ingest")
        return

    # Create orchestrator
    orchestrator = BugFixerOrchestrator.create_default(
        llm_client=llm_client,
        vector_store=vector_db,
        temperature=0.1,
    )

    # Execute bug fixing
    result = await orchestrator.fix_bug(
        user_goal=user_goal,
        max_retries=max_retries,
    )
    
    save_fix_result_report(result, 1, config.result_folder)

    # Display result
    logger.info("=" * 60)
    logger.info("FINAL RESULT")
    logger.info("=" * 60)
    logger.info(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))



def save_fix_result_report(fix_result: FixResult, index: int, folder: str):
    """Save fix result as JSON and Markdown reports.
    
    Args:
        fix_result: The FixResult object to save
        index: Index number for the report file names
    """
    import json
    from datetime import datetime
    from pathlib import Path
    
    # Create reports directory if it doesn't exist
    reports_dir = Path(folder)
    reports_dir.mkdir(exist_ok=True)
    
    # Unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"fix_result_{index}_{timestamp}"

    json_path = reports_dir / f"{base_name}.json"
    md_path = reports_dir / f"{base_name}.md"
    
    # Get the fix result as a dictionary
    fix_result_dict = fix_result.to_dict()
    
    # Add timestamp
    fix_result_dict["timestamp"] = timestamp
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            fix_result_dict,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    
    # Format the fix properly for Markdown
    fix_code = fix_result_dict.get("fix") or "No fix generated"
    formatted_fix = f"```python\n{fix_code}\n```" if fix_code != "No fix generated" else fix_code
    
    report = f"""
# Fix Result Report
## Summary
- **Success**: {fix_result_dict.get('success', 'Unknown')}
- **Approved**: {fix_result_dict.get('is_approved', 'Unknown')}
- **Retries**: {fix_result_dict.get('retry_count', 0)}
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## Proposed Fix
{formatted_fix}
## Feedback
{fix_result_dict.get('feedback', 'No feedback provided')}
## Issues Found
"""
    
    if fix_result.issues:
        for i, issue in enumerate(fix_result.issues):
            report += f"{i}. {issue}\n"
    else:
        report += "No issues reported"
    
    # Write markdown report
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Reports saved to {reports_dir}/ (JSON: fix_result_{index}.json, Markdown: fix_result_{index}.md)")

def ask_use_llm() -> bool:
    while True:
        choice = input("\nUse LLM for detection? (y/n): ").strip().lower()

        if choice in ("y", "yes"):
            return True
        elif choice in ("n", "no"):
            return False

        print("Please enter 'y' or 'n'")

async def main():
    """Main entry point with CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AgenticSource - AI-Powered Bug Fixing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ingest
      Ingest codebase into vector store

  %(prog)s --detect ./src
      Detect bugs using static analysis

  %(prog)s --detect-and-fix ./src --interactive
      Detect and fix bugs with confirmation

  %(prog)s --fix "Fix the divide_numbers function"
      Fix a specific bug described by user
        """
    )

    # Existing commands
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest codebase into vector store",
    )
    parser.add_argument(
        "--fix",
        type=str,
        help="Fix a bug (provide description)",
    )

    # New detection commands
    detection_group = parser.add_argument_group("Bug Detection")
    detection_group.add_argument(
        "--detect",
        type=str,
        metavar="DIRECTORY",
        help="Detect bugs in directory using static analysis",
    )
    detection_group.add_argument(
        "--detect-and-fix",
        type=str,
        metavar="DIRECTORY",
        help="Detect bugs and fix them (requires --ingest first)",
    )
    detection_group.add_argument(
        "--severity",
        type=str,
        choices=["ERROR", "WARNING", "INFO"],
        default=None,
        help="Minimum severity to report/fix (default: from config)",
    )
    detection_group.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Ask for confirmation before each fix",
    )
    detection_group.add_argument(
        "--use-llm-discovery",
        action="store_true",
        help="Use LLM-based discovery (slower but finds semantic bugs)",
    )

    # Common arguments
    parser.add_argument(
        "--directory",
        type=str,
        default="./src",
        help="Directory to ingest (default: ./src)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)",
    )

    args = parser.parse_args()

    try:
        if args.ingest:
            # Run ingestion
            stats = await ingest_codebase(args.directory)
            logger.info("Ingestion complete!")
            logger.info(f"Files processed: {stats['total_files']}")
            logger.info(f"Entities indexed: {stats['total_entities']}")

        elif args.detect:
            # Detect only
            await detect_bugs(args.detect, args.severity, args.use_llm_discovery)

        elif args.detect_and_fix:
            # Detect and fix
            await detect_and_fix_bugs(
                args.detect_and_fix,
                args.severity,
                args.interactive,
                args.use_llm_discovery,
            )

        elif args.fix:
            # Fix specific bug
            await fix_bug(args.fix, args.max_retries)

        else:
            # Interactive mode
            logger.info("=" * 60)
            logger.info("AgenticSource - Interactive Mode")
            logger.info("=" * 60)
            logger.info("Commands:")
            logger.info("1. Ingest codebase")
            logger.info("2. Detect bugs")
            logger.info("3. Detect and fix bugs")
            logger.info("4. Fix a specific bug")
            logger.info("5. Exit")

            choice = input("\nSelect option (1-5): ").strip()

            if choice == "1":
                stats = await ingest_codebase(args.directory)
                logger.info("Ingestion complete!")
            elif choice == "2":
                use_llm = ask_use_llm()
                await detect_bugs(args.directory, use_llm=use_llm)
            elif choice == "3":
                use_llm = ask_use_llm()
                await detect_and_fix_bugs(args.directory, interactive=True, use_llm=use_llm)
            elif choice == "4":
                user_goal = input("\nDescribe the bug to fix: ").strip()
                if user_goal:
                    await fix_bug(user_goal, args.max_retries)
                else:
                    logger.info("No bug description provided")
            else:
                logger.info("Goodbye!")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code if exit_code else 0)
