"""
============================================================
catalog.py — SHL Product Catalog Parser & Store
============================================================

PURPOSE:
    Load, parse, validate, and provide access to the SHL product
    catalog. This is the SINGLE SOURCE OF TRUTH for all
    assessment data. Every recommendation must come from here.

THEORY — Why a dedicated parser?
    The catalog JSON has 377 assessments with 15 fields each.
    Some fields have inconsistencies (empty strings, missing data,
    control characters). A dedicated parser:
    1. Loads the raw JSON safely
    2. Validates and cleans each entry
    3. Maps 'keys' to test_type codes
    4. Creates a searchable text representation for embeddings
    5. Provides fast lookup by name/URL for post-validation

    If the JSON schema ever changes, ONLY THIS FILE needs updating.

WHAT HAPPENS IF WE REMOVE THIS FILE:
    - No way to load assessment data
    - No validation = garbage in, garbage out
    - No test_type mapping = evaluator fails
    - Every other service breaks (embeddings, retrieval, agent)

INTERVIEW QUESTION:
    Q: "Why parse the JSON at startup instead of on each request?"
    A: "The catalog doesn't change during runtime. Parsing once
       at startup is O(n) one time vs O(n) per request.
       For 377 items, it's fast either way, but parsing once
       is the right pattern. It also lets us catch errors
       early — fail fast at startup, not mid-conversation."
============================================================
"""

import json
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

# Set up logging for this module
logger = logging.getLogger(__name__)


# ============================================================
# TEST TYPE MAPPING
# ============================================================
# Maps the catalog's 'keys' field values to single-letter codes.
# FROM SAMPLE CONVERSATIONS:
#   K = Knowledge & Skills
#   P = Personality & Behavior
#   A = Ability & Aptitude
#   S = Simulations
#   B = Biodata & Situational Judgment
#   C = Competencies
#   D = Development & 360
#   E = Assessment Exercises
#
# WHY A DICT: O(1) lookup, easy to extend, easy to read.
# WHY NOT if/elif: Harder to maintain, violates Open-Closed Principle.
# ============================================================
KEYS_TO_TEST_TYPE: Dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


@dataclass
class Assessment:
    """
    Represents a single SHL assessment from the product catalog.
    
    WHY A DATACLASS (not a Pydantic model)?
        Pydantic is for API validation (external data boundary).
        Dataclasses are for internal data structures (simpler,
        faster, no validation overhead). Our catalog data is
        already validated by the parser, so we don't need
        Pydantic's validation here.
    
    INTERVIEW QUESTION:
        Q: "When would you use a dataclass vs a Pydantic model?"
        A: "Pydantic for API boundaries (request/response). 
           Dataclass for internal data that's already validated.
           Pydantic has ~5x overhead per instantiation due to
           validation. For 377 assessments at startup, either
           works, but dataclass is the semantically correct choice."
    
    Attributes:
        entity_id: Unique identifier from the catalog
        name: Assessment name (used in recommendations)
        url: Catalog URL (used in recommendations)
        description: Detailed description of the assessment
        test_type: Single-letter code(s) derived from 'keys'
        keys: Category labels from the catalog
        job_levels: Target job levels (e.g., ["Graduate", "Mid-Professional"])
        languages: Available languages (e.g., ["English (USA)"])
        duration: Completion time (e.g., "13 minutes", "Untimed")
        remote: Whether remote administration is supported
        adaptive: Whether the test is adaptive
        embedding_text: Combined text used for creating embeddings
    """
    entity_id: str
    name: str
    url: str
    description: str
    test_type: str
    keys: List[str]
    job_levels: List[str]
    languages: List[str]
    duration: str
    remote: str
    adaptive: str
    embedding_text: str = ""


def _derive_test_type(keys: List[str]) -> str:
    """
    Convert the catalog's 'keys' list to test type code(s).
    
    PURPOSE:
        The catalog uses full names like "Knowledge & Skills".
        The API requires short codes like "K".
        This function maps one to the other.
    
    LOGIC:
        - For each key in the assessment's 'keys' list,
          look up the corresponding letter code.
        - Join multiple codes with commas: "K,S" for Knowledge + Simulations.
        - If a key isn't in our mapping, skip it (log a warning).
    
    Args:
        keys: List of key strings from the catalog
              e.g., ["Knowledge & Skills", "Simulations"]
    
    Returns:
        Comma-separated test type codes
        e.g., "K,S"
    
    Examples:
        ["Knowledge & Skills"] → "K"
        ["Personality & Behavior"] → "P"
        ["Knowledge & Skills", "Simulations"] → "K,S"
        [] → "K"  (default to Knowledge)
    """
    # Map each key to its code, skip unknown keys
    codes = []
    for key in keys:
        # Strip whitespace and look up the code
        code = KEYS_TO_TEST_TYPE.get(key.strip())
        if code:
            codes.append(code)
        else:
            # Log unknown keys so we can add them if needed
            logger.warning(f"Unknown assessment key: '{key}' — skipping")
    
    # If no valid codes found, default to "K" (Knowledge)
    # This is a safe default since most assessments test knowledge
    if not codes:
        return "K"
    
    # Join with comma: ["K", "S"] → "K,S"
    return ",".join(codes)


def _build_embedding_text(assessment: dict) -> str:
    """
    Create a rich text representation of an assessment for embedding.
    
    PURPOSE:
        This text is converted to a vector (embedding) for FAISS.
        The vector captures the SEMANTIC MEANING of the assessment.
        When a user searches for "Java developer", the embedding
        of their query will be similar to assessments about Java.
    
    WHY COMBINE MULTIPLE FIELDS:
        A user might search by:
        - Name: "OPQ32r" → we need the name in the text
        - Description: "personality assessment" → need description
        - Skill: "Java" → need description that mentions Java
        - Level: "graduate" → need job_levels
        - Category: "knowledge test" → need keys
        
        By combining all fields, we capture all search angles.
    
    Args:
        assessment: Raw assessment dict from the catalog JSON
    
    Returns:
        A combined text string optimized for embedding
    
    Example output:
        "Core Java (Advanced Level) (New). Multi-choice test that 
        measures the knowledge of Core Java concepts... 
        Categories: Knowledge & Skills. 
        Job levels: Mid-Professional, Professional Individual Contributor.
        Duration: 13 minutes."
    """
    # Get fields with safe defaults for missing data
    name = assessment.get("name", "")
    description = assessment.get("description", "")
    keys = ", ".join(assessment.get("keys", []))
    job_levels = ", ".join(assessment.get("job_levels", []))
    duration = assessment.get("duration", "")
    languages = ", ".join(assessment.get("languages", [])[:5])  # Limit to 5 languages
    
    # Build the combined text
    # Format: "Name. Description. Categories: X. Job levels: Y. Duration: Z."
    parts = [name]
    
    if description:
        parts.append(description)
    
    if keys:
        parts.append(f"Categories: {keys}")
    
    if job_levels:
        parts.append(f"Job levels: {job_levels}")
    
    if duration:
        parts.append(f"Duration: {duration}")
    
    if languages:
        parts.append(f"Languages: {languages}")
    
    # Join with ". " for natural sentence flow
    return ". ".join(parts)


class CatalogService:
    """
    Loads, parses, and provides access to the SHL product catalog.
    
    This is the SINGLE SOURCE OF TRUTH for all assessment data.
    Every recommendation, comparison, and validation goes through here.
    
    DESIGN PATTERN: Repository Pattern
        This class encapsulates all data access logic for the catalog.
        Other services don't need to know where the data comes from
        (file, URL, database) — they just call methods on this class.
    
    INTERVIEW QUESTION:
        Q: "What design pattern is this?"
        A: "Repository pattern — it abstracts the data source.
           If we switch from a JSON file to a database, only
           this class changes. Everything else stays the same."
    
    Attributes:
        assessments: List of all parsed Assessment objects
        _name_lookup: Dict mapping lowercase name → Assessment (for fast validation)
        _url_lookup: Dict mapping URL → Assessment (for fast validation)
    """
    
    def __init__(self):
        """Initialize with empty data. Call load() to populate."""
        self.assessments: List[Assessment] = []
        self._name_lookup: Dict[str, Assessment] = {}
        self._url_lookup: Dict[str, Assessment] = {}
    
    def load(self, catalog_path: str) -> None:
        """
        Load and parse the catalog from a JSON file.
        
        This method does 4 things:
        1. Read the JSON file (handling encoding issues)
        2. Parse each assessment entry
        3. Build lookup dictionaries for fast validation
        4. Log statistics
        
        Args:
            catalog_path: Absolute path to shl_product_catalog.json
        
        Raises:
            FileNotFoundError: If the catalog file doesn't exist
            json.JSONDecodeError: If the JSON is malformed
        """
        logger.info(f"Loading catalog from: {catalog_path}")
        
        # ---- Step 1: Read the JSON file ----
        # strict=False allows control characters that some descriptions contain
        with open(catalog_path, "r", encoding="utf-8", errors="replace") as f:
            raw_data = json.load(f, strict=False)
        
        logger.info(f"Raw catalog contains {len(raw_data)} entries")
        
        # ---- Step 2: Parse each assessment ----
        parsed_count = 0
        skipped_count = 0
        
        for entry in raw_data:
            try:
                assessment = self._parse_entry(entry)
                if assessment:
                    self.assessments.append(assessment)
                    parsed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse entry: {entry.get('name', 'UNKNOWN')} — {e}")
                skipped_count += 1
        
        # ---- Step 3: Build lookup dictionaries ----
        # These enable O(1) validation: "Does this assessment name exist?"
        for assessment in self.assessments:
            # Lowercase name for case-insensitive matching
            self._name_lookup[assessment.name.lower().strip()] = assessment
            # URL for exact matching
            self._url_lookup[assessment.url.strip()] = assessment
        
        # ---- Step 4: Log statistics ----
        logger.info(
            f"Catalog loaded: {parsed_count} assessments parsed, "
            f"{skipped_count} skipped"
        )
    
    def _parse_entry(self, entry: dict) -> Optional[Assessment]:
        """
        Parse a single catalog entry into an Assessment object.
        
        Validates required fields and skips entries that are
        missing critical data (name, link, description).
        
        Args:
            entry: Raw dict from the catalog JSON
        
        Returns:
            Assessment object, or None if entry should be skipped
        """
        # ---- Validate required fields ----
        name = entry.get("name", "").strip()
        url = entry.get("link", "").strip()
        description = entry.get("description", "").strip()
        
        # Skip entries without name or URL (can't recommend them)
        if not name or not url:
            logger.debug(f"Skipping entry with missing name/url: {entry.get('entity_id')}")
            return None
        
        # ---- Extract and clean fields ----
        keys = entry.get("keys", [])
        test_type = _derive_test_type(keys)
        job_levels = entry.get("job_levels", [])
        languages = entry.get("languages", [])
        duration = entry.get("duration", "").strip()
        remote = entry.get("remote", "").strip()
        adaptive = entry.get("adaptive", "").strip()
        entity_id = entry.get("entity_id", "").strip()
        
        # ---- Build embedding text ----
        embedding_text = _build_embedding_text(entry)
        
        return Assessment(
            entity_id=entity_id,
            name=name,
            url=url,
            description=description,
            test_type=test_type,
            keys=keys,
            job_levels=job_levels,
            languages=languages,
            duration=duration,
            remote=remote,
            adaptive=adaptive,
            embedding_text=embedding_text,
        )
    
    def validate_recommendation(self, name: str, url: str) -> bool:
        """
        Check if a recommended assessment actually exists in the catalog.
        
        PURPOSE:
            This is our HALLUCINATION PREVENTION gate. After the
            LLM generates recommendations, we verify every single
            one exists in the real catalog. If it doesn't, we
            remove it from the response.
        
        WHY THIS IS CRITICAL:
            FROM PDF: "Items from catalog only in recommendations"
            is a HARD EVAL (must pass). If the LLM hallucates
            an assessment name, we fail the hard eval = zero score.
        
        Args:
            name: Assessment name from the LLM's response
            url: Assessment URL from the LLM's response
        
        Returns:
            True if the assessment exists in the catalog
        """
        # Check by URL first (most reliable)
        if url.strip() in self._url_lookup:
            return True
        # Fall back to name check (case-insensitive)
        if name.lower().strip() in self._name_lookup:
            return True
        return False
    
    def find_by_name(self, name: str) -> Optional[Assessment]:
        """
        Find an assessment by its name (case-insensitive).
        
        Args:
            name: Assessment name to search for
        
        Returns:
            Assessment object if found, None otherwise
        """
        return self._name_lookup.get(name.lower().strip())
    
    def find_by_url(self, url: str) -> Optional[Assessment]:
        """
        Find an assessment by its catalog URL.
        
        Args:
            url: Catalog URL to search for
        
        Returns:
            Assessment object if found, None otherwise
        """
        return self._url_lookup.get(url.strip())
    
    def get_all_assessment_names(self) -> List[str]:
        """Return all assessment names (useful for debugging)."""
        return [a.name for a in self.assessments]
    
    def get_assessment_details_text(self, assessment: Assessment) -> str:
        """
        Format an assessment's full details as readable text.
        
        Used when building prompt context — gives the LLM rich
        information about each assessment to make good recommendations.
        
        Args:
            assessment: The Assessment object
        
        Returns:
            Formatted text with all assessment details
        """
        details = [
            f"Name: {assessment.name}",
            f"URL: {assessment.url}",
            f"Test Type: {assessment.test_type}",
            f"Categories: {', '.join(assessment.keys)}",
            f"Description: {assessment.description}",
        ]
        
        if assessment.duration:
            details.append(f"Duration: {assessment.duration}")
        if assessment.job_levels:
            details.append(f"Job Levels: {', '.join(assessment.job_levels)}")
        if assessment.languages:
            # Show first 5 languages, then count of remaining
            langs = assessment.languages[:5]
            remaining = len(assessment.languages) - 5
            lang_text = ", ".join(langs)
            if remaining > 0:
                lang_text += f" (+{remaining} more)"
            details.append(f"Languages: {lang_text}")
        if assessment.remote:
            details.append(f"Remote: {assessment.remote}")
        if assessment.adaptive:
            details.append(f"Adaptive: {assessment.adaptive}")
        
        return "\n".join(details)
