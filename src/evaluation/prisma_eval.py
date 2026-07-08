import json
import os
import hashlib
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# This loads the environment variables from your .env file
load_dotenv()

# Now your existing code works seamlessly without manual exports!
api_key = os.getenv("GEMINI_API_KEY")

# Bring in your original external prompts
from src.evaluation.prisma_prompts import (
    FACT_EXTRACTION_PROMPT,
    BATCH_ENTAILMENT_PROMPT,
)

# --- Define Pydantic response structures matching your prompt targets ---

class FactExtractionResponse(BaseModel):
    facts: List[str] = Field(description="List of isolated atomic facts extracted from the text summary.")

class EntailmentResponse(BaseModel):
    results: List[bool] = Field(description="Boolean verification evaluations corresponding strictly to the list order of the evaluated facts.")


class PRISMAEvaluator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.fact_cache_file = os.path.join(self.cache_dir, "fact_cache.json")
        self.entailment_cache_file = os.path.join(self.cache_dir, "entailment_cache.json")

    def _hash_key(self, data: Any) -> str:
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def load_cache(self, filename: str) -> Dict[str, Any]:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def save_cache(self, filename: str, cache: Dict[str, Any]):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)

    def extract_atomic_facts(self, summary: str) -> List[str]:
        fact_cache = self.load_cache(self.fact_cache_file)
        text_hash = hashlib.md5(summary.strip().encode("utf-8")).hexdigest()
        
        if text_hash in fact_cache:
            print("✓ Using cached atomic facts")
            return fact_cache[text_hash]

        prompt = FACT_EXTRACTION_PROMPT.format(summary=summary)
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FactExtractionResponse,
                temperature=0.1
            )
        )

        data: FactExtractionResponse = response.parsed
        facts = self.filter_facts(data.facts)
        
        fact_cache[text_hash] = facts
        self.save_cache(self.fact_cache_file, fact_cache)
        return facts

    def filter_facts(self, facts: List[str]) -> List[str]:
        bad_substrings = {
            "someone", "something", "somebody", "is a person", 
            "is a character", "are people", "are characters"
        }
        filtered = []
        for fact in facts:
            fact = fact.strip()
            if not fact or fact == "<MALFORMED SENTENCE>" or len(fact.split()) <= 2:
                continue
            if any(x in fact.lower() for x in bad_substrings):
                continue
            
            fact_lower = fact.lower()
            if fact_lower.startswith("there is a") or "is in a room" in fact_lower:
                continue
            if "is talking" in fact_lower or "are talking" in fact_lower:
                continue
            if "made a statement" in fact_lower or "is mentioned" in fact_lower or "are mentioned" in fact_lower:
                continue
            if "is there" in fact_lower or "are there" in fact_lower or fact.endswith(" to"):
                continue
                
            filtered.append(fact)
        return filtered

    def remove_duplicate_facts(self, facts: List[str]) -> List[str]:
        unique = []
        seen = set()
        for fact in facts:
            if fact not in seen:
                unique.append(fact)
                seen.add(fact)
        return unique

    def batch_entailment(self, generated_facts: List[str], reference_facts: List[str]) -> List[bool]:
        if not generated_facts:
            return []
            
        entailment_cache = self.load_cache(self.entailment_cache_file)
        cache_data = {"generated": sorted(generated_facts), "reference": sorted(reference_facts)}
        cache_key = self._hash_key(cache_data)

        if cache_key in entailment_cache:
            print("✓ Using cached entailment tracking")
            return entailment_cache[cache_key]

        prompt = BATCH_ENTAILMENT_PROMPT.format(
            generated="\n".join(f"{i+1}. {fact}" for i, fact in enumerate(generated_facts)),
            reference="\n".join(reference_facts)
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EntailmentResponse,
                temperature=0.0
            )
        )

        data: EntailmentResponse = response.parsed
        results = data.results

        if len(results) != len(generated_facts):
            results = (results + [False] * len(generated_facts))[:len(generated_facts)]

        entailment_cache[cache_key] = results
        self.save_cache(self.entailment_cache_file, entailment_cache)
        return results

    def compute_precision(self, generated_facts: List[str], reference_facts: List[str]) -> float:
        if not generated_facts:
            return 0.0
        results = self.batch_entailment(generated_facts, reference_facts)
        return sum(1 for r in results if r is True) / len(generated_facts)

    def compute_recall(self, generated_facts: List[str], reference_facts: List[str]) -> float:
        if not reference_facts:
            return 0.0
        
        # To calculate Recall, we evaluate each reference fact against the generated text context.
        # Passing them positionally flips their roles cleanly for batch_entailment.
        results = self.batch_entailment(reference_facts, generated_facts)
    
        return sum(1 for r in results if r is True) / len(reference_facts)

    def evaluate(self, generated_summary: str, reference_summary: str) -> Dict[str, Any]:
        generated_facts = self.remove_duplicate_facts(self.extract_atomic_facts(generated_summary))
        reference_facts = self.remove_duplicate_facts(self.extract_atomic_facts(reference_summary))

        precision = self.compute_precision(generated_facts, reference_facts)
        recall = self.compute_recall(generated_facts, reference_facts)

        prisma = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "generated_facts": generated_facts,
            "reference_facts": reference_facts,
            "fact_precision": round(precision, 4),
            "fact_recall": round(recall, 4),
            "prisma_score": round(prisma, 4)
        }