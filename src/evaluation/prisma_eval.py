import json
import os
from google import genai
from evaluation.prisma_prompts import (
    FACT_EXTRACTION_PROMPT,
    BATCH_ENTAILMENT_PROMPT,
)

class PRISMAEvaluator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.fact_cache_file = os.path.join(self.cache_dir, "fact_cache.json")
        self.entailment_cache_file = os.path.join(self.cache_dir, "entailment_cache.json")

    def load_cache(self, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_cache(self, filename, cache):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)

    def extract_atomic_facts(self, summary):
        fact_cache = self.load_cache(self.fact_cache_file)
        if summary in fact_cache:
            print("✓ Using cached atomic facts")
            return fact_cache[summary]

        prompt = FACT_EXTRACTION_PROMPT.format(summary=summary)
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw_output = response.text.strip()
        facts = []
        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue
            while (line.startswith("-") or line.startswith("*") or (len(line) > 2 and line[0].isdigit())):
                if "." in line:
                    line = line.split(".", 1)[1].strip()
                else:
                    line = line[1:].strip()
            facts.append(line)

        facts = self.filter_facts(facts)
        fact_cache[summary] = facts
        self.save_cache(self.fact_cache_file, fact_cache)
        return facts

    def filter_facts(self, facts):
        bad_substrings = [
            "someone", "something", "somebody", "is a person", 
            "is a character", "are people", "are characters"
        ]
        filtered = []
        for fact in facts:
            fact = fact.strip()
            if fact == "":
                continue
            if fact != "<MALFORMED SENTENCE>" and len(fact.split()) <= 2:
                continue
            if any(x in fact.lower() for x in bad_substrings):
                continue
            if fact.lower().startswith("there is a") or "is in a room" in fact.lower():
                continue
            if "is talking" in fact.lower() or "are talking" in fact.lower():
                continue
            if "made a statement" in fact.lower() or "is mentioned" in fact.lower() or "are mentioned" in fact.lower():
                continue
            if "is there" in fact.lower() or "are there" in fact.lower() or fact.endswith(" to"):
                continue
            filtered.append(fact)
        return filtered

    def remove_duplicate_facts(self, facts):
        unique = []
        seen = set()
        for fact in facts:
            if fact not in seen:
                unique.append(fact)
                seen.add(fact)
        return unique

    def clean_json(self, text):
        text = text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "")
        if text.startswith("```"):
            text = text.replace("```", "")
        return json.loads(text.strip())

    def batch_entailment(self, generated_facts, reference_facts):
        entailment_cache = self.load_cache(self.entailment_cache_file)
        cache_key = json.dumps({"generated": generated_facts, "reference": reference_facts}, sort_keys=True)

        if cache_key in entailment_cache:
            print("✓ Using cached entailment")
            return entailment_cache[cache_key]

        prompt = BATCH_ENTAILMENT_PROMPT.format(
            generated="\n".join(f"{i+1}. {fact}" for i, fact in enumerate(generated_facts)),
            reference="\n".join(reference_facts)
        )
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        data = self.clean_json(response.text)
        entailment_cache[cache_key] = data["results"]
        self.save_cache(self.entailment_cache_file, entailment_cache)
        return data["results"]

    def compute_precision(self, generated_facts, reference_facts):
        if len(generated_facts) == 0:
            return 0.0
        results = self.batch_entailment(generated_facts, reference_facts)
        return sum(results) / len(generated_facts)

    def compute_recall(self, generated_facts, reference_facts):
        if len(reference_facts) == 0:
            return 0.0
        results = self.batch_entailment(reference_facts, generated_facts)
        return sum(results) / len(reference_facts)

    def evaluate(self, generated_summary, reference_summary):
        generated_facts = self.remove_duplicate_facts(self.extract_atomic_facts(generated_summary))
        reference_facts = self.remove_duplicate_facts(self.extract_atomic_facts(reference_summary))

        precision = self.compute_precision(generated_facts, reference_facts)
        recall = self.compute_recall(generated_facts, reference_facts)

        if precision + recall == 0:
            prisma = 0.0
        else:
            prisma = (2 * precision * recall) / (precision + recall)

        return {
            "generated_facts": generated_facts,
            "reference_facts": reference_facts,
            "fact_precision": round(precision, 4),
            "fact_recall": round(recall, 4),
            "prisma_score": round(prisma, 4)
        }