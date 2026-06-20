from google import genai
from evaluation.prisma_prompts import (FACT_EXTRACTION_PROMPT,ENTAILMENT_PROMPT)


class PRISMAEvaluator:

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def extract_atomic_facts(self, summary):

        prompt = FACT_EXTRACTION_PROMPT.format(
            summary=summary
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw_output = response.text.strip()

        print("\n===== GEMINI RAW OUTPUT =====\n")
        print(raw_output)
        print("\n=============================\n")

        facts = []

        for line in raw_output.splitlines():

            line = line.strip()

            if not line:
                continue

            # Remove numbering/bullets if Gemini adds them
            while (
                line.startswith("-")
                or line.startswith("*")
                or (len(line) > 2 and line[0].isdigit())
            ):
                if "." in line:
                    line = line.split(".", 1)[1].strip()
                else:
                    line = line[1:].strip()

            facts.append(line)

        print("Parsed Facts:")
        print(facts)
        print(f"\nTotal Facts: {len(facts)}\n")

        return facts
    
    def check_entailment(self, statement, reference):

        prompt = ENTAILMENT_PROMPT.format(
            statement=statement,
            reference=reference
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text.strip().upper()
        print(answer)

        return answer.startswith("YES")
    
    def compute_precision(self, generated_facts, reference_facts):
        """
        Precision=Generated facts supported by refernce."""

        supported=0
        reference_text="\n".join(reference_facts)
        for fact in generated_facts:
            if self.check_entailment(fact,reference_facts):
                supported +=1
        if len(generated_facts)==0:
            return 0.0
        
        return supported/ len(generated_facts)
    
    def compute_recall(self, generated_facts, reference_facts):
        """
        Recall = Reference facts supported by generated summary
        """

        supported = 0

        generated_text = "\n".join(generated_facts)

        for fact in reference_facts:
            if self.check_entailment(fact, generated_text):
                supported += 1

        if len(reference_facts) == 0:
            return 0.0

        return supported / len(reference_facts)
    def evaluate(self,generated_summary,reference_summary):
        generated_facts=self.extract_atomic_facts(generated_summary)
        reference_facts = self.extract_atomic_facts(reference_summary)
        
        precision = self.compute_precision(generated_facts,reference_facts)

        recall = self.compute_recall(generated_facts,reference_facts)

        if precision + recall == 0:
            prisma=0.0
        else:
            prisma=(
                2*precision*recall)/(precision+recall)
        return {
            "generated_facts": generated_facts,
            "reference_facts": reference_facts,
            "fact_precision": precision,
            "fact_recall": recall,
            "prisma_score": prisma,
        }