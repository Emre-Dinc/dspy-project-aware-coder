import os
from typing import List
import dspy
from dspy import Signature, InputField, OutputField
from dspy import LM
from dspy_agent.retrieval import retrieve
from dspy_agent.assertions import RefAssertion
import sys
import json

#Simple for now 
class Query(Signature):
    question: str = InputField(desc="The user's question that needs to be answered.")

class SnippetContext(Signature):
    question: str = InputField(desc="The original user question.")
    snippets: List[str] = InputField(desc="A list of code snippets retrieved as context.")
    rationale: str = OutputField(desc="Explanation of why these snippets were retrieved or how they are relevant.")

class Answer(Signature):
    solution: str = OutputField(desc="The generated answer to the question.")
    references: List[str] = OutputField(desc="List of source snippets or identifiers used to generate the solution.")

class RetrieveModule(dspy.Module):
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k

    def forward(self, question: str) -> SnippetContext: # type: ignore[override]
        hits = retrieve(question, k=self.k)
        snippets = [h["code"] for h in hits]
        return SnippetContext(question=question, snippets=snippets, rationale="vector search")

class ReasonSig(Signature):
    question: str = InputField(desc="The user's question.")
    snippets: List[str] = InputField(desc="Contextual code snippets to help answer the question.")
    solution: str = OutputField(desc="The detailed answer to the question, synthesized from the snippets.")
    references: List[str] = OutputField(desc="The specific snippets or parts of snippets used as references for the solution.")

class ReasonModule(dspy.ChainOfThought): # type: ignore[attr-defined]
    def __init__(self):
        super().__init__(ReasonSig)
    signature = ReasonSig

_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
_MODEL = os.getenv("LLM_MODEL", "DeepSeek-8B-Q6_K")
_RETR_K = int(os.getenv("RETRIEVE_K", "10"))
MAX_TOKEN = int(os.getenv("Max_Token","8000"))
TEMPERATURE= int(os.getenv("Temperature"," 0.6"))
lm = LM(
    model=_MODEL,
    api_base=_API_BASE,
    api_key="LOCAL",
    max_tokens=MAX_TOKEN
    temperature=TEMPERATURE
)
dspy.settings.configure(lm=lm)

retrieve_module = RetrieveModule(k=_RETR_K)
reason_module = ReasonModule()
ref_assertion = RefAssertion()

def answer(question: str) -> Answer: 
    retrieved_context = retrieve_module(question=question)
    
    reasoned_output = reason_module(
        question=retrieved_context.question,
        snippets=retrieved_context.snippets
    )
    
    assertion_passed = ref_assertion(preds=reasoned_output)

    if not assertion_passed:
        print(f"Warning: ReferenceAssertion failed for question: \"{question}\". "
              "The solution may not contain all specified references.")

    final_answer_obj = Answer(
        solution=reasoned_output.solution,
        references=reasoned_output.references
    )
    
    return final_answer_obj

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python -m dspy_agent.pipeline \"<question>\"")
        sys.exit(1)

    user_question = sys.argv[1]
    print(f"Processing question: \"{user_question}\"")

    final_answer = answer(user_question)
    
    output_dict = final_answer.model_dump()
    print(json.dumps(output_dict, indent=2))