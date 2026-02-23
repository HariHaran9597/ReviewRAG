from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# RAGAS specific imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Internal wrappers
from app.generator import get_groq_llm
from app.embedder import get_bge_embeddings

# Langchain wrapper modules for backward compatibility needed in custom RAGAS setups
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

def evaluate_answer(
    question: str,
    answer: str,
    retrieved_contexts: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluates the newly generated answer using the RAGAS framework.
    Uses the Groq LLM ('moonshotai/kimi-k2-instruct-0905') via a custom wrapper
    so that we don't have to pay for OpenAI API keys.
    """
    try:
        # Define LLM and Embeddings for Evaluation.
        # We pass them precisely to bypass the default OpenAI requirement.
        eval_llm = get_groq_llm()
        eval_embeddings = get_bge_embeddings()

        # Format retrieved contexts as a simple list of string contents for RAGAS
        contexts = [chunk["page_content"] for chunk in retrieved_contexts]
        
        # In ragas, we build a single HuggingFace 'dataset'
        data = {
            "question": [question],
            "contexts": [contexts],
            "answer": [answer],
            # To measure specific variables, it can be extended. 
            # We omit "ground_truth" as we only test faithfulness and answer_relevancy right now.
        }
        dataset = Dataset.from_dict(data)

        # -------------------------------------------------------------
        # Note on Ragas Versions:
        # In newer Ragas versions, it is recommended to bind your model 
        # objects explicitly. Ragas typically wraps the LLM into an 
        # internal object (llm_factory). 
        # For simplicity, we directly pass `llm` and `embeddings` in `.run()`.
        # -------------------------------------------------------------

        # Run Eval Engine
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,       # Ensures the answer doesn't hallucinate facts not found in 'contexts'
                answer_relevancy    # Ensures the answer actually addresses the user's 'question'
            ],
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False  # Do not blow up if evaluation times out
        )

        return {
            "faithfulness": result.get("faithfulness", 0.0),
            "answer_relevancy": result.get("answer_relevancy", 0.0)
        }
    except Exception as e:
        print(f"RAGAS evaluation failed (sometimes happens due to free-tier rate limits or timeouts): {e}")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "error": str(e)
        }
