import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_recall,
)
from src.llm_chain import LLMChainBuilder
from src.retrieval_engine import RetrievalEngine
from src.config import ModelConfig

# Note: Ragas requires an OpenAI API key by default for evaluation metrics 
# unless configured with a different LLM for evaluation.
# Ensure OPENAI_API_KEY is in .env if using default Ragas configuration.

class RAGEvaluator:
    """Helper class to evaluate RAG pipeline using Ragas."""
    
    def __init__(self, chain):
        self.chain = chain

    def evaluate_pipeline(self, questions: list, ground_truths: list):
        """
        Runs the RAG pipeline on questions and evaluates using Ragas.
        """
        answers = []
        contexts = []

        print("🔮 Generating answers for evaluation...")
        for query in questions:
            response = self.chain({'question': query})
            answers.append(response['answer'])
            # Extract page content from source documents
            valid_contexts = [doc.page_content for doc in response['source_documents']]
            contexts.append(valid_contexts)

        # Prepare dataset for Ragas
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        print("📊 Running Ragas evaluation...")
        results = evaluate(
            dataset = dataset,
            metrics=[
                faithfulness,
                answer_relevance,
                context_recall,
            ],
        )

        return results

if __name__ == "__main__":
    # Example Usage
    print("This is a template script. To use it, instantiate the components and provide data.")
    # Example:
    # retrieval_engine = RetrievalEngine()
    # retrieval_engine.initialize_vector_store(["dummy chunks"]) # Needs actual data
    # llm_builder = LLMChainBuilder()
    # retriever = retrieval_engine.get_hybrid_retriever() # Or wrapped with reranker
    # chain = llm_builder.create_chain(retriever)
    
    # evaluator = RAGEvaluator(chain)
    # results = evaluator.evaluate_pipeline(
    #     questions=["What is NVIDIA NIM?", "How does RAG work?"],
    #     ground_truths=["NVIDIA NIM is...", "RAG works by..."]
    # )
    # print(results)
