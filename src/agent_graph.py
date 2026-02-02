from typing import List, Annotated, Dict, TypedDict, Any
from langgraph.graph import StateGraph, END

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.docstore.document import Document
from src.config import ModelConfig
import json

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    steps: List[str]  # Trace of agent thoughts

# --- Nodes ---

class AgentNodes:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatNVIDIA(
            base_url=ModelConfig.NVIDIA_BASE_URL,
            model_name=ModelConfig.LLM_MODEL,
            temperature=0, # Low temp for reasoning
            max_tokens=1024,
        )
        self.gen_llm = ChatNVIDIA(
            base_url=ModelConfig.NVIDIA_BASE_URL,
            model_name=ModelConfig.LLM_MODEL,
            temperature=0.3, # Slight creep for generation
            max_tokens=1024,
        )

    def retrieve(self, state: AgentState):
        """
        Retrieve documents from vector store.
        """
        question = state["question"]
        steps = state.get("steps", [])
        steps.append("Retrieving documents from Vector DB...")
        
        # Retrieval
        documents = self.retriever.invoke(question)
        
        steps.append(f"Retrieved {len(documents)} documents.")
        return {"documents": documents, "question": question, "steps": steps}

    def grade_documents(self, state: AgentState):
        """
        Determines whether the retrieved documents are relevant to the question.
        """
        question = state["question"]
        documents = state["documents"]
        steps = state.get("steps", [])
        steps.append("Grading retrieved documents for relevance...")

        # Grader Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            If the document contains keywords or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
            input_variables=["question", "document"],
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        
        filtered_docs = []
        for d in documents:
            try:
                score = chain.invoke({"question": question, "document": d.page_content})
                grade = score.get("score", "no")
            except:
                grade = "yes" # Fallback to keeping it if parsing fails
            
            if grade == "yes":
                filtered_docs.append(d)
        
        steps.append(f"Grading complete. {len(filtered_docs)}/{len(documents)} documents relevant.")
        
        return {"documents": filtered_docs, "question": question, "steps": steps}

    def generate(self, state: AgentState):
        """
        Generate answer.
        """
        question = state["question"]
        documents = state["documents"]
        steps = state.get("steps", [])
        steps.append("Generating final answer...")

        # Generator Prompt
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \n
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \n
            
            Question: {question} \n
            Context: {context} \n
            
            Answer:""",
            input_variables=["question", "context"],
        )
        
        # Format context
        context = "\n\n".join([d.page_content for d in documents])
        
        rag_chain = prompt | self.gen_llm | StrOutputParser()
        
        try:
            generation = rag_chain.invoke({"context": context, "question": question})
        except Exception as e:
            generation = f"Error during generation: {e}"
            
        steps.append("Generation complete.")
        return {"documents": documents, "question": question, "generation": generation, "steps": steps}


    def document_router(self, state: AgentState):
        """
        Route question to Retrieval or End (if chat).
        """
        # For this implementation, we will assume all inputs in "Chat with Data" are meant for RAG 
        # unless explicitly just "hi".
        # We can make this smarter later.
        # But per requirements, let's keep it simple: Always Retrieve for now, 
        # or implement a simple check.
        
        # Let's implement a simple router using the LLM for the "Agentic" requirement
        question = state["question"]
        steps = state.get("steps", [])
        if not steps:
            steps = ["Agent started."] # Initialize steps if empty
        
        prompt = PromptTemplate(
            template="""You are an expert at routing a user question to a vectorstore or general chat. \n
            Use the vectorstore for questions on specific topics or documents. \n
            Query: {question} \n
            Return a JSON with a single key 'datasource' and value 'vectorstore' or 'chat'.""",
            input_variables=["question"]
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        try:
            source = chain.invoke({"question": question})
            decision = source.get("datasource", "vectorstore")
        except:
            decision = "vectorstore" # Default default
            
        if decision == "vectorstore":
            steps.append("Router: Routing to Vector Store.")
            # We can't return state here effectively in conditional edge, 
            # so we just return the decision string
            return "retrieve"
        else:
            steps.append("Router: Routing to General Chat (skip retrieval).")
            return "generate_no_rag"

# --- Graph Construction ---

def build_graph(retriever):
    workflow = StateGraph(AgentState)
    nodes = AgentNodes(retriever)

    # Define Nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    
    # Simple direct generation node for non-RAG
    def generate_chat(state):
        steps = state.get("steps", [])
        steps.append("Generating details without context...")
        return {"generation": "I am a RAG agent. I can only help with document questions for now.", "steps": steps}
    
    workflow.add_node("generate_no_rag", generate_chat)

    # Define Edges
    workflow.set_conditional_entry_point(
        nodes.document_router,
        {
            "retrieve": "retrieve",
            "generate_no_rag": "generate_no_rag"
        }
    )
    
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("generate_no_rag", END)

    app = workflow.compile()
    return app
