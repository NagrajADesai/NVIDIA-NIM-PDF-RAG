from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.config import ModelConfig

class LLMChainBuilder:
    """Builds the Conversational Retrieval Chain."""

    def __init__(self):
        self.llm = ChatNVIDIA(
            base_url=ModelConfig.NVIDIA_BASE_URL,
            model_name=ModelConfig.LLM_MODEL,
            temperature=0.3,
            top_p=0.7,
            max_tokens=1024,
        )

    def create_chain(self, retriever):
        """
        Creates a conversational retrieval chain.
        """
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        return conversation_chain
