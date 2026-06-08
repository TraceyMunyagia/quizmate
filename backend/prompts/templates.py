from langchain_core.prompts import ChatPromptTemplate


GROUNDED_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI study assistant. Answer only from the provided notes "
            "context. Think through the evidence internally before answering, but "
            "do not reveal hidden reasoning. If the context does not contain the "
            "answer, say you do not know from the notes. Keep answers concise and "
            "cite page numbers when available.",
        ),
        (
            "human",
            "Notes context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)
