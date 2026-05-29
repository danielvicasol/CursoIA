from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from operator import itemgetter

def get_rag_chain(llm, vectorstore):
    """Configura la cadena RAG usando LCEL (LangChain Expression Language)."""
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 1. Prompt para reformular la pregunta (Condensación)
    # Esto convierte una pregunta como "¿Y él?" en "Quién es el autor mencionado..."
    condense_system_prompt = (
        "Dada la siguiente conversación y una pregunta de seguimiento, "
        "reformula la pregunta de seguimiento para que sea una pregunta independiente en español. "
        "No respondas la pregunta, solo reformúlala si es necesario."
    )
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    condense_chain = condense_prompt | llm | StrOutputParser()

    # 2. Prompt para la respuesta final (QA)
    qa_system_prompt = (
        "Eres un asistente experto. Utiliza los siguientes fragmentos de contexto para responder la pregunta. "
        "Si no conoces la respuesta, di que no lo sabes. Responde en español.\n\n"
        "Contexto:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Lógica para decidir qué buscar: la pregunta original o la condensada
    def select_query(input_data):
        if input_data.get("chat_history"):
            return condense_chain
        return input_data["question"]

    # Cadena RAG completa
    full_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(select_query) | retriever
        )
        | RunnableParallel({
            "answer": qa_prompt | llm | StrOutputParser(),
            "source_documents": itemgetter("context")
        })
    )
    
    return full_chain