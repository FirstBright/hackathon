from vector_store import Vector_store
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from vector_store import Vector_store




class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

class ConversationBufferMemory:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.buffer = []

    def add_message(self, message):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(message)

    def get_all_messages(self):
        return " ".join(self.buffer)

class Chat():
    def __init__(self):
        self.memory_buffer = ConversationBufferMemory()
        
    def ask(self, text):
        vector = Vector_store()
        retriever = vector.process_pdf()
        template = '''Answer the question based only on the following context:
        {context}

        Question: {question}
        '''

        prompt = ChatPromptTemplate.from_template(template)
        
        self.memory_buffer = ConversationBufferMemory()
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0,
            streaming=True,
            callbacks=[StreamCallback()],
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        question_message = f"Question: {text}"
        response = rag_chain.invoke(text)
        answer_message = f"Answer: {response}"
        self.memory_buffer.add_message(question_message)
        self.memory_buffer.add_message(answer_message)

        return response