from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant whos purpose is to answer questions about the time machine for peace social invention program. You are given the following extracted parts of a book called A Study In Peace and a question. Provide a conversational answer. If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer. Lists should be itemized.

Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(input_variables=["question", "context"], template=template)
'''
def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
       # qa_prompt=QA_PROMPT,
       # condense_question_prompt=CONDENSE_QUESTION_PROMPT
    )
    print(CONDENSE_QUESTION_PROMPT)
    print(QA_PROMPT)
    return qa_chain
'''
chat_gpt = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def get_chain(vectorstore):
    #streaming_llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
    question_generator = LLMChain(llm=chat_gpt, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm=chat_gpt, chain_type="stuff", prompt=QA_PROMPT)
    qa_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator
    )
    return qa_chain

