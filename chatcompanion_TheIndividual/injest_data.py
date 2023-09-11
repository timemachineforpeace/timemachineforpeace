# The Individual AI Assistant: (inject_data.py)
#
# an LLM chat bot for facilitating solo-person conversation over custom corpus +
# OpenAI corpus (c2021), with local corrective fine-tuning.
#
# The purpose of this bot is to facilitate the universal piece process and make
# the operator's world piece computer less shitty.
#
# License: The Human Imperative...
#
#          ...use this to maintain the universal piece and satisfy The Human Imperative.
#             Failure to do so with mal-intent will result in legal action
#             to protect the service-marked time machine for peace social
#             invention program, under the purview of the federal government
#             for The United States of America. Ask The Individual if you need
#             additional clarification.
#
# BEGINS #################################################################################

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import subprocess
import os

#subprocess.call(["rm", "-r", ".vectordb"])

# TODO: the loader shit needs to be iterable over docs in directory
#directory = os.fsencode(directory_in_str)

# specify the embedding function
embeddings = OpenAIEmbeddings()

# these are all unstructured texts that DO NOT care about chunk size
txt_loader_study = UnstructuredFileLoader('./data/plaintext_astudyinpeace.txt')
txt_loader_sign1 = UnstructuredFileLoader('./data/marketresearchbooth_sign1.txt')
txt_loader_sign2 = UnstructuredFileLoader('./data/marketresearchbooth_sign2.txt')
txt_loader_sign3 = UnstructuredFileLoader('./data/marketresearchbooth_sign3.txt')
txt_loader_sign4 = UnstructuredFileLoader('./data/marketresearchbooth_sign4.txt')
txt_loader_sign5 = UnstructuredFileLoader('./data/marketresearchbooth_sign5.txt')

# these are all unstructured texts that DO care about chunk size
txt_loader_nolap_doc1 = UnstructuredFileLoader('./data/marketresearchbooth_docx1.txt')

# these are all the loaders for texts that don't care about chunk size
# ...again, this should be dynamic
loaders = [
    txt_loader_study,
    txt_loader_sign1,
    txt_loader_sign2,
    txt_loader_sign3,
    txt_loader_sign4,
    txt_loader_sign5,
]

# these are all the loaders for texts that don't care about chunk size
# ...again, this should be dynamic
loaders_nolap = [
    txt_loader_nolap_doc1
]

# create text splitting objects for both types of loaders
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
text_splitter_nolap = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)

# uncomment this if you need to instantiate the feedback vector store

# this is a dummy loader to make feedback vector store
feedback_loader = UnstructuredFileLoader('./data/blank.txt')

# now initialize store
feedback = []
feed = feedback + text_splitter.split_documents(feedback_loader.load())
feedback_store = Chroma.from_documents(feed, embeddings, persist_directory=".feedbackdb/")
feedback_store.persist()

# split that shit
texts = []
for text in loaders:
    texts = texts + text_splitter.split_documents(text.load())
for text in loaders_nolap:
    texts = texts + text_splitter_nolap.split_documents(text.load())

# store it all in the main vector store
vector_store = Chroma.from_documents(texts, embeddings, persist_directory=".vectordb/")
vector_store.persist()

# ENDS ###################################################################################
# .
# .
# .
# _we need to erect a global peace system_ - tW
