"""
Digest, retrieve, and generation

"""

import logging
import os
import re
from datetime import datetime

import torch
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from prompts import contextualize_q_prompt, qa_prompt

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


DOCSTORE_DIR = "docstores"


def load_unstructured_docs(file_path):
    """
    Load the pdf file
    """
    try:
        loader = PyPDFLoader(file_path)
        # loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        logger.info(f"Successfully loaded. type out = {type(docs)}, {type(docs[0])}")
        return docs
    except Exception as e:
        logger.error(f"Error while loading {file_path}\n{str(e)}")
        raise e


def write_tmp_file(uploaded_file, file_path) -> None:
    with open(file_path, "wb") as h:
        h.write(uploaded_file.getvalue())
        logger.info(f"Created temp file at {file_path}")


def chunk_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    logger.info(f"Successfully splitted. type out = {type(splits)}, {type(splits[0])}")
    return splits


def create_docstore_dir() -> None:
    if not os.path.exists(DOCSTORE_DIR):
        os.makedirs(DOCSTORE_DIR)


def get_embedding_model(model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": False}

    hf = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return hf


def create_vectorstore(chunks, persist_directory):
    # Create `DOCSTORE_DIR` dir if not exists
    create_docstore_dir()

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_directory
    )


def get_persist_dir(file_path: str) -> str:
    """
    Get the persist directory for vector store of a single pdf
    """
    file_name = os.path.basename(file_path)
    file_name = re.sub(r"[^\w\s]", "", file_name)
    now = datetime.now()
    # Format date as 'YYYYMMDDHHMMSS'
    date_str = now.strftime("%Y%m%d%H%M%S")
    db_name = os.path.join(DOCSTORE_DIR, f"{file_name}{date_str}")
    return db_name


def digest(file_path):
    docs = load_unstructured_docs(file_path)

    try:
        docs = chunk_docs(docs)
    except Exception as e:
        logger.error(f"Error while chunking for {file_path}\n{str(e)}")
        raise e

    try:
        persist_directory = get_persist_dir(file_path)
        create_vectorstore(docs, persist_directory)
    except Exception as e:
        logger.error(f"Error while creating vectorstore for {file_path}\n{str(e)}")
        raise e

    logger.info(f"Created a vectorstore for {file_path} at {persist_directory} ")

    return persist_directory


def get_llm(
    PATH_TO_MISTRAL_GGUF: str = "/home/tung/.my_ollama_mods/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
):
    N_GPU_LAYERS = 2
    N_BATCH = 512

    llm = LlamaCpp(
        model_path=PATH_TO_MISTRAL_GGUF,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        n_ctx=6000,
        temperature=0,
        top_k=20,
        top_p=0.9,
    )

    llm.verbose = False
    return llm


def get_vectorstore(persist_directory, embedding_model):
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embedding_model
    )
    return vectorstore


def generate_response(prompt, vectorstore_path, chat_history):
    vectorstore = get_vectorstore(vectorstore_path, get_embedding_model())
    retriever = vectorstore.as_retriever()

    llm = get_llm()

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    response = rag_chain.invoke({"input": prompt, "chat_history": chat_history})

    # Here you can also return the relevant context in response
    return response["answer"]


def create_history_from_st_messages(st_messages):
    history = []
    for m in st_messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(m["content"])

    return history
