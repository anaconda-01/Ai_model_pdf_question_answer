from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


#creating promt for the chat
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import  StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
#read the data
path="stories.pdf"
loader = UnstructuredPDFLoader(path)
data=loader.load()

#creating vector embedding for chuning the data in vocab
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
chunks=text_splitter.split_documents(data)
vetor_db=Chroma.from_documents(documents=chunks,embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),collection_name="stories")



#RAG model for retriving and using mistral model(pretrianed)
local_modal="mistral"
llm=ChatOllama(model=local_modal,show_progress=True,device="cuda")
print(llm.num_gpu)


import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current GPU Memory Allocated: {torch.cuda.memory_allocated(0)} bytes")
print(f"Current GPU Memory Cached: {torch.cuda.memory_cached(0)} bytes")

#query paratmert for augmentation the  data 
query=PromptTemplate(input_variables=["question"],template="you are Ai model which give the differetn versions of question asked by user which will help to user to overcome on limitation for the distance-based search. proive the question seperated by newline the orignal question:{question}")


#retrive the data vectordb as for chunking the data and feed to mistral model
retriver=MultiQueryRetriever.from_llm(vetor_db.as_retriever(),llm,query)

#writing template for the chat wher  it will ask for the question 

template=""" answer the question based on the following context:{context} and question:{question}"""
chatpromt=ChatPromptTemplate.from_template(template=template)
#chainin the process

chain=({"context":retriver,"question":RunnablePassthrough()}|
       chatpromt|
       llm|
       StrOutputParser()
       )


chain.invoke(input("please enter the question : "))
