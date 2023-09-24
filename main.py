from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")

import os
os.environ["OPENAI_API_KEY"]= os.getenv("API_KEY")

pdf_loader = DirectoryLoader(os.getenv("DIR_PATH"), glob="**/*.pdf")
txt_loader = DirectoryLoader(os.getenv("DIR_PATH"), glob="**/*.txt")
word_loader = DirectoryLoader(os.getenv("DIR_PATH"), glob="**/*.docx")

loaders = [pdf_loader, txt_loader , word_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())