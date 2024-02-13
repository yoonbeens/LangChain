from google.colab import drive
drive.mount('/content/drive')


from IPython.display import Markdown


import os
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"


from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("네이버에 대해 보고서를 작성해줘")
Markdown(result.content)


# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader


loader = PyPDFLoader("/content/drive/MyDrive/강의 자료/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(pages)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

docsearch = Chroma.from_documents(texts, hf)


retriever = docsearch.as_retriever(
                                    search_type="mmr",
                                    search_kwargs={'k':3, 'fetch_k': 10})
retriever.get_relevant_documents("혁신성장 정책금융에 대해서 설명해줘")


from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

template = """Answer the question as based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature = 0)

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x['question']),
    "question": lambda x: x['question']
}) | prompt | gemini


Markdown(chain.invoke({'question': "혁신성장 정책금융에 대해서 설명해줘"}).content)