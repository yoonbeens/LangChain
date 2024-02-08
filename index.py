# 토큰 정보 로드를 위한 라이브러리
# 설치: pip install python-dotenv
from dotenv import load_dotenv
import os

#토큰 정보 로드
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Runnables"

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


# 1. 질문을 입력받는다
# 2. 프롬프트로 질문이 들어간다
# 3. 프롬프트를 모델로 전달한다
# 4. 결과를 출력한다

from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{country}의 수도는 어디야?")
prompt 

from langchain_openai import ChatOpenAI

model = ChatOpenAI(openai_api_key="")

chain = prompt | model

chain.invoke({"country": "대한민국"})