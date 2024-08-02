from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

class Vector_store:   
    def process_pdf(self):
        # PDF 파일 로드
        loader = PyPDFLoader("./PDF.pdf")
        docs = loader.load()
        # print(docs)
        # 텍스트 스플리터 생성
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

        # 스플리트된 텍스트들의 리스트 반환
        splits = text_splitter.split_documents(docs)
        # print(len(splits))

        # split 해놓은 텍스트 데이터를 embedding한 VectorDB를 생성
        # 문서에 대한 Chroma 객체를 생성하고, Google Generative AI Embeddings 모델을 사용하여 문장을 임베딩
        vectordb = Chroma.from_documents(documents=splits, 
                                          embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
      
        # 사용자 입력(질문)과 비교해서 가까운 K개의 결과 찾기        
        return  vectordb.as_retriever(k=1)

