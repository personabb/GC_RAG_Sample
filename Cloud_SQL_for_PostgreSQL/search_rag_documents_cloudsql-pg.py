from typing import Any, List

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_core.documents import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_core.retrievers import BaseRetriever
from pydantic import SkipValidation

import google.auth

from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_google_cloud_sql_pg.indexes import DistanceStrategy
from langchain_google_cloud_sql_pg import PostgresEngine


# 認証情報とプロジェクトIDを取得します。
credentials, PROJECT_ID = google.auth.default()

class VectorSearchRetriever(BaseRetriever):
    """
    ベクトル検索を行うためのRetrieverクラス。
    """
    vector_store: SkipValidation[Any]
    embedding_model: SkipValidation[Any]
    k: int = 5 # 返すDocumentのチャンク数

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Dense embedding
        embedding = self.embedding_model.embed_query(query)
        search_results = self.vector_store.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=self.k,
        )

        # Document のリストだけ取り出す
        return [doc for doc, _ in search_results]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def main():
    # --- 定数定義 ---
    REGION = "asia-northeast1"
    EM_MODEL_NAME = "text-multilingual-embedding-002"
    LLM_MODEL_NAME = "gemini-2.0-flash-001" 
    SQL_INSTANCE = "my-pg-instance"  
    SQL_DATABASE = "postgres"  
    SQL_TABLE_NAME = "vector_store"  
    SQL_DB_USER = "postgres"
    SQL_DB_PASSWORD = "!password!"


    # 質問文
    query = "16歳未満のユーザーが海外から当社サービスを利用した場合、親権者が同意していないときはどう扱われますか？ そのときデータは国外にも保存される可能性がありますか？"
    #query = "おはよう"
    #query = "今日は12歳の誕生日なんだ。これから初めて海外に行くんだよね"


   
    # テキスト エンベディング モデルを定義する (dense embedding)
    embedding_model = VertexAIEmbeddings(model=EM_MODEL_NAME)

    #==============---CloudSQL---================

    engine = PostgresEngine.from_instance(
        project_id=PROJECT_ID, region=REGION, instance=SQL_INSTANCE, database=SQL_DATABASE, user=SQL_DB_USER, password=SQL_DB_PASSWORD
    )

    vector_store = PostgresVectorStore.create_sync( 
        engine=engine,
        table_name=SQL_TABLE_NAME,
        embedding_service=embedding_model,
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    )

    #===========================================

    # Chatモデル (LLM)
    llm = ChatVertexAI(
        model_name=LLM_MODEL_NAME,
        max_output_tokens=512,
        temperature=0.2
    )

    
    # DenceRetrieverを用意
    dence_retriever = VectorSearchRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        k=5,
    )

    #dence_retriever = vector_store.as_retriever() # search_kwargs={'k': 4}

    
    # Prompt 定義
    prompt_template = """
あなたは、株式会社asapに所属するAIアシスタントです。
ユーザからサービスに関連する質問や雑談を振られた場合に、適切な情報を提供することもが求められています。
ユーザからサービスに関する情報を質問された場合は、下記の情報源から情報を取得して回答してください。
下記の情報源から情報を取得して回答する場合は、ユーザが一時情報を確認できるように、取得した情報の文章も追加で出力してください。

情報源
{context}
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{query}"),
        ]
    )


    # チェーンを定義（retriever で文脈を取り、Prompt に当てはめて、LLM へ）
    chain = (
        {"context": dence_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 実行
    print("===== DenseRetriever の実行結果 =====")
    dense_docs = dence_retriever.invoke(query)
    print("\nDenseRetrieved Documents:", dense_docs)

    print("\n================= LLMの実行結果 =================")
    result = chain.invoke(query)
    print(result)

  
    

if __name__ == "__main__":
    main()
