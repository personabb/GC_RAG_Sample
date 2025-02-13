from typing import Any, List

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_google_cloud_sql_pg.indexes import DistanceStrategy
from langchain_google_cloud_sql_pg import PostgresEngine

from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.retrievers import BaseRetriever
from pydantic import SkipValidation
from langgraph.prebuilt import create_react_agent

import google.auth


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


def print_agent_result_details(result):
    messages = result["messages"]
    for idx, msg in enumerate(messages):
        # メッセージの型名 (HumanMessage, AIMessage, ToolMessageなど)
        msg_type = type(msg).__name__
        print(f"\n=== Message {idx} ===")
        print(f"Type: {msg_type}")

        # content (人間の発話・ツールの返答・AIの返答など) を表示
        if hasattr(msg, "content"):
            print("Content:")
            print(msg.content)

        # AIMessage や ToolMessage だけにあるパラメータを表示したい場合はチェック
        if msg_type == "AIMessage":
            # tool_callsがあれば表示
            if hasattr(msg, "tool_calls"):
                print("Tool calls:")
                print(msg.tool_calls)
            # usage_metadataなど
            if hasattr(msg, "response_metadata"):
                print("Response Metadata:")
                print(msg.response_metadata)

        if msg_type == "ToolMessage":
            # ツールの name など
            if hasattr(msg, "name"):
                print(f"Tool name: {msg.name}")


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

    tool = dence_retriever.as_tool(
        name="Document_Search_Tool",
        description="サービスの規約に関する情報を取得するためのtoolです。"
    )
    
    # Prompt 定義
    prompt_template = """
あなたは、株式会社asapに所属するAIアシスタントです。
ユーザからサービスに関連する質問や雑談を振られた場合に、適切な情報を提供することもが求められています。
ユーザからサービスに関する情報を質問された場合は、toolから情報を取得して回答してください。
toolから情報を取得して回答する場合は、ユーザが一時情報を確認できるように、取得した情報の文章も追加で出力してください。
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{messages}"),
        ]
    )

    # エージェントの作成
    agent = create_react_agent(
        model=llm,
        tools=[tool],
        state_modifier=prompt
    )

    print("\n================= エージェントの実行結果 =================")
    result = agent.invoke({'messages':query})
    print_agent_result_details(result)

    print("\n================= 最終的な出力結果 =================")
    print(result['messages'][-1].content)

  
    

if __name__ == "__main__":
    main()
