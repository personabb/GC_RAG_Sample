from typing import Any, Dict, List

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_community import GCSDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

import os
import glob
import uuid
import google.auth

from langchain_google_cloud_sql_pg import PostgresEngine
from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_google_cloud_sql_pg.indexes import DistanceStrategy

# 認証情報とプロジェクトIDを取得します。
credentials, PROJECT_ID = google.auth.default()


def main():
    # --- 定数定義 ---
    REGION = "asia-northeast1"
    BUCKET_NAME = "gc_rag_sample_storage"
    GCS_FOLDER = "Inputs"
    EM_MODEL_NAME = "text-multilingual-embedding-002"
    SQL_INSTANCE = "my-pg-instance"  
    SQL_DATABASE = "postgres"  
    SQL_TABLE_NAME = "vector_store"  
    SQL_DB_USER = "postgres"
    SQL_DB_PASSWORD = "!password!"
    RESET_TABLE_FLAG = True

    
    # テキスト エンベディング モデルを定義する (dense embedding)
    embedding_model = VertexAIEmbeddings(model_name=EM_MODEL_NAME)


    #==============---CloudSQL---================
    # https://github.com/googleapis/langchain-google-cloud-sql-pg-python/blob/3aae0c69eaeb9f2781436363a788a3765d294923/src/langchain_google_cloud_sql_pg/engine.py#L98
    engine = PostgresEngine.from_instance(
        project_id=PROJECT_ID, region=REGION, instance=SQL_INSTANCE, database=SQL_DATABASE, user=SQL_DB_USER, password=SQL_DB_PASSWORD
    )

    if RESET_TABLE_FLAG:
        #Postgre SQLのテーブルを初期化
        engine.init_vectorstore_table(
            table_name=SQL_TABLE_NAME,
            vector_size=768,
            overwrite_existing=True,
        )
    else:
        print("既存のテーブルに追加します。")

    vector_store = PostgresVectorStore.create_sync(
        engine=engine,
        table_name=SQL_TABLE_NAME,
        embedding_service=embedding_model,
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    )

    #===========================================

    # GCS内の指定のフォルダ内のファイルを全てロード
    loader = GCSDirectoryLoader(
        project_name=PROJECT_ID,
        bucket=BUCKET_NAME,
        prefix=GCS_FOLDER
    )

    documents = loader.load()  # 読み込んだドキュメント（List[Document]）

    # チャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],)

    doc_splits = text_splitter.split_documents(documents)

    # チャンクのテキスト部分を抽出
    texts = [doc.page_content for doc in doc_splits]

    # optional IDs とメタデータ
    ids = [str(uuid.uuid4()) for _ in texts]
    metadatas = [{"my_metadata": i} for i in range(len(texts))]

    # ---- dense embedding ----
    dense_embeddings = embedding_model.embed_documents(texts)
    
    # embeddingsの中身を確認
    print("dense embeddings（一部）:", dense_embeddings[0][:5])  # 最初の埋め込みを確認
    print("dense embeddings length:", len(dense_embeddings))

    # ---- Cloud SQL for PostgresSQL Vector Store にデータを追加する ----
    # https://github.com/googleapis/langchain-google-cloud-sql-pg-python/blob/3aae0c69eaeb9f2781436363a788a3765d294923/src/langchain_google_cloud_sql_pg/vectorstore.py#L202
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    print("\nデータの登録が完了しました。\n")


if __name__ == "__main__":
    main()
