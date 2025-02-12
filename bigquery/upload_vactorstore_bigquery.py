from typing import Any, Dict, List

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_community import GCSDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_community import BigQueryVectorStore

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

import os
import glob

import google.auth
from google.cloud import bigquery
from langchain_community.vectorstores.utils import DistanceStrategy



# 認証情報とプロジェクトIDを取得します。
credentials, PROJECT_ID = google.auth.default()


def main():
    # --- 定数定義 ---
    REGION = "asia-northeast1"
    BUCKET_NAME = "gc_rag_sample_storage"
    GCS_FOLDER = "Inputs"
    EM_MODEL_NAME = "text-multilingual-embedding-002"
    DATASET = "gc_rag_sample_bigquery_dataset"
    TABLE = "gc_rag_sample_bigquery_table"
    RESET_TABLE_FLAG = True  # テーブルをリセットするかどうか

    
    # テキスト エンベディング モデルを定義する (dense embedding)
    embedding_model = VertexAIEmbeddings(model_name=EM_MODEL_NAME)

    # BigQuery Vector Store の初期化
    # データセットの作成
    client = bigquery.Client(project=PROJECT_ID, location=REGION)
    client.create_dataset(dataset=DATASET, exists_ok=True)

    # テーブルのリセット
    if RESET_TABLE_FLAG:
        table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"
        client.delete_table(table_id, not_found_ok=True)  # テーブル削除
        print(f"Deleted table {table_id}")
    else:
        print(f"add data to existing table ({table_id})")
    
    # ベクターストアの定義
    vector_store = BigQueryVectorStore(
            project_id=PROJECT_ID,
            dataset_name=DATASET,
            table_name=TABLE,
            location=REGION,
            embedding=embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )


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
    #ids = ["i_" + str(i + 1) for i in range(len(texts))]
    metadatas = [{"my_metadata": i} for i in range(len(texts))]

    # ---- dense embedding ----
    dense_embeddings = embedding_model.embed_documents(texts)
    
    # embeddingsの中身を確認
    print("dense embeddings（一部）:", dense_embeddings[0][:5])  # 最初の埋め込みを確認
    print("dense embeddings length:", len(dense_embeddings))

    # ---- BigQuery Vector Store にデータを追加する ----
    # https://github.com/langchain-ai/langchain-google/blob/a0027c820dafaddb1af4c0e536c8018c9940b794/libs/community/langchain_google_community/bq_storage_vectorstores/_base.py#L266
    vector_store.add_texts_with_embeddings(
        texts=texts,
        embs=dense_embeddings,
        metadatas=metadatas,
    )

    print("\nデータの登録が完了しました。\n")


if __name__ == "__main__":
    main()
