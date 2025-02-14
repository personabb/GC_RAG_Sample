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

#from langchain_community.vectorstores import MatchingEngine
from langchain_google_vertexai import VectorSearchVectorStore


# 認証情報とプロジェクトIDを取得します。
credentials, PROJECT_ID = google.auth.default()


def main():
    # --- 定数定義 ---
    REGION = "asia-northeast1"
    BUCKET_NAME = "gc_rag_sample_storage_vertexai"
    GCS_FOLDER = "Inputs"
    EM_MODEL_NAME = "text-multilingual-embedding-002"
    INDEX_ID = "6427709791606407168"
    ENDPOINT_ID = "4670197629211115520"
    
    # テキスト エンベディング モデルを定義する (dense embedding)
    embedding_model = VertexAIEmbeddings(model_name=EM_MODEL_NAME)


    # VectorSearchVectorStore の初期化
    vector_store = VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_bucket_name=BUCKET_NAME,
        index_id=INDEX_ID,
        endpoint_id=ENDPOINT_ID,
        embedding=embedding_model,
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
    ids = ["i_" + str(i + 1) for i in range(len(texts))]
    metadatas = [{"my_metadata": i} for i in range(len(texts))]

    # ---- dense embedding ----
    dense_embeddings = embedding_model.embed_documents(texts)
    
    # embeddingsの中身を確認
    print("dense embeddings（一部）:", dense_embeddings[0][:5])  # 最初の埋め込みを確認
    print("dense embeddings length:", len(dense_embeddings))

    # ---- Matching Engine にデータを追加する ----
    vector_store.add_texts_with_embeddings(
        texts=texts,
        embeddings=dense_embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print("\nデータの登録が完了しました。\n")


if __name__ == "__main__":
    main()
