"""조각 나누기 → 임베딩 → FAISS 저장, 그리고 Kiwi BM25 + FAISS 하이브리드 검색기."""
from __future__ import annotations

import hashlib
import json
import pickle
from functools import lru_cache

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import ROOT, get_config, path
from rag.embeddings import get_embeddings, slug
from rag.loader import corpus_files, load_documents, total_pages

SEPARATORS = ["\n\n", "\n", "다. ", ". ", " ", ""]


def split_documents(docs: list[Document]) -> list[Document]:
    cfg = get_config()
    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.rag.chunk_size, chunk_overlap=cfg.rag.chunk_overlap,
                                              separators=SEPARATORS)
    chunks = []
    for d in docs:
        if d.metadata.get("kind") == "table":  # 표는 자르지 않고 통째로 (너무 길면 잘라서)
            parts = [d] if len(d.page_content) <= cfg.rag.chunk_size * 2 else splitter.split_documents([d])
        else:
            parts = splitter.split_documents([d])
        chunks.extend(parts)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def _corpus_hash() -> str:
    cfg = get_config()
    h = hashlib.sha256()
    for f in corpus_files():
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    h.update((ROOT / cfg.rag.manifest).read_bytes())
    h.update(f"{cfg.rag.chunk_size}-{cfg.rag.chunk_overlap}".encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def get_chunks() -> list[Document]:
    cfg = get_config()
    pages = total_pages()
    if pages > cfg.rag.max_total_pages:
        raise ValueError(f"문서 총 {pages}페이지 > 제한 {cfg.rag.max_total_pages}페이지 (과제 조건 위반)")
    f = path(f"{cfg.cache.dir}/chunks_{_corpus_hash()}.json")
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        return [Document(page_content=d["t"], metadata=d["m"]) for d in data]
    docs, _ = load_documents()
    chunks = split_documents(docs)
    f.write_text(json.dumps([{"t": c.page_content, "m": c.metadata} for c in chunks], ensure_ascii=False),
                 encoding="utf-8")
    return chunks


@lru_cache(maxsize=4)
def get_vectorstore(model: str | None = None, query_prefix: str | None = None,
                    passage_prefix: str | None = None) -> FAISS:
    cfg = get_config()
    model = model or cfg.embedding.model
    emb = get_embeddings(model, query_prefix, passage_prefix)
    d = path(f"{cfg.cache.dir}/faiss/{slug(model)}_{_corpus_hash()}/.keep").parent
    if (d / "index.bin").exists():
        return _load_faiss(d, emb)
    vs = FAISS.from_documents(get_chunks(), emb)
    _save_faiss(vs, d)
    return vs


def _save_faiss(vs: FAISS, d) -> None:
    """faiss 의 C 파일 입출력은 Windows 한글 경로에서 실패하므로 파이썬으로 바이트를 직접 저장한다."""
    import faiss

    (d / "index.bin").write_bytes(faiss.serialize_index(vs.index).tobytes())
    with open(d / "store.pkl", "wb") as f:
        pickle.dump((vs.docstore, vs.index_to_docstore_id), f)


def _load_faiss(d, emb) -> FAISS:
    import faiss
    import numpy as np

    index = faiss.deserialize_index(np.frombuffer((d / "index.bin").read_bytes(), dtype=np.uint8))
    with open(d / "store.pkl", "rb") as f:
        docstore, id_map = pickle.load(f)
    return FAISS(embedding_function=emb, index=index, docstore=docstore, index_to_docstore_id=id_map)


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def kiwi_tokenize(text: str) -> list[str]:
    """형태소 단위 토큰 (명사·동사·형용사 어근·영문·숫자)."""
    keep = ("NN", "VV", "VA", "SL", "SN", "XR", "SH")
    return [t.form.lower() for t in _kiwi().tokenize(text) if t.tag.startswith(keep)]


@lru_cache(maxsize=2)
def get_bm25(k: int | None = None) -> BM25Retriever:
    cfg = get_config()
    return BM25Retriever.from_documents(get_chunks(), preprocess_func=kiwi_tokenize, k=k or cfg.rag.candidate_k)


@lru_cache(maxsize=4)
def get_hybrid_retriever(model: str | None = None, k: int | None = None) -> EnsembleRetriever:
    """Kiwi BM25(키워드) + FAISS(의미) 를 RRF 로 합친다."""
    cfg = get_config()
    ck = k or cfg.rag.candidate_k
    dense = get_vectorstore(model).as_retriever(search_kwargs={"k": ck})
    return EnsembleRetriever(retrievers=[get_bm25(ck), dense], weights=list(cfg.rag.ensemble_weights))


def search(query: str, top_k: int | None = None) -> list[Document]:
    cfg = get_config()
    return get_hybrid_retriever().invoke(query)[: top_k or cfg.rag.top_k]
