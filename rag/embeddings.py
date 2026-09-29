"""오픈소스 임베딩 (sentence-transformers) + 임베딩 캐시."""
from __future__ import annotations

from functools import lru_cache

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.embeddings import Embeddings

from core.config import get_config, path


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class STEmbeddings(Embeddings):
    """모델별 query/passage 접두어(e5 계열 등)를 처리하는 sentence-transformers 래퍼."""

    def __init__(self, model: str, query_prefix: str = "", passage_prefix: str = "",
                 batch_size: int = 16, normalize: bool = True, max_seq_length: int | None = 1024):
        self.model_name = model
        self.qp, self.pp = query_prefix or "", passage_prefix or ""
        self.batch_size, self.normalize, self.max_seq_length = batch_size, normalize, max_seq_length
        self._model = None

    @property
    def model(self):
        """처음 인코딩할 때 불러온다. 재현용 캐시(색인·질의 임베딩)만으로 실행하면 모델(약 2GB)을 받지 않는다."""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # transformers v5 는 저장된 dtype(fp16/bf16)으로 불러와 CPU 에서 10배 이상 느려질 수 있어 fp32 로 고정
            self._model = SentenceTransformer(self.model_name, device=_device(), model_kwargs={"dtype": torch.float32})
            if self.max_seq_length and self._model.max_seq_length > self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length  # 조각이 800자라 1024 토큰이면 충분
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode([self.pp + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=self.normalize, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.qp + text, normalize_embeddings=self.normalize,
                                 show_progress_bar=False).tolist()


def slug(model: str) -> str:
    return model.replace("/", "__")


@lru_cache(maxsize=4)
def get_embeddings(model: str | None = None, query_prefix: str | None = None,
                   passage_prefix: str | None = None) -> Embeddings:
    cfg = get_config()
    model = model or cfg.embedding.model
    base = STEmbeddings(
        model,
        query_prefix=cfg.embedding.query_prefix if query_prefix is None else query_prefix,
        passage_prefix=cfg.embedding.passage_prefix if passage_prefix is None else passage_prefix,
        batch_size=cfg.embedding.batch_size, normalize=cfg.embedding.normalize,
    )
    if not cfg.cache.embeddings:
        return base
    store = LocalFileStore(str(path(f"{cfg.cache.dir}/emb/{slug(model)}/.keep").parent))
    qstore = LocalFileStore(str(path(f"{cfg.cache.dir}/qemb/{slug(model)}/.keep").parent))  # 질의 임베딩 캐시
    return CacheBackedEmbeddings.from_bytes_store(base, store, namespace=slug(model), key_encoder="sha256",
                                                  query_embedding_cache=qstore)
