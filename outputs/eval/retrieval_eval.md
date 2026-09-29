질문 세트 {'natural': 40, 'keyword': 30} (정답 = 같은 문서·페이지), 조각 471개

## 종합 (두 세트 평균)

| retriever      | embedding                                  |   Hit@1 |   Hit@3 |   Hit@5 |   MRR@5 |
|:---------------|:-------------------------------------------|--------:|--------:|--------:|--------:|
| Dense          | nlpai-lab/KURE-v1                          |   0.8   |   0.938 |   0.971 |   0.868 |
| Dense          | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.771 |   0.954 |   0.971 |   0.864 |
| Dense          | BAAI/bge-m3                                |   0.734 |   0.929 |   0.958 |   0.831 |
| Hybrid 0.3:0.7 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.684 |   0.917 |   0.988 |   0.804 |
| Dense          | Qwen/Qwen3-Embedding-0.6B                  |   0.704 |   0.904 |   0.942 |   0.803 |
| Hybrid 0.3:0.7 | nlpai-lab/KURE-v1                          |   0.666 |   0.946 |   0.988 |   0.8   |
| Hybrid 0.3:0.7 | BAAI/bge-m3                                |   0.688 |   0.884 |   0.971 |   0.797 |
| Dense          | intfloat/multilingual-e5-large             |   0.684 |   0.85  |   0.866 |   0.763 |
| Hybrid 0.3:0.7 | Qwen/Qwen3-Embedding-0.6B                  |   0.629 |   0.838 |   0.958 |   0.752 |
| Hybrid 0.5:0.5 | nlpai-lab/KURE-v1                          |   0.6   |   0.917 |   0.958 |   0.748 |
| Hybrid 0.5:0.5 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.596 |   0.862 |   0.958 |   0.741 |
| Hybrid 0.3:0.7 | intfloat/multilingual-e5-large             |   0.659 |   0.775 |   0.871 |   0.736 |
| Hybrid 0.5:0.5 | BAAI/bge-m3                                |   0.587 |   0.866 |   0.929 |   0.728 |
| Hybrid 0.5:0.5 | Qwen/Qwen3-Embedding-0.6B                  |   0.612 |   0.812 |   0.888 |   0.718 |
| Hybrid 0.5:0.5 | intfloat/multilingual-e5-large             |   0.612 |   0.788 |   0.854 |   0.708 |
| Kiwi BM25      | -                                          |   0.6   |   0.675 |   0.721 |   0.644 |
| Hybrid 0.5:0.5 | intfloat/multilingual-e5-small             |   0.542 |   0.704 |   0.729 |   0.619 |
| Dense          | intfloat/multilingual-e5-small             |   0.504 |   0.638 |   0.666 |   0.575 |
| Hybrid 0.3:0.7 | intfloat/multilingual-e5-small             |   0.496 |   0.654 |   0.679 |   0.574 |

## 세트별

| set     | retriever      | embedding                                  |   Hit@1 |   Hit@3 |   Hit@5 |   MRR@5 |   query_ms |   load_or_build_sec |
|:--------|:---------------|:-------------------------------------------|--------:|--------:|--------:|--------:|-----------:|--------------------:|
| natural | Kiwi BM25      | -                                          |   0.5   |   0.55  |   0.575 |   0.526 |        0.6 |                     |
| keyword | Kiwi BM25      | -                                          |   0.7   |   0.8   |   0.867 |   0.761 |        0.3 |                     |
| natural | Dense          | BAAI/bge-m3                                |   0.6   |   0.925 |   0.95  |   0.759 |       22.2 |                13.8 |
| keyword | Dense          | BAAI/bge-m3                                |   0.867 |   0.933 |   0.967 |   0.903 |       20.5 |                13.8 |
| natural | Hybrid 0.5:0.5 | BAAI/bge-m3                                |   0.475 |   0.8   |   0.925 |   0.645 |       22.1 |                     |
| keyword | Hybrid 0.5:0.5 | BAAI/bge-m3                                |   0.7   |   0.933 |   0.933 |   0.811 |       21.1 |                     |
| natural | Hybrid 0.3:0.7 | BAAI/bge-m3                                |   0.575 |   0.8   |   0.975 |   0.716 |       21.8 |                     |
| keyword | Hybrid 0.3:0.7 | BAAI/bge-m3                                |   0.8   |   0.967 |   0.967 |   0.878 |       20.8 |                     |
| natural | Dense          | nlpai-lab/KURE-v1                          |   0.8   |   0.975 |   0.975 |   0.879 |       21.5 |                13.7 |
| keyword | Dense          | nlpai-lab/KURE-v1                          |   0.8   |   0.9   |   0.967 |   0.858 |       20.7 |                13.7 |
| natural | Hybrid 0.5:0.5 | nlpai-lab/KURE-v1                          |   0.5   |   0.9   |   0.95  |   0.678 |       22   |                     |
| keyword | Hybrid 0.5:0.5 | nlpai-lab/KURE-v1                          |   0.7   |   0.933 |   0.967 |   0.818 |       21.1 |                     |
| natural | Hybrid 0.3:0.7 | nlpai-lab/KURE-v1                          |   0.6   |   0.925 |   0.975 |   0.753 |       22   |                     |
| keyword | Hybrid 0.3:0.7 | nlpai-lab/KURE-v1                          |   0.733 |   0.967 |   1     |   0.847 |       21   |                     |
| natural | Dense          | intfloat/multilingual-e5-large             |   0.6   |   0.8   |   0.8   |   0.692 |       21.4 |                13.3 |
| keyword | Dense          | intfloat/multilingual-e5-large             |   0.767 |   0.9   |   0.933 |   0.834 |       20.4 |                13.3 |
| natural | Hybrid 0.5:0.5 | intfloat/multilingual-e5-large             |   0.525 |   0.675 |   0.775 |   0.614 |       22.2 |                     |
| keyword | Hybrid 0.5:0.5 | intfloat/multilingual-e5-large             |   0.7   |   0.9   |   0.933 |   0.803 |       21.5 |                     |
| natural | Hybrid 0.3:0.7 | intfloat/multilingual-e5-large             |   0.55  |   0.65  |   0.775 |   0.626 |       22.2 |                     |
| keyword | Hybrid 0.3:0.7 | intfloat/multilingual-e5-large             |   0.767 |   0.9   |   0.967 |   0.847 |       21.1 |                     |
| natural | Dense          | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.775 |   0.975 |   0.975 |   0.871 |       21.4 |                12.9 |
| keyword | Dense          | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.767 |   0.933 |   0.967 |   0.858 |       20.4 |                12.9 |
| natural | Hybrid 0.5:0.5 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.525 |   0.825 |   0.95  |   0.682 |       22   |                     |
| keyword | Hybrid 0.5:0.5 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.667 |   0.9   |   0.967 |   0.8   |       21   |                     |
| natural | Hybrid 0.3:0.7 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.6   |   0.9   |   0.975 |   0.748 |       22   |                     |
| keyword | Hybrid 0.3:0.7 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.767 |   0.933 |   1     |   0.859 |       21.1 |                     |
| natural | Dense          | Qwen/Qwen3-Embedding-0.6B                  |   0.675 |   0.875 |   0.95  |   0.784 |       39.7 |                22.2 |
| keyword | Dense          | Qwen/Qwen3-Embedding-0.6B                  |   0.733 |   0.933 |   0.933 |   0.822 |       33   |                22.2 |
| natural | Hybrid 0.5:0.5 | Qwen/Qwen3-Embedding-0.6B                  |   0.525 |   0.725 |   0.875 |   0.641 |       38.9 |                     |
| keyword | Hybrid 0.5:0.5 | Qwen/Qwen3-Embedding-0.6B                  |   0.7   |   0.9   |   0.9   |   0.794 |       33.5 |                     |
| natural | Hybrid 0.3:0.7 | Qwen/Qwen3-Embedding-0.6B                  |   0.525 |   0.775 |   0.95  |   0.68  |       39.2 |                     |
| keyword | Hybrid 0.3:0.7 | Qwen/Qwen3-Embedding-0.6B                  |   0.733 |   0.9   |   0.967 |   0.824 |       33.3 |                     |
| natural | Dense          | intfloat/multilingual-e5-small             |   0.475 |   0.575 |   0.6   |   0.531 |        5.6 |                 7.3 |
| keyword | Dense          | intfloat/multilingual-e5-small             |   0.533 |   0.7   |   0.733 |   0.619 |        5.4 |                 7.3 |
| natural | Hybrid 0.5:0.5 | intfloat/multilingual-e5-small             |   0.45  |   0.575 |   0.625 |   0.517 |        6.3 |                     |
| keyword | Hybrid 0.5:0.5 | intfloat/multilingual-e5-small             |   0.633 |   0.833 |   0.833 |   0.722 |        6.1 |                     |
| natural | Hybrid 0.3:0.7 | intfloat/multilingual-e5-small             |   0.425 |   0.575 |   0.625 |   0.503 |        6.3 |                     |
| keyword | Hybrid 0.3:0.7 | intfloat/multilingual-e5-small             |   0.567 |   0.733 |   0.733 |   0.644 |        5.8 |                     |
