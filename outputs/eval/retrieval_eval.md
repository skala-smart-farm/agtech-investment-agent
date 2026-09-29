질문 세트 {'natural': 40, 'keyword': 30} (정답 = 같은 문서·페이지), 조각 471개, 검색기마다 후보 k=8(런타임 candidate_k), 하이브리드는 합친 순서의 앞 8개. 시간 열(query_ms, load_or_build_sec)은 색인·임베딩 캐시가 있으면 0에 가깝다

## 종합 (두 세트 평균, Hit@4 → MRR@4 순)

| retriever      | embedding                                  |   Hit@1 |   Hit@3 |   Hit@4 |   Hit@5 |   Hit@8 |   MRR@4 |   MRR@5 |
|:---------------|:-------------------------------------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| Hybrid 0.3:0.7 | nlpai-lab/KURE-v1                          |   0.708 |   0.962 |   0.988 |   0.988 |   0.988 |   0.832 |   0.832 |
| Hybrid 0.3:0.7 | BAAI/bge-m3                                |   0.716 |   0.938 |   0.975 |   0.975 |   0.975 |   0.82  |   0.82  |
| Dense          | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.771 |   0.954 |   0.971 |   0.971 |   0.971 |   0.864 |   0.864 |
| Hybrid 0.5:0.5 | nlpai-lab/KURE-v1                          |   0.6   |   0.934 |   0.958 |   0.958 |   0.988 |   0.762 |   0.762 |
| Dense          | BAAI/bge-m3                                |   0.734 |   0.929 |   0.946 |   0.958 |   0.975 |   0.828 |   0.831 |
| Hybrid 0.3:0.7 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.708 |   0.929 |   0.942 |   0.971 |   0.971 |   0.814 |   0.82  |
| Dense          | nlpai-lab/KURE-v1                          |   0.8   |   0.938 |   0.938 |   0.971 |   0.988 |   0.861 |   0.868 |
| Hybrid 0.3:0.7 | Qwen/Qwen3-Embedding-0.6B                  |   0.629 |   0.875 |   0.934 |   0.958 |   0.958 |   0.756 |   0.76  |
| Hybrid 0.5:0.5 | BAAI/bge-m3                                |   0.587 |   0.879 |   0.934 |   0.946 |   0.988 |   0.736 |   0.739 |
| Dense          | Qwen/Qwen3-Embedding-0.6B                  |   0.704 |   0.904 |   0.929 |   0.942 |   0.958 |   0.8   |   0.803 |
| Hybrid 0.5:0.5 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.596 |   0.904 |   0.929 |   0.942 |   0.988 |   0.743 |   0.746 |
| Hybrid 0.5:0.5 | Qwen/Qwen3-Embedding-0.6B                  |   0.612 |   0.838 |   0.888 |   0.904 |   0.934 |   0.722 |   0.726 |
| Hybrid 0.3:0.7 | intfloat/multilingual-e5-large             |   0.659 |   0.788 |   0.854 |   0.884 |   0.908 |   0.736 |   0.741 |
| Dense          | intfloat/multilingual-e5-large             |   0.684 |   0.85  |   0.85  |   0.866 |   0.908 |   0.76  |   0.763 |
| Hybrid 0.5:0.5 | intfloat/multilingual-e5-large             |   0.612 |   0.788 |   0.842 |   0.854 |   0.896 |   0.709 |   0.712 |
| Hybrid 0.5:0.5 | intfloat/multilingual-e5-small             |   0.542 |   0.729 |   0.729 |   0.729 |   0.788 |   0.624 |   0.624 |
| Kiwi BM25      | -                                          |   0.6   |   0.675 |   0.708 |   0.721 |   0.721 |   0.641 |   0.644 |
| Dense          | intfloat/multilingual-e5-small             |   0.504 |   0.638 |   0.666 |   0.666 |   0.696 |   0.575 |   0.575 |
| Hybrid 0.3:0.7 | intfloat/multilingual-e5-small             |   0.496 |   0.654 |   0.666 |   0.666 |   0.696 |   0.571 |   0.571 |

## 세트별

| set     | retriever      | embedding                                  |   Hit@1 |   Hit@3 |   Hit@4 |   Hit@5 |   Hit@8 |   MRR@4 |   MRR@5 |   query_ms |   load_or_build_sec |
|:--------|:---------------|:-------------------------------------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|-----------:|--------------------:|
| natural | Kiwi BM25      | -                                          |   0.5   |   0.55  |   0.55  |   0.575 |   0.575 |   0.521 |   0.526 |        0.7 |                     |
| keyword | Kiwi BM25      | -                                          |   0.7   |   0.8   |   0.867 |   0.867 |   0.867 |   0.761 |   0.761 |        0.4 |                     |
| natural | Dense          | BAAI/bge-m3                                |   0.6   |   0.925 |   0.925 |   0.95  |   0.95  |   0.754 |   0.759 |        0.4 |                   0 |
| keyword | Dense          | BAAI/bge-m3                                |   0.867 |   0.933 |   0.967 |   0.967 |   1     |   0.903 |   0.903 |        0.4 |                   0 |
| natural | Hybrid 0.5:0.5 | BAAI/bge-m3                                |   0.475 |   0.825 |   0.9   |   0.925 |   0.975 |   0.648 |   0.653 |        0.9 |                     |
| keyword | Hybrid 0.5:0.5 | BAAI/bge-m3                                |   0.7   |   0.933 |   0.967 |   0.967 |   1     |   0.825 |   0.825 |        0.7 |                     |
| natural | Hybrid 0.3:0.7 | BAAI/bge-m3                                |   0.6   |   0.875 |   0.95  |   0.95  |   0.95  |   0.735 |   0.735 |        0.9 |                     |
| keyword | Hybrid 0.3:0.7 | BAAI/bge-m3                                |   0.833 |   1     |   1     |   1     |   1     |   0.906 |   0.906 |        0.7 |                     |
| natural | Dense          | nlpai-lab/KURE-v1                          |   0.8   |   0.975 |   0.975 |   0.975 |   0.975 |   0.879 |   0.879 |        0.3 |                   0 |
| keyword | Dense          | nlpai-lab/KURE-v1                          |   0.8   |   0.9   |   0.9   |   0.967 |   1     |   0.844 |   0.858 |        0.2 |                   0 |
| natural | Hybrid 0.5:0.5 | nlpai-lab/KURE-v1                          |   0.5   |   0.9   |   0.95  |   0.95  |   0.975 |   0.696 |   0.696 |        1   |                     |
| keyword | Hybrid 0.5:0.5 | nlpai-lab/KURE-v1                          |   0.7   |   0.967 |   0.967 |   0.967 |   1     |   0.828 |   0.828 |        0.7 |                     |
| natural | Hybrid 0.3:0.7 | nlpai-lab/KURE-v1                          |   0.65  |   0.925 |   0.975 |   0.975 |   0.975 |   0.787 |   0.787 |        0.9 |                     |
| keyword | Hybrid 0.3:0.7 | nlpai-lab/KURE-v1                          |   0.767 |   1     |   1     |   1     |   1     |   0.878 |   0.878 |        0.7 |                     |
| natural | Dense          | intfloat/multilingual-e5-large             |   0.6   |   0.8   |   0.8   |   0.8   |   0.85  |   0.692 |   0.692 |        0.3 |                   0 |
| keyword | Dense          | intfloat/multilingual-e5-large             |   0.767 |   0.9   |   0.9   |   0.933 |   0.967 |   0.828 |   0.834 |        0.2 |                   0 |
| natural | Hybrid 0.5:0.5 | intfloat/multilingual-e5-large             |   0.525 |   0.675 |   0.75  |   0.775 |   0.825 |   0.615 |   0.62  |        0.9 |                     |
| keyword | Hybrid 0.5:0.5 | intfloat/multilingual-e5-large             |   0.7   |   0.9   |   0.933 |   0.933 |   0.967 |   0.803 |   0.803 |        0.7 |                     |
| natural | Hybrid 0.3:0.7 | intfloat/multilingual-e5-large             |   0.55  |   0.675 |   0.775 |   0.8   |   0.85  |   0.629 |   0.634 |        1.9 |                     |
| keyword | Hybrid 0.3:0.7 | intfloat/multilingual-e5-large             |   0.767 |   0.9   |   0.933 |   0.967 |   0.967 |   0.842 |   0.848 |        0.7 |                     |
| natural | Dense          | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.775 |   0.975 |   0.975 |   0.975 |   0.975 |   0.871 |   0.871 |        0.3 |                   0 |
| keyword | Dense          | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.767 |   0.933 |   0.967 |   0.967 |   0.967 |   0.858 |   0.858 |        0.3 |                   0 |
| natural | Hybrid 0.5:0.5 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.525 |   0.875 |   0.925 |   0.95  |   0.975 |   0.692 |   0.697 |        1   |                     |
| keyword | Hybrid 0.5:0.5 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.667 |   0.933 |   0.933 |   0.933 |   1     |   0.794 |   0.794 |        0.7 |                     |
| natural | Hybrid 0.3:0.7 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.65  |   0.925 |   0.95  |   0.975 |   0.975 |   0.777 |   0.782 |        0.9 |                     |
| keyword | Hybrid 0.3:0.7 | dragonkue/snowflake-arctic-embed-l-v2.0-ko |   0.767 |   0.933 |   0.933 |   0.967 |   0.967 |   0.85  |   0.857 |        0.7 |                     |
| natural | Dense          | Qwen/Qwen3-Embedding-0.6B                  |   0.675 |   0.875 |   0.925 |   0.95  |   0.95  |   0.779 |   0.784 |        0.3 |                   0 |
| keyword | Dense          | Qwen/Qwen3-Embedding-0.6B                  |   0.733 |   0.933 |   0.933 |   0.933 |   0.967 |   0.822 |   0.822 |        0.2 |                   0 |
| natural | Hybrid 0.5:0.5 | Qwen/Qwen3-Embedding-0.6B                  |   0.525 |   0.775 |   0.875 |   0.875 |   0.9   |   0.65  |   0.65  |        1   |                     |
| keyword | Hybrid 0.5:0.5 | Qwen/Qwen3-Embedding-0.6B                  |   0.7   |   0.9   |   0.9   |   0.933 |   0.967 |   0.794 |   0.801 |        0.7 |                     |
| natural | Hybrid 0.3:0.7 | Qwen/Qwen3-Embedding-0.6B                  |   0.525 |   0.85  |   0.9   |   0.95  |   0.95  |   0.683 |   0.693 |        0.9 |                     |
| keyword | Hybrid 0.3:0.7 | Qwen/Qwen3-Embedding-0.6B                  |   0.733 |   0.9   |   0.967 |   0.967 |   0.967 |   0.828 |   0.828 |        0.6 |                     |
| natural | Dense          | intfloat/multilingual-e5-small             |   0.475 |   0.575 |   0.6   |   0.6   |   0.625 |   0.531 |   0.531 |        0.2 |                   0 |
| keyword | Dense          | intfloat/multilingual-e5-small             |   0.533 |   0.7   |   0.733 |   0.733 |   0.767 |   0.619 |   0.619 |        0.1 |                   0 |
| natural | Hybrid 0.5:0.5 | intfloat/multilingual-e5-small             |   0.45  |   0.625 |   0.625 |   0.625 |   0.675 |   0.521 |   0.521 |        0.8 |                     |
| keyword | Hybrid 0.5:0.5 | intfloat/multilingual-e5-small             |   0.633 |   0.833 |   0.833 |   0.833 |   0.9   |   0.728 |   0.728 |        0.5 |                     |
| natural | Hybrid 0.3:0.7 | intfloat/multilingual-e5-small             |   0.425 |   0.575 |   0.6   |   0.6   |   0.625 |   0.498 |   0.498 |        0.8 |                     |
| keyword | Hybrid 0.3:0.7 | intfloat/multilingual-e5-small             |   0.567 |   0.733 |   0.733 |   0.733 |   0.767 |   0.644 |   0.644 |        0.5 |                     |
