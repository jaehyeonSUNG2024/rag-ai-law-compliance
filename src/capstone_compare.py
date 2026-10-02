"""AI기본법 컴플라이언스 체커 - Baseline과 단계별 개선 Pipeline 비교.

1. Baseline       : 글자 수 Chunk + Similarity
2. +조단위Chunk    : 조(條)/FAQ 단위로 자르고 출처 라벨을 본문에 붙임
3. +ParentDoc     : 항·호 단위 작은 Chunk(+조 머리글)로 검색하고, 반환은 조 전체 (긴 조문 희석 해결)
4. +Hybrid        : ParentDoc + BM25 (EnsembleRetriever)
5. +Rewrite·Route : 사업 용어 → 법률 용어 변환 + 복합 질문 분해(RRF 병합) + "제N조" 직접 조회
6. Final          : Hybrid + "제N조" 직접 조회 (Rewrite 제외: MRR은 올리지만 Recall@4를 떨어뜨림)
"""

import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

K = 4

# TODO 1 (14.2) 문서 범위: 법률 + 시행령 + 사업자 FAQ
SOURCES = {
    "법": ("AI기본법", "data/law.txt"),
    "령": ("AI기본법 시행령", "data/decree.txt"),
    "가이드": ("사업자 FAQ", "data/guideline.txt"),
}
HEADER = re.compile(r"^(?:(제\d+조)\(.+?\)|(Q\d+)\.)", re.M)


def headers(prefix: str, text: str) -> list[tuple[int, str]]:
    """문서 안 모든 조/FAQ 머리글의 (시작 위치, 라벨) 목록."""
    return [(m.start(), f"{prefix} {m.group(1) or m.group(2)}") for m in HEADER.finditer(text)]


def labels_of(heads, start: int, end: int) -> list[str]:
    """[start, end) 구간이 걸쳐 있는 조항 라벨. 평가 시 정답 조항 판정에 사용."""
    inside = [label for pos, label in heads if start < pos < end]
    before = [label for pos, label in heads if pos <= start]
    return before[-1:] + inside


texts = {prefix: Path(path).read_text(encoding="utf-8") for prefix, (_, path) in SOURCES.items()}

# TODO 2 (14.3) Baseline: 글자 수 기준 Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30, add_start_index=True)
baseline_chunks = []
for prefix, text in texts.items():
    heads = headers(prefix, text)
    for doc in splitter.create_documents([text]):
        start = doc.metadata["start_index"]
        doc.metadata["labels"] = labels_of(heads, start, start + len(doc.page_content))
        baseline_chunks.append(doc)

# 개선: 조(條)/FAQ 단위 Chunk, 본문 앞에 "[AI기본법 제34조]" 같은 출처를 붙여 검색에도 쓰이게 함
article_chunks = []
for prefix, text in texts.items():
    heads = headers(prefix, text)
    ends = [pos for pos, _ in heads[1:]] + [len(text)]
    for (start, label), end in zip(heads, ends):
        name = SOURCES[prefix][0]
        body = text[start:end].strip()
        article_chunks.append(Document(page_content=f"[{name}] {body}", metadata={"labels": [label]}))

print(f"Baseline Chunk 수: {len(baseline_chunks)} / 조 단위 Chunk 수: {len(article_chunks)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
baseline = FAISS.from_documents(baseline_chunks, embeddings).as_retriever(search_kwargs={"k": K})
dense = FAISS.from_documents(article_chunks, embeddings).as_retriever(search_kwargs={"k": K})


def char_bigrams(text: str) -> list[str]:
    # ponytail: 형태소 분석 없이 글자 2-gram. "제34조는"도 "제34조"와 매칭됨. 품질 부족하면 kiwipiepy로 교체.
    s = re.sub(r"\s+", "", text)
    return [s[i : i + 2] for i in range(len(s) - 1)]


# ParentDocument: 항·호 단위 child Chunk로 검색 → 해당 조 전체(parent)를 반환.
# child마다 "[AI기본법] 제2조(정의)" 머리글을 붙여, 잘린 조각도 어느 조의 무슨 내용인지 검색되게 함 (Contextual Header)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
children = []
for parent in article_chunks:
    head = parent.page_content.split("\n", 1)[0]
    for child in child_splitter.split_documents([parent]):
        if not child.page_content.startswith(head):
            child.page_content = f"{head} {child.page_content}"
        children.append(child)
child_store = FAISS.from_documents(children, embeddings)
parent_by_label = {doc.metadata["labels"][0]: doc for doc in article_chunks}


def parent_search(question: str) -> list[Document]:
    hits = child_store.similarity_search(question, k=12)
    labels = dict.fromkeys(hit.metadata["labels"][0] for hit in hits)
    return [parent_by_label[label] for label in labels]


parent_doc = RunnableLambda(parent_search)

bm25 = BM25Retriever.from_documents(article_chunks, preprocess_func=char_bigrams, k=K)
hybrid = EnsembleRetriever(retrievers=[parent_doc, bm25], weights=[0.5, 0.5])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@lru_cache
def rewrite(question: str) -> tuple[str, ...]:
    """사업 용어를 법률 용어로 바꾸고, 여러 주제를 묻는 질문은 하위 질의로 나눔."""
    prompt = f"""
당신은 AI기본법(인공지능 발전과 신뢰 기반 조성 등에 관한 기본법) 검색 질의 작성기입니다.
사용자 질문을 법령 검색에 적합한 질의로 바꾸세요.
- 일상·사업 용어를 법률 용어로 바꿉니다. (예: 채용 솔루션 → 채용 판단에 활용되는 고영향 인공지능, 벌금 → 과태료, 그림 생성 AI → 생성형 인공지능 결과물 표시, 해외 회사 → 국내에 주소 또는 영업소가 없는 인공지능사업자)
- 기본은 한 줄입니다. 질문이 "A와 B"처럼 명시적으로 서로 다른 주제를 함께 물을 때만 주제별로 한 줄씩 나눕니다. 최대 3줄.
- 질문에 없는 주제(사례, 절차, 저작권 등)를 추가하지 않습니다.
- 설명, 번호 없이 질의만 출력합니다.

[사용자 질문]
{question}
"""
    lines = [line.strip("-• ").strip() for line in llm.invoke(prompt).content.splitlines()]
    return (question, *[line for line in lines if line][:3])


def rrf(doc_lists, c: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion: 여러 질의에서 공통으로 상위에 나온 문서를 앞으로."""
    scores, docs = {}, {}
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (c + rank)
            docs[key] = doc
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def route(question: str) -> list[Document]:
    """질문에 "제N조"가 있으면 AI기본법의 해당 조를 직접 조회 (메타데이터 라우팅)."""
    wanted = {f"법 {n}" for n in re.findall(r"제\d+조", question)}
    return [doc for doc in article_chunks if doc.metadata["labels"][0] in wanted]


def dedup(docs) -> list[Document]:
    return list({doc.page_content: doc for doc in docs}.values())  # 첫 등장 순서 유지


def rewrite_route(question: str) -> list[Document]:
    return dedup(route(question) + rrf(hybrid.invoke(q) for q in rewrite(question)))[:K]


def final(question: str) -> list[Document]:
    return dedup(route(question) + hybrid.invoke(question))[:K]


PIPELINES = {
    "1.Baseline": lambda q: baseline.invoke(q),
    "2.+조단위Chunk": lambda q: dense.invoke(q),
    "3.+ParentDoc": lambda q: parent_doc.invoke(q)[:K],
    "4.+Hybrid": lambda q: hybrid.invoke(q)[:K],
    "5.+Rewrite·Route": rewrite_route,
    "6.Final": final,
}

# TODO 3 (14.4) 고정 테스트 질문셋: expected = 정답 근거 조항 라벨 (None = 문서에 없는 질문)
test_cases = [
    {"type": "명확", "question": "고영향 인공지능이 정확히 뭐야?", "expected": ["법 제2조"]},
    {"type": "키워드", "question": "제34조는 무슨 내용이야?", "expected": ["법 제34조"]},
    {"type": "표현차이", "question": "이력서를 자동으로 걸러주는 채용 솔루션도 규제 대상이야?", "expected": ["령 제2조"]},
    {"type": "표현차이", "question": "그림 그려주는 AI 서비스인데 결과물에 뭘 해야 돼?", "expected": ["법 제31조", "령 제3조"]},
    {"type": "표현차이", "question": "한국에 지사가 없는 미국 회사인데 한국에 사람을 둬야 해?", "expected": ["법 제36조", "령 제5조"]},
    {"type": "교차참조", "question": "병원 진단 보조 AI를 출시하려면 뭘 준비해야 해?", "expected": ["령 제2조", "법 제34조"]},
    {"type": "복합", "question": "생성형 AI 챗봇의 고지 의무와 위반 시 과태료를 알려줘", "expected": ["법 제31조", "법 제43조"]},
    {"type": "표현차이", "question": "초거대 모델을 직접 학습시키는 회사는 어떤 안전 의무가 있어?", "expected": ["법 제32조", "령 제4조"]},
    {"type": "표현차이", "question": "법 시행되자마자 바로 벌금 맞아?", "expected": ["령 제7조", "가이드 Q5"]},
    {"type": "문서없음", "question": "미국 AI 규제와 비교하면 어떤 점이 달라?", "expected": None},
]


# TODO 4~6 (14.5~14.8) 동일 질문셋으로 Retrieval 평가
def score(docs, expected) -> tuple[float, float]:
    """(Recall@K: 정답 조항 중 찾은 비율, RR: 첫 정답 조항 순위의 역수)"""
    found = {label for doc in docs for label in doc.metadata["labels"]}
    recall = len(set(expected) & found) / len(expected)
    rr = next((1 / i for i, doc in enumerate(docs, 1) if set(expected) & set(doc.metadata["labels"])), 0.0)
    return recall, rr


summary = {name: [] for name in PIPELINES}
rows = []
for case in test_cases:
    print(f"\n### [{case['type']}] {case['question']}  (정답: {case['expected']})")
    row = [case["question"]]
    for name, run in PIPELINES.items():
        docs = run(case["question"])
        top = " | ".join(",".join(doc.metadata["labels"]) for doc in docs)
        if case["expected"] is None:
            print(f"  {name:15s} 평가 제외  Top-{K}: {top}")
            row.append("-")
            continue
        recall, rr = score(docs, case["expected"])
        summary[name].append((recall, rr))
        print(f"  {name:15s} Recall={recall:.2f} RR={rr:.2f}  Top-{K}: {top}")
        row.append(f"{recall:.2f}")
    rows.append(row)

print("\n=== Retrieval 비교 (문서에 없는 질문 제외) ===")
print("| Pipeline | Recall@%d | MRR | 정답 조항 전부 찾은 질문 |" % K)
print("|---|---|---|---|")
for name, scores in summary.items():
    recall = sum(r for r, _ in scores) / len(scores)
    mrr = sum(rr for _, rr in scores) / len(scores)
    full = sum(r == 1 for r, _ in scores)
    print(f"| {name} | {recall:.3f} | {mrr:.3f} | {full}/{len(scores)} |")

print("\n=== 질문별 Recall ===")
print("| 질문 | " + " | ".join(PIPELINES) + " |")
print("|---|" + "---|" * len(PIPELINES))
for row in rows:
    print("| " + " | ".join(row) + " |")

# TODO 6 (14.8) Generation: 최종 Pipeline으로 근거 조항을 인용해 답변
ANSWER_PROMPT = """
당신은 AI기본법 컴플라이언스 안내 도우미입니다.
아래 [문서]만 근거로 답하고, 각 문장 끝에 근거를 [AI기본법 제○조], [AI기본법 시행령 제○조], [사업자 FAQ Q○] 중 실제 출처에 맞는 형식으로 표시하세요.
해당 여부는 단정하지 말고 "해당할 가능성이 높습니다/낮습니다"처럼 표현하세요.
문서에서 근거를 찾을 수 없으면 "제공된 문서에서 확인할 수 없습니다."라고만 답하세요.
마지막 줄에 "※ 본 답변은 법률 자문이 아니며, 최종 판단은 전문가 확인이 필요합니다."를 붙이세요.

[문서]
{context}

[질문]
{question}
"""


def answer(question: str) -> str:
    docs = final(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return llm.invoke(ANSWER_PROMPT.format(context=context, question=question)).content


for q in [test_cases[2]["question"], test_cases[6]["question"], test_cases[9]["question"]]:
    print(f"\n=== 최종 답변: {q} ===")
    print(answer(q))
