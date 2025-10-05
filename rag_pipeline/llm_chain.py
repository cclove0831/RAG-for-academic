from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from .config import LLM_TIMEOUT
from .retrieval import fmt_docs

def build_chain(two_stage_fn):
    prompt = PromptTemplate.from_template("""
Based solely on the most relevant paper, summarize the main contribution in 2-3 sentences, followed by 2-3 key takeaways.Answer ONLY using the provided context. If not in the context, say you don't know.
Cite evidence like [1], [2] matching the context blocks.

Context:
{context}

Question: {question}

Answer (concise, with citations):
""")
    llm = ChatTongyi(model="qwen-turbo", temperature=0, streaming=False, request_timeout=LLM_TIMEOUT)

    def rewrite_query(q: str) -> str:
        ql = q.lower()
        if any(k in ql for k in ["contribution","main contribution","summary","abstract","摘要","贡献","结论"]):
            return (q + " abstract introduction contributions conclusion results discussion").strip()
        return q

    chain = (
        {"context": RunnableLambda(lambda q: two_stage_fn(q)) | RunnableLambda(fmt_docs),
         "question": RunnablePassthrough() | RunnableLambda(rewrite_query)}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
