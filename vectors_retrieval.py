import os

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

from my_llm import EMBEDDINGS

DB_DIR = 'faiss_db/'

def save_vectors_db():
    """构建向量数据库，并保存到磁盘"""
    if os.path.exists(DB_DIR):
        print('向量数据库已经构建，直接读取就OK！')
    else:
        with open('sales_datas.txt', encoding='utf8') as f:
            contents = f.read()
        # 把文本内容，切割成一个个，doc
        text_splitter = CharacterTextSplitter(
            separator=r'\d+\.\n',
            is_separator_regex=True,
            chunk_size=100,
            chunk_overlap=0,
            length_function=len
        )
        docs = text_splitter.create_documents([contents])
        print(len(docs))
        db = FAISS.from_documents(docs, EMBEDDINGS)
        db.save_local(DB_DIR)

        result = db.similarity_search('小区里面有绿化吗？')
        print(result)

def init_chain():
    """最终得到一个chain"""
    # 第一步: 加载向量数据库
    db = FAISS.load_local(DB_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)
    # 第二步: 创建一个提示模板
    # 创建一个问题的模板
    system_prompt = """你是一个问答任务的助手。
        使用以下检索到的上下文片段来回答问题这个问题。如果你不知道答案，就说:"这个问题，我建议你直接问人工！"。最多使用三句话，保持答案简洁。\n
        {context}
        """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 第三步: 创建一个chain
    # 创建一个搜索器: similarity_score_threshold: 根据相识度的分数来返回结果, 'score_threshold': 0.8: 分值 >= 0.7
    retriever = db.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.7})

    model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)
    chain1 = create_stuff_documents_chain(llm=model, prompt=prompt_template) # 将检索到的结果(多个docs)输入到提示模板中
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=chain1)



if __name__ == '__main__':


