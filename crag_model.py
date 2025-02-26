from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
import gzip
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing import List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from uuid import uuid4
from dotenv import load_dotenv
import os
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import  RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langgraph.graph import END, StateGraph
from openai import OpenAI
import markdown


# Load environment variables from the .env file
load_dotenv()

# Access your variables
langchain_key = os.getenv("LANGCHAIN_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_key)


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]

#embedding model
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-large')

wikipedia_filepath = 'F:/Lucio/Descargas/simplewiki-2020-11-01.jsonl.gz'
docs = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        #Add documents
        docs.append({
                        'metadata': {
                                        'title': data.get('title'),
                                        'article_id': data.get('id')
                        },
                        'data': ' '.join(data.get('paragraphs')[0:3]) 
        # restrict data to first 3 paragraphs to run later modules faster
        })
# We subset our data to use a subset of wikipedia documents to run things faster
docs = [doc for doc in docs for x in ['india']
              if x in doc['data'].lower().split()]
# Create docs
docs = [Document(page_content=doc['data'],
                 metadata=doc['metadata']) for doc in docs]
# Chunk docs
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunked_docs = splitter.split_documents(docs)

#index to initialize vector database
index = faiss.IndexFlatL2(len(openai_embed_model.embed_query("hello world")))

#FAISS vector database
vector_store = FAISS(
    embedding_function=openai_embed_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

#document IDs to alocate random tags to nuew documents added to the database
uuids = [str(uuid4()) for _ in range(len(chunked_docs))]


#initial documents as a reference
_ = vector_store.add_documents(documents=chunked_docs, ids=uuids)

# Data model for LLM output format
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
# LLM for grading 
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
# Prompt template for grading
SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}
                     User question:
                     {question}
                  """),
    ]
)
# Build grader chain
doc_grader = (grade_prompt
                  |
              structured_llm_grader)


# Create RAG prompt for response generation
prompt = """You are an polite and kind assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question.
            Question:
            {question}
            Context:
            {context}
            Answer:
         """
prompt_template = ChatPromptTemplate.from_template(prompt)
# Initialize connection with GPT-4o
chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=1)
# Used for separating context docs with new lines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# create QA RAG chain
qa_rag_chain = (
    {
        "context": (itemgetter('context')
                        |
                    RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
      |
    prompt_template
      |
    chatgpt
      |
    StrOutputParser()
)

# LLM for question rewriting
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Prompt template for rewriting
SYS_PROMPT = """Act as a question re-writer and perform the following task:
                 - Convert the following input question to a better version that is optimized for web search.
                 - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
             """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Here is the initial question:
                     {question}
                     Formulate an improved question.
                  """,
        ),
    ]
)

# Create rephraser chain
question_rewriter = (re_write_prompt
                        |
                       llm
                        |
                     StrOutputParser())

#retrieval method to consider relevant documents
similarity_threshold_retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                       search_kwargs={"k": 3,                                                                       
                       "score_threshold": 0.3})

#Tavily search engine to get new docuemnets in case of retrieval strategies don't find a suitable context for the answe
tv_search = TavilySearchResults(max_results=10, search_depth='advanced',max_tokens=10000)

#retrieval function
def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents - that contains retrieved context documents
    """
    question = state["question"]
    # Retrieval
    documents = similarity_threshold_retriever.invoke(question)
    return {"documents": documents, "question": question}

#function to assign relevance score to new docuemnts
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.
    If any document are not relevant to question or documents are empty - Web Search needs to be done
    If all documents are relevant to question - Web Search is not needed
    Helps filtering out irrelevant documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    question = state["question"]
    documents = state["documents"]
    # Score each doc
    filtered_docs = []
    web_search_needed = "No"
    if documents:
        for d in documents:
            score = doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                filtered_docs.append(d)
            else:
                web_search_needed = "Yes"
                continue
    else:
        web_search_needed = "Yes"
    return {"documents": filtered_docs, "question": question, 
            "web_search_needed": web_search_needed}
    
#block to take user queries and improve them to generate better answers  
def rewrite_query(state):
    """
    Rewrite the query to produce a better question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates question key with a re-phrased or re-written question
    """
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

#optional block to search for new documents to compelment the answer, and add possible docuements to the vector dataase
def web_search(state):
    """
    Web search based on the re-written question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with appended web results
    """
    question = state["question"]
    documents = state["documents"]
    
    # Web search
    docs = tv_search.invoke(question)
    web_results_text = "\n\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results_text)
    
    # Chunk the Document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300
    )
    # Pass a list containing the Document to be split
    chunked_docs = splitter.split_documents([web_results])
    
    # Generate new UUIDs for each chunk
    uuids = [str(uuid4()) for _ in range(len(chunked_docs))]
    
    # Add the chunks to the vector store
    _ = vector_store.add_documents(documents=chunked_docs, ids=uuids)
    
    # Instead of appending a list, extend the documents list with individual chunks
    documents.extend(chunked_docs)
    
    return {"documents": documents, "question": question}

#function to read the final context and generate the suitable answer
def generate_answer(state):
    """
    Generate answer from context document using LLM
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
  
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, 
            "generation": generation}
    
#evaluatior function to define when to answer or modify queries and search for new information
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    web_search_needed = state["web_search_needed"]
    if web_search_needed == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        return "rewrite_query"
    else:
        # We have relevant documents, so generate answer
        return "generate_answer"
    
    
#langgraph structure to build agent chain
agentic_rag = StateGraph(GraphState)
# Define the nodes
agentic_rag.add_node("retrieve", retrieve)  # retrieve
agentic_rag.add_node("grade_documents", grade_documents)  # grade documents
agentic_rag.add_node("rewrite_query", rewrite_query)  # transform_query
agentic_rag.add_node("web_search", web_search)  # web search
agentic_rag.add_node("generate_answer", generate_answer)  # generate answer
# Build graph
agentic_rag.set_entry_point("retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")
agentic_rag.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
)
agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_answer")
agentic_rag.add_edge("generate_answer", END)
# Compile
agentic_rag = agentic_rag.compile()


#function to invoke agent model


def agent(query: str) -> str:
    response = agentic_rag.invoke({"question": query})
    # Convert Markdown text to HTML using the markdown package
    html_output = markdown.markdown(response['generation'], extensions=['md_in_html'])
    return html_output



#moderator function in case of inapropriate words
def moderate_response( question: str) -> bool:
        response =client.moderations.create(
        model="omni-moderation-latest",
        input=question,
        )
        return response.results[0].flagged