# Python base imports - Default ones
import os

# Dependent software imports
import asyncio
from lightrag.utils import setup_logger
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Custom created imports

setup_logger("lightrag", level = "DEBUG")

WORKING_DIR = "./rag_storage"
os.makedirs(WORKING_DIR, exist_ok = True)  # Improved: exist_ok avoids nested dir issues

TEXT_TO_INSERT = [
    "Artificial Intelligence (AI) is one of the most transformative technologies of the 21st century. It refers to the development of computer systems capable of performing tasks that normally require human intelligence. These tasks include learning, reasoning, problem solving, perception, and language understanding. AI is used in many industries such as healthcare, finance, education, manufacturing, and transportation.",
    "Machine Learning is a subset of Artificial Intelligence that allows computers to learn patterns from data without being explicitly programmed. Instead of following rigid instructions, machine learning models improve their performance by analyzing large datasets. Supervised learning, unsupervised learning, and reinforcement learning are the three main categories of machine learning.",
    "Supervised learning uses labeled datasets to train algorithms that classify data or predict outcomes accurately. Common supervised learning algorithms include linear regression, logistic regression, decision trees, and neural networks. For example, supervised learning can be used to detect spam emails by training a model on labeled email datasets.",
    "Unsupervised learning works with data that does not have labeled outputs. The algorithm tries to identify hidden patterns or groupings within the data. Clustering algorithms such as K-Means and hierarchical clustering are commonly used in unsupervised learning. Businesses often use clustering to segment customers based on purchasing behavior.",
    "Reinforcement learning is inspired by behavioral psychology and involves an agent that learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and gradually learns an optimal strategy. Reinforcement learning has been used successfully in robotics, gaming, and autonomous driving systems.",
    "Deep Learning is a specialized area of machine learning that uses neural networks with many layers. These networks are inspired by the structure of the human brain. Deep learning models are especially powerful in tasks such as image recognition, speech recognition, and natural language processing.",
    "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand and generate human language. NLP techniques are used in applications such as chatbots, virtual assistants, translation systems, and sentiment analysis tools.",
    "Large Language Models are advanced AI systems trained on massive datasets of text. They are capable of understanding context, generating coherent responses, summarizing information, and assisting with programming tasks. Many modern AI assistants are powered by large language models.",
    "Retrieval Augmented Generation (RAG) is a technique that improves the accuracy of language models by combining them with external knowledge retrieval systems. Instead of relying only on training data, a RAG system retrieves relevant documents and provides them as context to the language model.",
    "Knowledge Graphs are structured representations of knowledge where entities are connected through relationships. They allow machines to reason about information more effectively. In a knowledge graph, nodes represent entities while edges represent relationships between them.",
    "Graph-based Retrieval Augmented Generation systems combine traditional document retrieval with knowledge graph reasoning. These systems allow more precise answers because they can understand relationships between entities and concepts.",
    "Modern RAG frameworks often combine vector search with graph-based reasoning. Vector search helps retrieve semantically similar documents while knowledge graphs help maintain structured relationships between concepts.",
    "Applications of RAG systems include enterprise knowledge assistants, document search engines, research assistants, customer support bots, and AI-powered analytics platforms.",
    "Despite its advantages, RAG also faces challenges such as document chunking strategies, embedding quality, retrieval accuracy, and hallucination reduction. Researchers continue to develop new techniques to improve RAG pipelines.",
    "The future of AI will likely involve systems that combine reasoning, knowledge graphs, retrieval systems, and large language models. These hybrid systems aim to provide more reliable and explainable artificial intelligence solutions.",
]


# async def tracked_llm_call(*args, **kwargs):
#     response = await gpt_4o_mini_complete(*args, **kwargs)

#     if hasattr(response, "usage"):
#         usage = response.usage
#         print("\n---- TOKEN USAGE ----")
#         print(f"Input tokens: {usage.prompt_tokens}")
#         print(f"Output tokens: {usage.completion_tokens}")
#         print(f"Total tokens: {usage.total_tokens}")
#         print("---------------------\n")

    # return response


async def initialize_rag():
    rag = LightRAG(working_dir = WORKING_DIR, embedding_func = openai_embed, llm_model_func = gpt_4o_mini_complete)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag




async def main():
    rag = None
    try:
        rag = await initialize_rag()

        # for text in TEXT_TO_INSERT:
        #     await rag.ainsert(text)

        # modes = ["naive", "local", "global", "hybrid"]
        modes = ["hybrid"]

        for mode in modes:
            result = await rag.aquery("How do machine learning, deep learning, and natural language processing relate to artificial intelligence?", param = QueryParam(mode = mode))
            print(f"\n{mode.upper()} mode: {result}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
