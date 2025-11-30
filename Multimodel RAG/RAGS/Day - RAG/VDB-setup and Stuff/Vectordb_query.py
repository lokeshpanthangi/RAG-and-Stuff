from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_unstructured import UnstructuredLoader
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


file_paths = [
    "./example_data/layout-parser-paper.pdf",
    "./example_data/state_of_the_union.txt",
]


loader = UnstructuredLoader(file_paths)
docs = loader.load()
docs[0]