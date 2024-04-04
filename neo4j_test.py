from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

llm = LlamaCpp(
    model_path=r"G:\Meu Drive\Modelos\llama-2-7b-chat.Q3_K_M.gguf",
    n_gpu_layers=20,
    max_tokens=256,
    n_ctx=2048,
    temperature=0.1,
    top_p=1,
    verbose=True,
    )

graph = Neo4jGraph(
    url="neo4j+s://0df231b2.databases.neo4j.io", username="neo4j", password="5T5C0frCeJimyvRRiU5QOZmC-OKb9DAZsGQoGXOEVIw"
)

graph.refresh_schema()

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions: 
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# Quantas pessoas atuaram em Top Gun?
MATCH (m:Filme {{title:"Top Gun"}})<-[:ATUOU_EM]-()
RETURN count(*) AS numeroDeAtores

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

print(graph.schema)

chain = GraphCypherQAChain.from_llm(
    llm, graph=graph, verbose=True, return_intermediate_steps=True, cypher_prompt=CYPHER_GENERATION_PROMPT, validate_cypher=True,
)

#result = chain.run({"query": "Quais são os pratos do Restaurante Hamburgueria do Futuro?"})
result = chain.run({"schema": graph.schema, "question": "Quais são os pratos do Restaurante Hamburgueria do Futuro?", "query": "Quais são os pratos do Restaurante Hamburgueria do Futuro?"})
print(f"Intermediate steps: {result['intermediate_steps']}")
print(f"Final answer: {result['result']}")