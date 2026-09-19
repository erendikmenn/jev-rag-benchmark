from app import query_engine


QUESTIONS = [
    "What is the capital of Türkiye?",
    "Which city spans Europe and Asia?",
    "Why is Mars called the Red Planet?",
]

for question in QUESTIONS:
    response = query_engine.query(question)
    sources = [node.node.metadata.get("file_name") for node in response.source_nodes]
    print({"question": question, "response": str(response), "sources": sources})
