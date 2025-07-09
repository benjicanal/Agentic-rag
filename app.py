from rag_chain import get_rag_chain

rag = get_rag_chain()

while True :
    question = input("Ask a question: ")
    
    if question.lower() in ["exit","quit"] :
        break
    result = rag({"query": question})
    print("Answer:", result["result"])
    print("Sources:")
    for doc in result["source_documents"]:
        print("→", doc.metadata.get("source"))