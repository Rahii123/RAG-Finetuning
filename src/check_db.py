import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)

collection = client.get_collection("clinical_guidelines")

print("Total documents in collection:", collection.count())
