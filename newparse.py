import json

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
embedder = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

def md_to_json(md_file_path):
    """Convert markdown file to structured JSON format"""
    sections = []
    current_section = {"main title": "", "section title": "", "content": []}
    main_title = ""
    
    with open(md_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            if current_section["section title"]:
                sections.append(current_section)
            main_title = line[2:].strip()
            current_section = {"main title": main_title, "section title": "", "content": []}
        elif line.startswith("## "):
            if current_section["section title"]:
                sections.append(current_section)
            current_section = {"main title": main_title, "section title": line[3:].strip(), "content": []}
        else:
            current_section["content"].append(line)
    
    if current_section["section title"]:
        sections.append(current_section)
    
    for section in sections:
        section["content"] = "\n".join(section["content"]).strip()
    
    return sections

def create_or_load_collection():
    """Check if collection exists; if not, create it"""
    collection_name = "md_embeddings"
    connections.connect("default", host="localhost", port="19530")

    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Loading it.")
        return Collection(name=collection_name)  # Load existing collection
    
    print(f"Creating new collection: {collection_name}")

    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="main_title_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="section_title_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="content_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ], description="Embeddings collection for markdown files")

    collection = Collection(name=collection_name, schema=schema)
    return collection

def create_indexes(collection):
    """Create indexes for the collection fields in Milvus."""
    print(f"Creating indexes for collection '{collection.name}'...")
    
    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index("main_title_embedding", index_params)
    collection.create_index("section_title_embedding", index_params)
    collection.create_index("content_embedding", index_params)
    
    print(f"Indexes created for collection '{collection.name}'.")

def generate_embeddings(text):
    """Generate embeddings for the given text"""
    return embedder.encode(text).tolist() if text else [0.0] * 1024

def insert_data_into_milvus(collection, json_data):
    """Insert data into Milvus collection with generated embeddings"""
    main_title_embeddings = []
    section_title_embeddings = []
    content_embeddings = []
    texts = []
    
    for data in json_data:
        main_title_embeddings.append(generate_embeddings(data["main title"]))
        section_title_embeddings.append(generate_embeddings(data["section title"]))
        content_embeddings.append(generate_embeddings(data["content"]))
        texts.append(data["content"])
    
    entities = [main_title_embeddings, section_title_embeddings, content_embeddings, texts]
    
    collection.insert(entities)
    collection.flush()
    print("Data inserted successfully.")

def query(query_text, anns_field="section_title_embedding", limit=1, threshold=0.90):
    """Query the Milvus collection with a given text and filter results based on similarity threshold"""
    
    collection_name = "md_embeddings"
    connections.connect("default", host="localhost", port="19530")

    collection = Collection(name=collection_name)

    collection.load()

    query_embedding = generate_embeddings(query_text)

    search_params = {"metric_type": "IP", "params": {"ef": 128}}

    results = collection.search(
        data=[query_embedding],
        anns_field=anns_field,
        param=search_params,
        limit=limit,
        output_fields=["text"]
    )

    filtered_result = []
    for res in results:
        for hit in res:
            if hit.distance >= threshold:
                filtered_result.append({
                    "text": hit.entity.get("text"),
                    "similarity": hit.distance
                })

    return filtered_result

def main():
    md_file_path = "md_output/output.md"  # Replace with actual file path
    json_data = md_to_json(md_file_path)

    with open("cnn1.json", "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4)
    
    collection = create_or_load_collection()
    
    # CREATE INDEXES AFTER COLLECTION CREATION
    create_indexes(collection)
    
    insert_data_into_milvus(collection, json_data)

    query_text = "Introduction"
    result = query(query_text)
    print(result)

if __name__ == "__main__":
    main()
