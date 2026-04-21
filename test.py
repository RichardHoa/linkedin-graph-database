import requests
import time
import json

url = "https://apollo.quocanmeomeo.io.vn/v1/chat/completions"
headers = {
    "Authorization": "Bearer FUV_danghoa_4lLx6kx1oWMn6H86Px0J",
    "Content-Type": "json"
}

data = {
    "model": "llama2:7b-chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a Cypher expert for Neo4j v2026. Transform legacy Cypher queries into Cypher 25 using the SEARCH sub-clause. Available vector indexes: Professional (professional_embeddings), Experience (experience_embeddings), Education (education_embeddings), Certification (certification_embeddings)."
      },
      {
        "role": "user",
        "content": "Transform: MATCH (e:Experience)-[:ROLE_WAS]->(j:JobTitle {name: \"digital designer\"}) WHERE NOT EXISTS { MATCH (e)-[:HAS_EDUCATION]->(edu:Education)-[:AT_UNIVERSITY]->(u:University) RETURN edu } RETURN count(DISTINCT e) AS count"
      },
      {
        "role": "assistant",
        "content": "CYPHER 25\nMATCH (j:JobTitle)\nSEARCH j IN (VECTOR INDEX job_title_embeddings FOR $emb_role LIMIT 50)\nSCORE AS score\nWHERE score > 0.8\nMATCH (e:Experience)-[:ROLE_WAS]->(j)\nWHERE NOT EXISTS {\n    MATCH (e)-[:HAS_EDUCATION]->(:Education)-[:AT_UNIVERSITY]->(:University)\n}\nRETURN count(DISTINCT e) AS count"
      },
      {
        "role": "user",
        "content": "Transform: MATCH (p:Professional)-[:HAS_EXPERIENCE]->(e:Experience)-[:ROLE_WAS]->(jt:JobTitle) WHERE jt.name = \"developer\" RETURN count(DISTINCT p)"
      }
    ],
    "stream": False
}

# Start timer
start_time = time.time()

response = requests.post(url, headers=headers, json=data)
result = response.json()

# End timer
end_time = time.time()
duration = end_time - start_time

# Beautiful Output
print("-" * 30)
print(f"⏱️  Response Time: {duration:.2f} seconds")
print("-" * 30)
print("🤖 AI Response:")
print(result['choices'][0]['message']['content'])
print("-" * 30)