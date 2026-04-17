import time
import json
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
# from transformers import AutoModelForCausalLM, AutoTokenizer (REPLACED BY OLLAMA)
from neo4j_graphrag.schema import get_schema
import sys
sys.path.append('CyVer')
from CyVer import SyntaxValidator, SchemaValidator, PropertiesValidator
import ollama
import json_repair
import requests

load_dotenv()

URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_SECRET") )
DB_NAME = "neo4j"

OLLAMA_MODEL_ID = "hf.co/mradermacher/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0-GGUF:Q8_0"
EMBED_MODEL = "mxbai-embed-large"

driver = GraphDatabase.driver(URI, auth=AUTH)

class GraphRAGPipeline:
    def __init__(self, log_indent=2):
        self.driver = driver
        self.log_indent = log_indent
        
        self.model_id = OLLAMA_MODEL_ID
        self.log("init", f"Using Ollama model {self.model_id}")
        
        self.log("init", "Initializing System Context...")
        self.cached_context = self.get_system_context()
        self.log("system context", self.cached_context) # Optional: reduce noise
        
        self.syntax_validator = SyntaxValidator(self.driver, check_multilabeled_nodes=False)
        self.schema_validator = SchemaValidator(self.driver)
        self.props_validator = PropertiesValidator(self.driver)

        # API Configuration
        self.api_key = os.getenv("API_KEY")
        self.api_url = "https://apollo.quocanmeomeo.io.vn/v1/chat/completions"
        self.api_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def log(self, stage, message, data=None):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage.upper(),
            "message": message,
            "data": data
        }
        print(json.dumps(log_entry, indent=self.log_indent, default=str))

    def get_system_context(self):
        """Retrieves the LIVE database state with rich metadata and property examples."""
        self.log("context", "Fetching rich schema from Neo4j...")
        
        # 1. Fetch labels and properties with examples
        node_props = []
        with self.driver.session(database=DB_NAME) as session:
            labels = session.run("CALL db.labels()").value()
            for label in labels:
                # Get a sample node to see properties and values
                sample_res = session.run(f"MATCH (n:{label}) RETURN n LIMIT 1").single()
                if not sample_res:
                    continue
                node = sample_res[0]
                
                label_lines = [f"- **{label}**"]
                for p_key, p_val in node.items():
                    p_type = type(p_val).__name__.upper()
                    # Sanitize example value
                    example = str(p_val).replace("\n", " ")
                    if len(example) > 100:
                        example = example[:97] + "..."
                    label_lines.append(f"  - `{p_key}`: {p_type} Example: \"{example}\"")
                node_props.append("\n".join(label_lines))

            # 2. Fetch relationships using path pattern matching
            rel_records = session.run("""
                MATCH (n)-[r]->(m) 
                RETURN DISTINCT labels(n)[0] AS source, type(r) AS type, labels(m)[0] AS target 
                LIMIT 50
            """).data()
            rels = [f"(:{r['source']})-[:{r['type']}]->(:{r['target']})" for r in rel_records]

            # 3. Fetch Vector Indexes for context
            vector_indexes = session.run("""
                SHOW INDEXES YIELD name, type, labelsOrTypes 
                WHERE type = 'VECTOR' 
                RETURN name, labelsOrTypes[0] AS label
            """).data()
            v_idx_info = [f"Vector Index: `{idx['name']}` on Label `{idx['label']}`" for idx in vector_indexes]

        # Assemble the final context string
        schema_context = "Node properties:\n" + "\n".join(node_props)
        schema_context += "\n\nThe relationships:\n" + "\n".join(rels)
        if v_idx_info:
            schema_context += "\n\nAvailable Vector Indexes:\n" + "\n".join(v_idx_info)
            schema_context += "\n(Use `$embedding` parameter with `db.index.vector.queryNodes` for similarity search)"

        return schema_context

    def get_embedding(self, text):
        """Generates embedding using the local Ollama instance."""
        try:
            res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            return res['embedding']
        except Exception as e:
            self.log("Embedding Error", f"Failed to generate embedding: {str(e)}")
            return None


    def generate_completion(self, messages, max_tokens=512):
        """Generates completion using Ollama."""
        try:
            res = ollama.chat(model=self.model_id, messages=messages)
            return res['message']['content'].strip()
        except Exception as e:
            self.log("Generation Error", f"Failed to call Ollama: {str(e)}")
            return ""

    def transform_to_vector_query(self, standard_cypher):
        """Transform standard Cypher to Cypher 25 Vector Search using external API."""
        self.log("transformation", "Applying AI semantic transformation...")

        # 1. Extract search term from original query for embedding purposes
        where_pattern = r"WHERE\s+(?:toLower\()?\s*(\w+)\.(\w+)\s*\)?\s*(?:=|(?:CONTAINS))\s*(['\"])(.*?)\3"
        where_match = re.search(where_pattern, standard_cypher, re.IGNORECASE)
        search_term = where_match.group(4) if where_match else None

        # 2. Prepare API call
        messages = [
            {
                "role": "system",
                "content": "You are a Cypher expert for Neo4j v2026. Transform legacy Cypher queries into Cypher 25 using the SEARCH sub-clause. Available vector indexes: Professional (professional_embeddings), Experience (experience_embeddings), Education (education_embeddings), Certification (certification_embeddings). Return ONLY the Cypher query and nothing else."
            },
            {
                "role": "user",
                "content": "Transform: MATCH (e:Experience)-[:ROLE_WAS]->(j:JobTitle {name: \"digital designer\"}) WHERE NOT EXISTS { MATCH (e)-[:HAS_EDUCATION]->(edu:Education)-[:AT_UNIVERSITY]->(u:University) RETURN edu } RETURN count(DISTINCT e) AS count"
            },
            {
                "role": "assistant",
                "content": "CYPHER 25\nMATCH (j:JobTitle)\nSEARCH j IN (VECTOR INDEX experience_embeddings FOR $emb_role LIMIT 100)\nSCORE AS score\nWHERE score > 0.8\nMATCH (e:Experience)-[:ROLE_WAS]->(j)\nWHERE NOT EXISTS {\n    MATCH (e)-[:HAS_EDUCATION]->(:Education)-[:AT_UNIVERSITY]->(:University)\n}\nRETURN count(DISTINCT e) AS count"
            },
            {
                "role": "user",
                "content": "Transform: MATCH (p:Professional)-[:HAS_EXPERIENCE]->(e:Experience)-[:ROLE_WAS]->(jt:JobTitle) WHERE jt.name = \"developer\" RETURN count(DISTINCT p)"
            },
            {
                "role": "assistant",
                "content": "CYPHER 25\nMATCH (jt:JobTitle)\nSEARCH jt IN (VECTOR INDEX experience_embeddings FOR $emb_role LIMIT 100)\nSCORE AS score\nWHERE score > 0.8\nMATCH (p:Professional)-[:HAS_EXPERIENCE]->(e:Experience)-[:ROLE_WAS]->(jt)\nRETURN count(DISTINCT p)"
            },
            {
                "role": "user",
                "content": "Transform: MATCH (p:Professional) WHERE p.headline CONTAINS \"Data Scientist\" RETURN p.name LIMIT 10"
            },
            {
                "role": "assistant",
                "content": "CYPHER 25\nMATCH (p:Professional)\nSEARCH p IN (VECTOR INDEX professional_embeddings FOR $emb_role LIMIT 100)\nSCORE AS score\nWHERE score > 0.8\nRETURN p.name LIMIT 10"
            },
            {
                "role": "user",
                "content": f"Transform: {standard_cypher}"
            }
        ]

        try:
            data = {
                "model": "qwen2.5-coder:14b",
                "messages": messages,
                "stream": False
            }
            response = requests.post(self.api_url, headers=self.api_headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            transformed_cypher = self.extract_cypher_only(result['choices'][0]['message']['content'])
            self.log("transformation", "AI Transformation successful.")
            return transformed_cypher, search_term
        except Exception as e:
            self.log("transformation error", f"AI Transformation failed: {str(e)}")
            return standard_cypher, None


    def generate_cypher_query(self, user_query, schema_context):
        USER_PROMPT_TEMPLATE="""Generate a standard Neo4j Cypher query for the Question below.
Use only the provided relationship types, node labels, and properties from the Schema section.

#### Schema:
{schema}
#### Question:
{question}"""
        prompt = USER_PROMPT_TEMPLATE.format(schema=schema_context, question=user_query)
        messages = [
            {"role": "user", "content": prompt}
        ]
        res = self.generate_completion(messages)
        return self.extract_cypher_only(res)

    def extract_cypher_only(self, res):
        """Extracts the Cypher query from the model response, handling markdown blocks."""
        cypher = res.strip()
        if "```" in cypher:
            match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", cypher, re.DOTALL)
            if match:
                cypher = match.group(1)
            else:
                cypher = re.sub(r"```cypher|```", "", cypher)
        
        # Clean up literal backslash-n sequences
        cypher = cypher.replace('\\n', '\n')
        return cypher.strip()


    def execute_query(self, cypher, params=None):
        params = params or {}
        with self.driver.session(database=DB_NAME) as session:
            try:
                t0 = time.time()
                result = session.run(cypher, **params)
                data = result.data()
                t1 = time.time()
                self.log("Neo4j Query", f"Executed in {t1 - t0:.2f}s | Rows: {len(data)}")
                return data
            except Exception as e:
                self.log("Neo4j Error", f"Query failed: {str(e)}")
                return [{"error": str(e)}]
        
    def run(self, user_query):
        context = self.cached_context
        max_retries = 3
        
        # Stage 1: Generate Standard Cypher using Gemma 3
        self.log("generation", f"Stage 1: Generating standard Cypher intent...")
        standard_cypher = self.generate_cypher_query(user_query, context)
        self.log("generation", f"Gemma Output: {standard_cypher}")

        # Stage 2: Transform to Vector Search Query using External API
        # We only transform if the user is asking for roles/skills (heuristic or always check)
        cypher_query, semantic_term = self.transform_to_vector_query(standard_cypher)

        # Fallback Logic: If Vector Cypher is same as standard (failed API), just proceed
        if cypher_query == standard_cypher:
            self.log("Fallback", "Vector Transformation not applied or failed. Using Standard Cypher.")
            semantic_term = None  # Disable embedding injection

        # Stage 4: Inject embedding parameter if needed
        params = {}
        if semantic_term and "$emb_role" in cypher_query:
            self.log("embedding", f"Generating vector for semantic term: {semantic_term}")
            vector = self.get_embedding(semantic_term)
            if vector:
                params["emb_role"] = vector
            else:
                self.log("embedding", "Proceeding without embedding due to generation failure.")
        elif "$emb_role" in cypher_query:
            self.log("embedding", f"Fallback: Generating vector for query: {user_query}")
            vector = self.get_embedding(user_query)
            if vector: params["emb_role"] = vector

        # Step 5: Final validation and execution
        try:
            schema_score, schema_meta = self.schema_validator.validate(cypher_query, database_name=DB_NAME)
            if schema_score != 1:
                 self.log("Schema Warning", f"Schema validation score: {schema_score}. Meta: {schema_meta}")
                
            props_score, props_meta = self.props_validator.validate(cypher_query, database_name=DB_NAME, strict=False)
            if props_score is not None and props_score != 1:
                self.log("Props Warning", f"Props validation score: {props_score}. Meta: {props_meta}")
        except Exception as e:
            self.log("Validation Cleanup", f"Validation skipped or failed gracefully: {str(e)}")
            
        final_data = self.execute_query(cypher_query, params)
        return {
            "user_query": user_query,
            "cypher_query": cypher_query,
            "final_data": final_data
        }
