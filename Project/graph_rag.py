import time
import json
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from neo4j_graphrag.schema import get_schema
import sys
sys.path.append('CyVer')
from CyVer import SyntaxValidator, SchemaValidator, PropertiesValidator
import ollama

load_dotenv()

URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_SECRET") )
DB_NAME = "neo4j"

HF_MODEL_ID = "neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0"
EMBED_MODEL = "mxbai-embed-large"

driver = GraphDatabase.driver(URI, auth=AUTH)

class GraphRAGPipeline:
    def __init__(self, log_indent=2):
        self.driver = driver
        self.log_indent = log_indent
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.log("init", "Initializing System Context...")
        self.cached_context = self.get_system_context()
        self.log("system context", self.cached_context)
        
        self.syntax_validator = SyntaxValidator(self.driver, check_multilabeled_nodes=False)
        self.schema_validator = SchemaValidator(self.driver)
        self.props_validator = PropertiesValidator(self.driver)
        
        self.log("init", f"Loading Hugging Face model {HF_MODEL_ID} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        
        if self.device == "cuda":
            from transformers import BitsAndBytesConfig
            # Use 4-bit quantization (NF4) to save VRAM while preserving accuracy
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Fallback for CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)

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
        """Generic text generation helper to replace Ollama."""
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        input_len = inputs['input_ids'].shape[-1]
        response_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return response_text.strip()

    def generate_cypher_query(self, user_query, schema_context):
        USER_PROMPT_TEMPLATE="""Generate a Cypher query for the Question below.
Use the information about the nodes, relationships, and properties from the Schema section below to generate the best possible Cypher query.

Respond ONLY with a JSON object in the following format:
{{
  "cypher": "The generated Cypher query",
  "embed_text": "If the query uses a vector index via $embedding, provide the specific word or phrase to be vectorized here. Otherwise, null."
}}

####Schema:
{schema}
####Question:
{question}"""
        prompt = USER_PROMPT_TEMPLATE.format(schema=schema_context, question=user_query)
        messages = [
            {"role": "user", "content": prompt}
        ]
        res = self.generate_completion(messages)
        
        # Robust extraction for structured output
        cypher = ""
        embed_text = None
        
        try:
            # Try to find JSON block first
            if "{" in res and "}" in res:
                match = re.search(r"(\{.*?\})", res, re.DOTALL)
                if match:
                    data = json.loads(match.group(1))
                    cypher = data.get("cypher", "")
                    embed_text = data.get("embed_text")
            
            # Fallback if the model didn't return valid JSON but returned raw Cypher
            if not cypher:
                cypher = res
                if "```" in res:
                    match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", res, re.DOTALL)
                    if match:
                        cypher = match.group(1)
                    else:
                        cypher = re.sub(r"```cypher|```", "", res)
        except Exception as e:
            self.log("Extraction Error", f"Failed to parse model response: {str(e)}. Using raw output.")
            cypher = res

        # Clean up literal backslash-n sequences
        cypher = cypher.replace('\\n', '\n')
        
        return cypher.strip(), embed_text

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
        cypher_query, embed_text = self.generate_cypher_query(user_query, context)
        
        params = {}
        if embed_text:
            self.log("embedding", f"Generating vector for: {embed_text}")
            vector = self.get_embedding(embed_text)
            if vector:
                params["embedding"] = vector
            else:
                self.log("embedding", "Proceeding without embedding due to generation failure.")

        is_valid, syntax_meta = self.syntax_validator.validate(cypher_query, database_name=DB_NAME)
        if not is_valid:
            return {
                "user_query": user_query,
                "cypher_query": cypher_query,
                "final_data": [{"error": f"CyVer Syntax Error: {syntax_meta}"}]
            }
            
        # ... rest of validation logic ...
        # (Using minimal parameters to avoid conflicts with CyVer which sometimes validates without params)
        extracted_node_labels, extracted_rel_labels, extracted_paths = self.schema_validator.extract(cypher_query, database_name=DB_NAME)
        schema_score, schema_meta = self.schema_validator.validate(cypher_query, database_name=DB_NAME)
        if schema_score != 1:
             return {
                "user_query": user_query,
                "cypher_query": cypher_query,
                "final_data": [{"error": f"CyVer Schema Validation Error: {schema_meta}"}]
            }
            
        variables_properties, labels_properties = self.props_validator.extract(cypher_query, strict=False, database_name=DB_NAME)
        props_score, props_meta = self.props_validator.validate(cypher_query, database_name=DB_NAME, strict=False)
        if props_score is not None and props_score != 1:
            return {
                "user_query": user_query,
                "cypher_query": cypher_query,
                "final_data": [{"error": f"CyVer Properties Validation Error: {props_meta}"}]
            }
            
        final_data = self.execute_query(cypher_query, params)
        return {
            "user_query": user_query,
            "cypher_query": cypher_query,
            "final_data": final_data
        }
