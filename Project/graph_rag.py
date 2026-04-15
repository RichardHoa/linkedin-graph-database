import time
import json
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama

load_dotenv()

URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_SECRET") )
DB_NAME = "neo4j"

LLM_MODEL = "qwen2.5-coder:14b"
HF_MODEL_ID = "neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0"

driver = GraphDatabase.driver(URI, auth=AUTH)

class GraphRAGPipeline:
    def __init__(self, log_indent=2):
        self.driver = driver
        self.log_indent = log_indent
        self.log("init", "Initializing System Context...")
        self.cached_context = self.get_system_context()
        
        self.log("init", f"Loading Hugging Face model {HF_MODEL_ID} on CUDA...")
        
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
        """
        Retrieves the LIVE database state with grouped schema, 
        truncated property examples, and vector index samples.
        """
        with self.driver.session() as session:
            # 1. Grouped Graph Structure
            schema_query = """
            CALL apoc.meta.graph() YIELD nodes, relationships
            UNWIND relationships AS rel
            WITH startNode(rel).name AS source, type(rel) AS rel_type, endNode(rel).name AS target
            RETURN source, rel_type, target ORDER BY source
            """
            schema_data = session.run(schema_query).data()
            formatted_schema = [f"(:{r['source']})-[:{r['rel_type']}]->(:{r['target']})" for r in schema_data]
    
            # 2. Properties with Examples
            props_query = """
            CALL apoc.meta.nodeTypeProperties() 
            YIELD nodeLabels, propertyName, propertyTypes 
            WITH nodeLabels[0] AS label, propertyName, propertyTypes[0] AS type
            WHERE NOT propertyName IN ['embedding', 'embedding_summary']
            ORDER BY label, propertyName
            
            // Find a non-null sample for each specific property
            CALL (label, propertyName) {
                MATCH (n) 
                WHERE label IN labels(n) AND n[propertyName] IS NOT NULL
                RETURN n[propertyName] AS sample_value 
                LIMIT 1
            }
            
            RETURN label, propertyName, type, sample_value
            """
            props_data = session.run(props_query).data()
            
            props_dict = {}
            for p in props_data:
                label = p['label']
                if label not in props_dict:
                    props_dict[label] = []
                
                val = p['sample_value']
                # Truncate strings to 50 chars, otherwise stringify
                if isinstance(val, str):
                    example = f"'{val[:50]}...'" if len(val) > 50 else f"'{val}'"
                else:
                    example = str(val)
                    
                props_dict[label].append(f"\n  - {p['propertyName']} ({p['type']}): {example}")
    
            # 3. Vector Indexes with 'embedding_summary' samples
            index_query = """
            SHOW INDEXES YIELD name, type, labelsOrTypes 
            WHERE type = 'VECTOR' 
            RETURN name, labelsOrTypes[0] AS label
            """
            index_records = session.run(index_query).data()
            
            formatted_indexes = []
            for idx in index_records:
                label = idx['label']
                # Corrected: IS NOT NULL to find a real sample
                summary_sample = session.run(
                    f"MATCH (n:{label}) WHERE n.embedding_summary IS NOT NULL RETURN n.embedding_summary LIMIT 1"
                ).single()
                
                sample_text = summary_sample[0] if summary_sample else "No summary available"
                formatted_indexes.append(
                    f"Index: {idx['name']} (Label: {label})\n"
                    f"   -> Sample embedding_summary: \"{sample_text[:200]}...\""
                )
    
        # Building the Final Markdown String
        context = ["### LIVE DATABASE SCHEMA (LLM CONTEXT GUIDE)"]
        context.append("\n**1. GRAPH STRUCTURE (Grouped by Source):**")
        context.extend(list(dict.fromkeys(formatted_schema))) # Remove duplicates and keep order
        
        context.append("\n**2. NODE PROPERTIES & EXAMPLES:**")
        for label, properties in props_dict.items():
            context.append(f"- **{label}**: {'; '.join(properties)}")
            
        context.append("\n**3. VECTOR INDEXES (Use for semantic search):**")
        context.extend(formatted_indexes)
    
        return "\n".join(context)

    def generate_cypher_query(self, user_query, schema_context):
        """Generate Cypher using the given huggingface model directly."""
        prompt = f"""Given the following Neo4j Graph Schema:
{schema_context}

Generate a valid Cypher query to answer the following user question:
{user_query}
"""
        messages = [
            {"role": "system", "content": "You are a Neo4j Cypher expert. Convert the user's natural language question into a Cypher query using the provided schema."},
            {"role": "user", "content": prompt}
        ]
        
        self.log("Cypher Generation", f"Generating query using {HF_MODEL_ID}...")
        
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False
        )
        
        # Decode only the generated response
        response_text = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Clean formatting
        clean_cypher = re.sub(r"```cypher|```", "", response_text).strip()
        
        self.log("Cypher Generation", "Generated Cypher", data=clean_cypher)
        return clean_cypher

    def execute_query(self, cypher, params=None):
        params = params or {}
        print(cypher)
        with self.driver.session() as session:
            try:
                t0 = time.time()
                result = session.run(cypher, **params)
                data = result.data()
                t1 = time.time()

                self.log(
                    "Neo4j Query",
                    f"Executed in {t1 - t0:.2f}s  |  Rows returned: {len(data)}"
                )

                return data
            except Exception as e:
                self.log("Neo4j Error", f"Query failed: {str(e)}\n\nCypher:\n{cypher}")
                return [f"Cypher Error: {str(e)}"]

    def generate_final_answer(self, user_query, db_data):
        prompt = f"""
        Answer the User's Question based STRICTLY and ONLY on the 'Retrieved Data'.
        
        User Question: "{user_query}"
        Retrieved Data: {json.dumps(db_data, indent=2)}
        
        If data is empty or indicates an error, state that you could not find the information. Be direct and objective.
        """
        res = ollama.generate(model=LLM_MODEL, prompt=prompt)
        return res['response']
        
    def run(self, user_query):
        print(f"\n--- Processing: {user_query} ---")
        
        context = self.cached_context
        
        # ── STAGE 1: Cypher Generation ───────────────────────────────────────────
        t0 = time.time()
        cypher_query = self.generate_cypher_query(user_query, context)
        t1 = time.time()
        self.log(
            "Timing",
            f"[1/2] generate_cypher_query  |  model: {HF_MODEL_ID}  |  {t1 - t0:.2f}s"
        )
        
        if not cypher_query:
            return {
                "user_query": user_query,
                "final_data": None,
                "error": "Failed to generate Cypher query."
            }

        # ── STAGE 2: Query Execution ───────────────────────────────────────────
        t0 = time.time()
        final_data = self.execute_query(cypher_query, {})
        t1 = time.time()
        self.log(
            "Timing",
            f"[2/2] execute_query  |  {t1 - t0:.2f}s"
        )

        return {
            "user_query": user_query,
            "final_data": final_data
        }
