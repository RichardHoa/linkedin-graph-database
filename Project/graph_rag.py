import time
import json
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_SECRET") )
DB_NAME = "neo4j"

HF_MODEL_ID = "neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0"

driver = GraphDatabase.driver(URI, auth=AUTH)

class GraphRAGPipeline:
    def __init__(self, log_indent=2):
        self.driver = driver
        self.log_indent = log_indent
        
        # FIX: Initialize device first to avoid AttributeError
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.log("init", "Initializing System Context...")
        self.cached_context = self.get_system_context()
        
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
        """Retrieves the LIVE database state."""
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
                if isinstance(val, str):
                    example = f"'{val[:50]}...'" if len(val) > 50 else f"'{val}'"
                else:
                    example = str(val)
                props_dict[label].append(f"\n  - {p['propertyName']} ({p['type']}): {example}")
    
            # 3. Vector Indexes
            index_query = """
            SHOW INDEXES YIELD name, type, labelsOrTypes 
            WHERE type = 'VECTOR' 
            RETURN name, labelsOrTypes[0] AS label
            """
            index_records = session.run(index_query).data()
            
            formatted_indexes = []
            for idx in index_records:
                label = idx['label']
                summary_sample = session.run(
                    f"MATCH (n:{label}) WHERE n.embedding_summary IS NOT NULL RETURN n.embedding_summary LIMIT 1"
                ).single()
                sample_text = summary_sample[0] if summary_sample else "No summary available"
                formatted_indexes.append(
                    f"Index: {idx['name']} (Label: {label})\n"
                    f"   -> Sample embedding_summary: \"{sample_text[:200]}...\""
                )
    
        context = ["### LIVE DATABASE SCHEMA (LLM CONTEXT GUIDE)"]
        context.append("\n**1. GRAPH STRUCTURE (Grouped by Source):**")
        context.extend(list(dict.fromkeys(formatted_schema)))
        context.append("\n**2. NODE PROPERTIES & EXAMPLES:**")
        for label, properties in props_dict.items():
            context.append(f"- **{label}**: {'; '.join(properties)}")
        context.append("\n**3. VECTOR INDEXES:**")
        context.extend(formatted_indexes)
        return "\n".join(context)

    def generate_completion(self, messages, max_tokens=512):
        """Generic text generation helper to replace Ollama."""
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
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
        prompt = f"Given the following Neo4j Graph Schema:\n{schema_context}\n\nGenerate a valid Cypher query to answer the user question:\n{user_query}"
        messages = [
            {"role": "system", "content": "You are a Neo4j Cypher expert. Convert the user's natural language question into a Cypher query using the provided schema. Return ONLY the raw cypher code."},
            {"role": "user", "content": prompt}
        ]
        res = self.generate_completion(messages)
        return re.sub(r"```cypher|```", "", res).strip()

    def execute_query(self, cypher, params=None):
        params = params or {}
        with self.driver.session() as session:
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
        cypher_query = self.generate_cypher_query(user_query, context)
        final_data = self.execute_query(cypher_query, {})
        return {
            "user_query": user_query,
            "final_data": final_data
        }
