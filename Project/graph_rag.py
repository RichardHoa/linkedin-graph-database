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
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.log("init", "Initializing System Context...")
        self.cached_context = self.get_system_context()
        
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
        """Retrieves the LIVE database state."""
        return get_schema(self.driver, is_enhanced=False, database=DB_NAME, sanitize=False)

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
Return only the Cypher query as your final output, without any additional text or explanation.
####Schema:
{schema}
####Question:
{question}"""
        prompt = USER_PROMPT_TEMPLATE.format(schema=schema_context, question=user_query)
        messages = [
            {"role": "user", "content": prompt}
        ]
        res = self.generate_completion(messages)
        
        # Robust extraction for Cypher queries inside markdown blocks
        cypher = res
        if "```" in res:
            match = re.search(r"```(?:cypher)?\s*(.*?)\s*```", res, re.DOTALL)
            if match:
                cypher = match.group(1)
            else:
                cypher = re.sub(r"```cypher|```", "", res)
        
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
        cypher_query = self.generate_cypher_query(user_query, context)
        
        is_valid, syntax_meta = self.syntax_validator.validate(cypher_query, database_name=DB_NAME)
        if not is_valid:
            return {
                "user_query": user_query,
                "cypher_query": cypher_query,
                "final_data": [{"error": f"CyVer Syntax Error: {syntax_meta}"}]
            }
            
        extracted_node_labels, extracted_rel_labels, extracted_paths = self.schema_validator.extract(cypher_query, database_name=DB_NAME)
        self.log("Validation", f"Schema extracted nodes: {extracted_node_labels}, rels: {extracted_rel_labels}")
        schema_score, schema_meta = self.schema_validator.validate(cypher_query, database_name=DB_NAME)
        if schema_score != 1:
            return {
                "user_query": user_query,
                "cypher_query": cypher_query,
                "final_data": [{"error": f"CyVer Schema Validation Error: {schema_meta}"}]
            }
            
        variables_properties, labels_properties = self.props_validator.extract(cypher_query, strict=False, database_name=DB_NAME)
        self.log("Validation", f"Props extracted variables: {variables_properties}, labels: {labels_properties}")
        props_score, props_meta = self.props_validator.validate(cypher_query, database_name=DB_NAME, strict=False)
        if props_score is not None and props_score != 1:
            return {
                "user_query": user_query,
                "cypher_query": cypher_query,
                "final_data": [{"error": f"CyVer Properties Validation Error: {props_meta}"}]
            }
            
        final_data = self.execute_query(cypher_query, {})
        return {
            "user_query": user_query,
            "cypher_query": cypher_query,
            "final_data": final_data
        }
