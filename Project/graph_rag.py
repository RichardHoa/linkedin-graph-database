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
import json_repair

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

    def transform_to_vector_query(self, standard_cypher):
        """Uses Qwen 14B to transform standard Cypher into a vector search query."""
        self.log("transformation", f"Sending query to Qwen 14B for semantic transformation...")
        
        prompt = f"""You are a Cypher translation expert. Convert the following standard Cypher query into a Neo4j 5.x vector search query.
        
        ### Standard Cypher: 
        {standard_cypher}
        
        ### Rules:
        1. Target Vector Index: 'experience_embeddings'
        2. Boilerplate: CALL db.index.vector.queryNodes('experience_embeddings', 100000, $embedding) YIELD node, score WHERE score > 0.8
        3. Use the variable 'node' as the semantic anchor for subsequent MATCH clauses.
        4. Extract the core role or skill being filtered in the original query as the 'embedding_term'.
        
        Respond ONLY with a JSON object:
        {{
            "transformed_cypher": "...",
            "embedding_term": "..."
        }}
        """
        
        try:
            res = ollama.generate(model="qwen2.5-coder:14b", prompt=prompt, format="json", stream=False)
            data = json.loads(res['response'])
            transformed = data.get("transformed_cypher", standard_cypher)
            term = data.get("embedding_term", "")
            self.log("transformation", f"Transformed to Vector Cypher. Term: {term}")
            return transformed, term
        except Exception as e:
            self.log("Transformation Error", f"Fallback to standard Cypher: {str(e)}")
            return standard_cypher, None

    def generate_cypher_query(self, user_query, schema_context):
        USER_PROMPT_TEMPLATE="""Generate a standard Neo4j Cypher query for the Question below.
Use only the provided relationship types, node labels, and properties from the Schema section.

Respond ONLY with the Cypher query. No explanation. No additional text.

#### Examples:
Question: "How many Python developers are there?"
Cypher: MATCH (p:Professional {{role: 'Python developer'}}) RETURN count(p)

Question: "Who has the linkedin id '123'?"
Cypher: MATCH (p:Professional {{linkedin_id: '123'}}) RETURN p.name

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

    def fix_cypher_syntax(self, bad_cypher, error_msg, schema_context):
        """Asks the model to fix a syntactically incorrect Cypher query."""
        self.log("Correction", f"Sending error feedback to LLM...")
        prompt = f"""The following Cypher query generated for the database schema below is syntactically INCORRECT.
#### Schema:
{schema_context}

#### Faulty Query:
{bad_cypher}

#### Error Message:
{error_msg}

Please fix the syntax error and return ONLY the corrected Cypher query. No explanation. No JSON."""
        
        messages = [{"role": "user", "content": prompt}]
        res = self.generate_completion(messages)
        return self.extract_cypher_only(res)

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

        # Stage 2: Transform to Vector Search Query using Qwen 14B
        # We only transform if the user is asking for roles/skills (heuristic or always check)
        cypher_query, semantic_term = self.transform_to_vector_query(standard_cypher)

        # Stage 3: Self-correction loop for syntax errors
        for attempt in range(max_retries):
            self.log("Validation", f"Attempt {attempt + 1}: Validating Transformation...")
            is_valid, syntax_meta = self.syntax_validator.validate(cypher_query, database_name=DB_NAME)
            
            if is_valid:
                break
            
            if attempt < max_retries - 1:
                self.log("Correction", f"Syntax error detected: {syntax_meta}. Retrying...")
                cypher_query = self.fix_cypher_syntax(cypher_query, str(syntax_meta), context)
            else:
                return {
                    "user_query": user_query,
                    "cypher_query": cypher_query,
                    "final_data": [{"error": f"CyVer Syntax Error after {max_retries} attempts: {syntax_meta}"}]
                }

        # Stage 4: Inject embedding parameter if needed
        params = {}
        if semantic_term and "$embedding" in cypher_query:
            self.log("embedding", f"Generating vector for semantic term: {semantic_term}")
            vector = self.get_embedding(semantic_term)
            if vector:
                params["embedding"] = vector
            else:
                self.log("embedding", "Proceeding without embedding due to generation failure.")
        elif "$embedding" in cypher_query:
            self.log("embedding", f"Fallback: Generating vector for query: {user_query}")
            vector = self.get_embedding(user_query)
            if vector: params["embedding"] = vector

        # Step 5: Final validation and execution
        schema_score, schema_meta = self.schema_validator.validate(cypher_query, database_name=DB_NAME)
        if schema_score != 1:
             self.log("Schema Warning", f"Schema validation score: {schema_score}. Meta: {schema_meta}")
            
        props_score, props_meta = self.props_validator.validate(cypher_query, database_name=DB_NAME, strict=False)
        if props_score is not None and props_score != 1:
            self.log("Props Warning", f"Props validation score: {props_score}. Meta: {props_meta}")
            
        final_data = self.execute_query(cypher_query, params)
        return {
            "user_query": user_query,
            "cypher_query": cypher_query,
            "final_data": final_data
        }
