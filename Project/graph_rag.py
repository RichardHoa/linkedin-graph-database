# Example complete graph pipeline
import ollama
import time
import json
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_SECRET") )
DB_NAME = "neo4j"

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "qwen2.5-coder:14b"

driver = GraphDatabase.driver(URI, auth=AUTH)

class GraphRAGPipeline:
    def __init__(self):
        self.driver = driver
        print("Initializing System Context...")
        self.cached_context = self.get_system_context()

    def log(self, stage, message, data=None):
        print(f"\n{'='*20} [DEBUG: {stage.upper()}] {'='*20}")
        print(message)
        if data:
            print(f"\n--- DATA ({stage}) ---")
            if isinstance(data, (dict, list)):
                print(json.dumps(data, indent=2, default=str))
            else:
                print(data)
        print("="*60)

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
        
    def get_embedding(self, text):
        res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return res['embedding']

    def plan_execution(self, user_query, schema_context):
        """Step 1: Ask the AI to identify what needs embedding and the overall strategy."""
        prompt = f"""
        Role: You are the 'Strategic Planner' in a GraphRAG pipeline. Your task is to analyze the user question against the provided Neo4j schema to determine the execution strategy.

        ### 1. LIVE DATABASE CONTEXT:
        {schema_context}
        
        ### 2. USER QUESTION: 
        "{user_query}"
        
        ### 3. CLASSIFICATION TASK:
        Assign exactly one 'query_type' based on these definitions:
        - "stats": Requests for counts, lists, or simple aggregations existing within the schema.
        - "multi_step_analysis": Complex requests requiring path traversals, comparisons, or multi-node correlations.
        - "out_of_scope": Use this if the question cannot be answered using the provided Schema or Vector Indexes. You must carefully validate if the entities or relationships requested exist in the Context. If the data is missing, select this type.

        ### 4. ENTITY EXTRACTION:
        Identify all conceptual entities requiring semantic (vector) search. Map them to the correct 'embedding_name' found in the Vector Indexes section of the context.

        ### 5. OUTPUT REQUIREMENTS:
        - Provide a clear 'reasoning' for your classification. If 'out_of_scope', explicitly state what data is missing from the schema.
        - Return ONLY valid JSON.

        EXPECTED JSON FORMAT:
        {{
            "query_type": "stats" | "multi_step_analysis" | "out_of_scope",
            "reasoning": "Explain why this is stats vs analysis.",
            "embeddings_needed": [
                {{
                    "variable_name": "Unique variable name (e.g., emb_role)", 
                    "search_text": "Conceptual search term (e.g., 'Software Developer')", 
                    "embedding_name": "// Read the schema and carefully select the matching embedding index name"
                }},
                {{
                    "variable_name": "Next unique variable (e.g., emb_cert)", 
                    "search_text": "Next search term (e.g., 'Cloud Architecture')", 
                    "embedding_name": "// Select the matching embedding index name for this specific entity"
                }}
                // Add as many additional embedding objects as needed to satisfy the query.
            ]
        }}
        """
        self.log("Planning", "Analyzing intent and extracting entities...")
        
        res = ollama.generate(model=LLM_MODEL, prompt=prompt)
        
        try:
            # Clean potential markdown from LLM output
            clean_json = re.sub(r"```json|```", "", res['response']).strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            self.log("Error", "Failed to parse JSON plan.", res['response'])
            return {"query_type": "fallback", "embeddings_needed": []}

    def generate_cypher_query(self, user_query, schema_context, plan):
        """Step 3: Generate Cypher using the dynamically created embedding variables."""
        query_type = plan.get("query_type")
        
        available_vars = "\n".join([
            f"- Parameter: ${item['variable_name']} | Target Embedding: '{item['embedding_name']}' | Search: '{item['search_text']}'" 
            for item in plan.get('embeddings_needed', [])
        ])

        if query_type == 'stats':
            prompt = f"""
        Role: Neo4j Cypher Expert. Translate the user query into a high-performance Cypher query using the Schema and Vector Parameters provided.

        ### 1. CONTEXT:
        SCHEMA: {schema_context}
        PARAMETERS: {available_vars}

        ### 2. STRICT SYNTAX RULES:
        - **Vector Search**: Use `CALL db.index.vector.queryNodes(target_embedding, 100000, $parameter_name) YIELD node, score`.
        - **Parameters**: Use the `$` prefix for all provided vector variables (e.g., $emb_role). Never hardcode strings in the vector call.
        - **Quality**: Apply `WHERE score > 0.8` unless the query implies a broader search.
        - **Efficiency**: Use only the necessary nodes and relationships. For simple counts, return `count(node)`.
        - **Regex**: Use `~ '(?i)...'` for case-insensitive string matching on non-vector properties.

        ### 3. REFERENCE EXAMPLE:
        Question: "Python experts at Google"
        Query: 
        CALL db.index.vector.queryNodes('experience_embeddings', 100000, $emb_role) YIELD node AS p, score 
        WHERE score > 0.8
        MATCH (p)-[:AT_COMPANY]->(c:Company) WHERE c.name =~ '(?i)google.*'
        RETURN p.name

        ### 4. TASK:
        User Question: "{user_query}"
        
        RETURN RAW CYPHER ONLY. NO MARKDOWN. NO EXPLANATION.
        """
        elif query_type == 'multi_step_analysis':
            prompt = f"""
            Role: Neo4j Graph Data Scientist & Strategic Researcher.
            
            ### PIPELINE CONTEXT:
            You are the 'Extraction Agent' in a 3-stage GraphRAG pipeline:
            1. Planner (Current Stage): You break the user's question into Cypher tasks.
            2. Executor: A Python runner executes these tasks against Neo4j.
            3. Synthesizer (Final Stage): A Final LLM reads the results of all tasks to answer the user.
            
            ### OBJECTIVE:
            Generate a sequence of Cypher queries that provide the Synthesizer with a clear, evidentiary data trail to answer: "{user_query}"

            ### SYSTEM CONSTRAINTS:
            1. **Live Schema**: {schema_context}
            2. **Parameters**: {available_vars}
            3. **Vector Syntax**: `CALL db.index.vector.queryNodes(target_embedding, top_k, $param) YIELD node, score WHERE score > 0.8`.
            4. **No Hallucinations**: Only use labels/properties provided in the schema context.
            
            ### STRICT VECTOR LIMIT & TRAVERSAL RULES:
            1. **Path-Walking/ID-Extraction**: You MUST set `top_k` to `1000`. 
            2. **Stats/Aggregation**: You MUST set `top_k` to `100000`. Never use arbitrary numbers like 50.

            ### CRITICAL CYPHER SYNTAX RULES:
            1. **ID Extraction**: ALWAYS use `elementId(node)`. NEVER use the deprecated `id(node)` function.
            2. **Single Column Return Rule**: Intermediate steps used to pass IDs to subsequent steps MUST return EXACTLY ONE column (`RETURN elementId(node) AS id`). Returning multiple columns will break the Python pipeline's list flattening logic.
            3. **Consuming Previous IDs**: Use `$step1_ids`, `$step2_ids`, etc., dynamically assigned by the executor. Example: `WHERE elementId(n) IN $step1_ids`. Do not invent parameter names.
            4. **Max Steps**: Generate a maximum of 3 steps.

            ### TASK DESIGN PHILOSOPHY:
            - **Discovery**: Steps 1 & 2 extract anchor IDs using embeddings with `top_k: 1000`.
            - **Correlation**: Middle steps bridge nodes via `MATCH` and `WHERE elementId(n) IN $stepN_ids`, returning a single column of target IDs.
            - **Synthesis**: The final step executes the comparison or counting using `UNWIND $stepN_results` to rehydrate and compare previous data.

            ### OUTPUT FORMAT:
            Return ONLY a JSON object with the following structure:
            {{
                "tasks": [
                    {{
                        "step": 1,
                        "description": "description of the cypher",
                        "cypher": "cypher here"
                    }}
                ]
            }}
            """
        else:
            return ""
        
        self.log("Cypher Generation", f"Generating {plan.get('query_type')} query using dynamic parameters...")
        # self.log("Cypher Generation", f"{prompt}")
        res = ollama.generate(model=LLM_MODEL, prompt=prompt)
        return re.sub(r"```json|```cypher|```", "", res['response']).strip()

    def execute_query(self, cypher, params):
        with self.driver.session() as session:
            try:
                result = session.run(cypher, **params)
                return result.data()
            except Exception as e:
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
        
    def validate_and_refine_query(self, user_query, history):
        """
        Validates the query against the 600,000 LinkedIn records context.
        Determines if the question is database-related, requires clarification, or should be rejected.
        """
        prompt = f"""
        Role: Senior Gatekeeper & Graph Schema Analyst for a 600,000-record LinkedIn GraphRAG system.
        
        ### 1. LIVE DATABASE SCHEMA (The Source of Truth):
        {self.cached_context}
        
        ### 2. USER INPUT:
        "{user_query}"

        Our history conversation so far: {history}
        
        ### 3. VALIDATION MANDATE:
        Your goal is to determine if this question can be translated into a Neo4j Cypher query or a Vector Search. 
        
        **CRITICAL REJECTION RULES (Read Carefully):**
        - DO NOT reject a query just because a word (e.g., 'developer') isn't a "Property Name". 
        - DO NOT reject a query if it requires counting or listing; the system is fully capable of aggregations.
        - DO NOT reject if the information can be found via a PATH. (e.g., Professionals link to Experience, which links to JobTitles).
        - DO NOT reject if the intent matches a VECTOR INDEX. 'experience_embeddings' can find "Developers", "Software Engineers", etc., even if the schema doesn't list them explicitly.
        
        ### 4. STATUS DETERMINATION:
        - **status: "approved"** -> If a path exists in the "GRAPH STRUCTURE" or a "VECTOR INDEX" exists that relates to the user's entities (Professional, Job, Education, etc.).
        - **status: "clarify"** -> If the question is purely general (e.g., "What is AI?") and doesn't mention or imply LinkedIn data/professionals.
        - **status: "rejected"** -> ONLY if the request is for data types we fundamentally do not store (e.g., "What is the stock price of Apple?" or "What is this person's home address?").

        ### 5. REFINEMENT TASK (For 'approved' status):
        Create a 'refined_query' that will be passed to the internal system.
        - Fix any spelling or grammar errors in the user's original question.
        - Keep the query natural and direct, matching the user's intent.
        - Add a short hint at the end to guide the system.

        Example:
        User: "how many devs in the datbaase"
        Refined: "How many developers are in the database? Use embeddings to find this out."


        ### OUTPUT RULES:
        - Return ONLY JSON.
        - No conversational filler.

        EXPECTED JSON FORMAT:
        {{
            "status": "approved" | "clarify" | "rejected",
            "message": "Reasoning for rejection or the clarification question.",
            "refined_query": "The detailed internal version of the query for the system."
        }}
        """

        self.log("Validation", "Checking query relevance and refining intent...")
        res = ollama.generate(model="qwen2.5:7b", prompt=prompt)
        
        try:
            # Clean and parse JSON
            clean_json = re.sub(r"```json|```", "", res['response']).strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            # Fallback for safety
            return {"status": "approved", "refined_query": user_query, "message": ""}

    def run(self, user_query,history):
        print(f"\n--- Processing: {user_query} ---")
        
        validation = self.validate_and_refine_query(user_query,history)
        
        if validation['status'] != "approved":
            return {
                "status": validation['status'],
                "reply": validation['message'],
                "user_query": user_query,
                "is_validation_hit": True
            }
        
        # Use the AI-refined query for the rest of the pipeline
        refined_query = validation['refined_query']
        self.log("Refinement", f"Original: {user_query}\nRefined: {refined_query}")
        
        context = self.cached_context
        
        # 1. Plan Strategy
        plan = self.plan_execution(refined_query, context)
        if plan.get('query_type') == 'out_of_scope':
            return {
                "user_query": user_query, 
                "final_data": None, 
                "error": plan.get('reasoning')
            }

        query_params = {}
        for entity in plan.get('embeddings_needed', []):
            query_params[entity['variable_name']] = self.get_embedding(entity['search_text'])
            
        # 3. Handle Branching Logic
        raw_output = self.generate_cypher_query(user_query, context, plan)
        results_registry = {} 

        if plan.get('query_type') == 'multi_step_analysis':
            try:
                task_data = json.loads(raw_output)
                tasks = task_data.get("tasks", [])
                for task in tasks:
                    step_id = task['step']
                    cypher_query = task['cypher']
                    
                    for prev_step, prev_data in results_registry.items():
                        query_params[f"step{prev_step}_results"] = prev_data
                        extracted_ids = [list(record.values())[0] for record in prev_data] if prev_data else []
                        query_params[f"step{prev_step}_ids"] = extracted_ids

                    step_result = self.execute_query(cypher_query, query_params)
                    results_registry[step_id] = step_result
                final_data = results_registry
            except json.JSONDecodeError:
                return {"user_query": user_query, "final_data": None, "error": "Multi-step JSON parse error"}
        else:
            final_data = self.execute_query(raw_output, query_params)

        return {
            "user_query": user_query, 
            "final_data": final_data
        }
