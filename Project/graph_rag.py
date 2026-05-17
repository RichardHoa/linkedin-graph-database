import time
import json
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

from neo4j_graphrag.schema import get_schema
import sys

sys.path.append("CyVer")
from CyVer import SyntaxValidator, SchemaValidator, PropertiesValidator
import json_repair
import requests

load_dotenv()

URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_SECRET"))
DB_NAME = "neo4j"

INTENT_MODEL = "hf.co/mradermacher/text-to-cypher-Gemma-3-27B-Instruct-2025.04.0-i1-GGUF:Q4_K_S"
TRANSFORM_MODEL = "gpt-5.4-mini"
CHAT_MODEL = "gpt-5.4-mini"
EMBED_MODEL = "mxbai-embed-large"

driver = GraphDatabase.driver(URI, auth=AUTH)


class GraphRAGPipeline:
    def __init__(self, log_indent=2):
        self.driver = driver
        self.log_indent = log_indent

        self.api_key = os.getenv("API_KEY")
        self.api_url = "https://apollo.quocanmeomeo.io.vn/v1/chat/completions"
        self.api_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.gpt_api_key = os.getenv("GPT_API_KEY")
        self.gpt_api_url = "https://api.openai.com/v1/chat/completions"
        self.gpt_api_headers = {
            "Authorization": f"Bearer {self.gpt_api_key}",
            "Content-Type": "application/json",
        }

        self.log("init", f"Using remote models for pipeline")

        self.log("init", "Initializing System Context...")
        self.cached_context = self.get_system_context()
        self.log("system context", "System context loaded.")

        self.syntax_validator = SyntaxValidator(
            self.driver, check_multilabeled_nodes=False
        )
        self.schema_validator = SchemaValidator(self.driver)
        self.props_validator = PropertiesValidator(self.driver)

    def log(self, stage, message, data=None):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage.upper(),
            "message": message,
            "data": data,
        }
        print(json.dumps(log_entry, indent=self.log_indent, default=str))

    def get_system_context(self):
        """Retrieves the LIVE database state with rich metadata and property examples."""
        schema_file = "schema_cache.txt"
        if os.path.exists(schema_file):
            self.log("context", "Loading schema from cache...")
            with open(schema_file, "r") as f:
                return f.read()

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
                    if p_type in ("STR", "STRING"):
                        distinct_cnt = session.run(
                            f"MATCH (n:{label}) WHERE n.`{p_key}` IS NOT NULL RETURN count(DISTINCT n.`{p_key}`) AS cnt"
                        ).single()["cnt"]
                        if 0 < distinct_cnt <= 25:
                            vals = session.run(
                                f"MATCH (n:{label}) WHERE n.`{p_key}` IS NOT NULL RETURN DISTINCT n.`{p_key}` AS val LIMIT 25"
                            ).value()
                            enum_str = f"Enum values: {vals}"
                        else:
                            example = str(p_val).replace("\n", " ")
                            if len(example) > 100:
                                example = example[:97] + "..."
                            enum_str = f'Example: "{example}"'
                    else:
                        example = str(p_val).replace("\n", " ")
                        if len(example) > 100:
                            example = example[:97] + "..."
                        enum_str = f'Example: "{example}"'
                    label_lines.append(f"  - `{p_key}`: {p_type} {enum_str}")
                node_props.append("\n".join(label_lines))

            # 2. Fetch relationships using path pattern matching
            rel_records = session.run(
                """
                MATCH (n)-[r]->(m) 
                RETURN DISTINCT labels(n)[0] AS source, type(r) AS type, labels(m)[0] AS target 
                LIMIT 50
            """
            ).data()

            rels = []
            for r in rel_records:
                rel_type = r["type"]
                source = r["source"]
                target = r["target"]

                sample_rel = session.run(
                    f"MATCH (:{source})-[r:{rel_type}]->(:{target}) RETURN r LIMIT 1"
                ).single()
                if sample_rel and sample_rel[0]:
                    r_props = sample_rel[0]
                    prop_lines = []
                    for p_key, p_val in r_props.items():
                        p_type = type(p_val).__name__.upper()
                        if p_type in ("STR", "STRING"):
                            distinct_cnt = session.run(
                                f"MATCH (:{source})-[r:{rel_type}]->(:{target}) WHERE r.`{p_key}` IS NOT NULL RETURN count(DISTINCT r.`{p_key}`) AS cnt"
                            ).single()["cnt"]
                            if 0 < distinct_cnt <= 25:
                                vals = session.run(
                                    f"MATCH (:{source})-[r:{rel_type}]->(:{target}) WHERE r.`{p_key}` IS NOT NULL RETURN DISTINCT r.`{p_key}` AS val LIMIT 25"
                                ).value()
                                enum_str = f"Enum values: {vals}"
                            else:
                                example = str(p_val).replace("\n", " ")
                                if len(example) > 100:
                                    example = example[:97] + "..."
                                enum_str = f'Example: "{example}"'
                        else:
                            example = str(p_val).replace("\n", " ")
                            if len(example) > 100:
                                example = example[:97] + "..."
                            enum_str = f'Example: "{example}"'
                        prop_lines.append(f"    - `{p_key}`: {p_type} {enum_str}")

                    rel_desc = f"(:{source})-[:{rel_type}]->(:{target})"
                    if prop_lines:
                        rels.append(rel_desc + "\n" + "\n".join(prop_lines))
                    else:
                        rels.append(rel_desc)
                else:
                    rels.append(f"(:{source})-[:{rel_type}]->(:{target})")

            # 3. Fetch Vector Indexes for context
            vector_indexes = session.run(
                """
                SHOW INDEXES YIELD name, type, labelsOrTypes 
                WHERE type = 'VECTOR' 
                RETURN name, labelsOrTypes[0] AS label
            """
            ).data()
            v_idx_info = [
                f"Vector Index: `{idx['name']}` on Label `{idx['label']}`"
                for idx in vector_indexes
            ]

        # Assemble the final context string
        schema_context = "Node properties:\n" + "\n".join(node_props)
        schema_context += "\n\nThe relationships:\n" + "\n".join(rels)
        if v_idx_info:
            schema_context += "\n\nAvailable Vector Indexes:\n" + "\n".join(v_idx_info)

        with open(schema_file, "w") as f:
            f.write(schema_context)

        return schema_context

    def get_embedding(self, text):
        """Generates embedding using the remote API."""
        try:
            data = {
                "model": EMBED_MODEL,
                "messages": [{"role": "user", "content": text}],
                "stream": False
            }
            response = requests.post(
                self.api_url, headers=self.api_headers, json=data, timeout=30
            )
            response.raise_for_status()
            result = response.json()
            # The server returns the embedding in the message content for this specific model
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            self.log("Embedding Error", f"Failed to generate embedding: {str(e)}")
            return None

    def generate_completion(self, messages, model=INTENT_MODEL, temperature=0.1):
        """Generates completion using the remote API."""
        processed_messages = []
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n\n"
            else:
                if msg["role"] == "user" and system_content:
                    processed_messages.append({"role": "user", "content": system_content + msg["content"]})
                    system_content = ""
                else:
                    processed_messages.append(msg)
        
        if not processed_messages and system_content:
             processed_messages.append({"role": "user", "content": system_content})

        if model.startswith("gpt-"):
            req_url = self.gpt_api_url
            req_headers = self.gpt_api_headers
        else:
            req_url = self.api_url
            req_headers = self.api_headers

        try:
            data = {
                "model": model,
                "messages": processed_messages,
                "temperature": temperature,
                "stream": False
            }
            print(data)
            response = requests.post(
                req_url, headers=req_headers, json=data, timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self.log("Generation Error", f"Failed to call remote API: {str(e)}")
            return ""

    def transform_to_vector_query(self, standard_cypher):
        """Transform standard Cypher to Cypher 25 Vector Search using external API."""
        self.log("transformation", "Applying AI semantic transformation...")

        # 1. Prepare API call
        messages = [
            {
                "role": "system",
                "content": "You are a Cypher expert. Transform legacy Cypher queries into vector semantic searches using Neo4j's CALL db.index.vector.queryNodes. Available vector indexes: Professional ('professional_embeddings'), Experience ('experience_embeddings'), Education ('education_embeddings'), Certification ('certification_embeddings'). Return ONLY a JSON object with two keys: `cypher_query` (the updated query string), and `embeddings` (a key-value map of variable names you invented to the string values that need to be embedded). Do NOT format the json in backticks.",
            },
            {
                "role": "user",
                "content": 'Transform: MATCH (e:Experience)-[:ROLE_WAS]->(j:JobTitle {name: "digital designer"}) WHERE NOT EXISTS { MATCH (e)-[:HAS_EDUCATION]->(edu:Education)-[:AT_UNIVERSITY]->(u:University) RETURN edu } RETURN count(DISTINCT e) AS count',
            },
            {
                "role": "assistant",
                "content": '{"cypher_query": "CALL db.index.vector.queryNodes(\\"experience_embeddings\\", 100000, $emb_role) YIELD node AS j, score\\nWHERE score > 0.8\\nMATCH (e:Experience)-[:ROLE_WAS]->(j)\\nWHERE NOT EXISTS {\\n    MATCH (e)-[:HAS_EDUCATION]->(:Education)-[:AT_UNIVERSITY]->(:University)\\n}\\nRETURN count(DISTINCT e) AS count", "embeddings": {"emb_role": "digital designer"}}',
            },
            {
                "role": "user",
                "content": 'Transform: MATCH (p:Professional)-[:HAS_EXPERIENCE]->(e:Experience)-[:ROLE_WAS]->(jt:JobTitle) WHERE jt.name = "developer" RETURN count(DISTINCT p)',
            },
            {
                "role": "assistant",
                "content": '{"cypher_query": "CALL db.index.vector.queryNodes(\\"experience_embeddings\\", 100000, $emb_role) YIELD node AS jt, score\\nWHERE score > 0.8\\nMATCH (p:Professional)-[:HAS_EXPERIENCE]->(e:Experience)-[:ROLE_WAS]->(jt)\\nRETURN count(DISTINCT p)", "embeddings": {"emb_role": "developer"}}',
            },
            {
                "role": "user",
                "content": 'Transform: MATCH (p:Professional) WHERE p.headline CONTAINS "Data Scientist" AND p.location CONTAINS "New York" RETURN p.name LIMIT 10',
            },
            {
                "role": "assistant",
                "content": '{"cypher_query": "CALL db.index.vector.queryNodes(\\"professional_embeddings\\", 100000, $emb_prof) YIELD node AS p, score\\nWHERE score > 0.8 AND p.location CONTAINS \\"New York\\"\\nRETURN p.name LIMIT 10", "embeddings": {"emb_prof": "Data Scientist"}}',
            },
            {"role": "user", "content": f"Transform: {standard_cypher}"},
        ]

        raw_text = self.generate_completion(messages, model=TRANSFORM_MODEL)
        if not raw_text:
            return standard_cypher, {}

        # Use json_repair to safely load potentially malformed json
        try:
            parsed_data = json_repair.loads(raw_text)
            transformed_cypher = parsed_data.get("cypher_query", standard_cypher)
            embeddings_map = parsed_data.get("embeddings", {})

            self.log("transformation", "AI Transformation successful.")
            return transformed_cypher, embeddings_map
        except Exception as e:
            self.log("transformation error", f"AI Transformation parsing failed: {str(e)}")
            return standard_cypher, {}

    def generate_cypher_query(self, user_query, schema_context):
        USER_PROMPT_TEMPLATE = """Generate a standard Neo4j Cypher query for the Question below.
Use only the provided relationship types, node labels, and properties from the Schema section.

#### Schema:
{schema}
#### Question:
{question}"""
        prompt = USER_PROMPT_TEMPLATE.format(schema=schema_context, question=user_query)
        messages = [{"role": "user", "content": prompt}]
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
        cypher = cypher.replace("\\n", "\n")
        return cypher.strip()

    def execute_query(self, cypher, params=None):
        params = params or {}
        with self.driver.session(database=DB_NAME) as session:
            try:
                t0 = time.time()
                result = session.run(cypher, **params)
                data = result.data()
                t1 = time.time()
                self.log(
                    "Neo4j Query", f"Executed in {t1 - t0:.2f}s | Rows: {len(data)}"
                )
                return data
            except Exception as e:
                self.log("Neo4j Error", f"Query failed: {str(e)}")
                return [{"error": str(e)}]

    def decide_and_respond(self, user_query, history):
        """Determine if a database query is needed or if we can respond directly."""
        self.log("router", "Deciding if DB query is necessary...")
        
        hist_str = ""
        for msg in history[-6:]: # Include last 6 messages
            hist_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
        system_instructions = (
            "You are an AI assistant that determines if a user's message requires searching an external knowledge base. "
            "Rule 1: If the user asks for factual information, data, or statistics not in the chat history, you MUST trigger a lookup. Reply EXACTLY with `SEARCH: <question>` where <question> is the user's inquiry rewritten as a single, clear English sentence. "
            "Example: `SEARCH: How many people have a development skill?`\n"
            "Rule 2: If the user is just saying hello, saying thank you, or asking a question that can be fully answered using ONLY the chat history, reply directly using a friendly, conversational tone. Do NOT include the word `SEARCH`.\n"
            "CRITICAL: NEVER write SQL, Cypher, or database code. Always use plain English."
        )
        
        prompt = f"Chat History:\n{hist_str}\n\nUser: {user_query}"
        
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt}
        ]
        
        res = self.generate_completion(messages, model=CHAT_MODEL, temperature=0.4)
        return res if res else f"SEARCH: {user_query}"
            
    def generate_chat_response(self, user_message, cypher_query, final_data, history=None):
        """Form a conversational reply based on database results and history."""
        self.log("chat response", "Structuring response...")
        system_instructions = (
            "You are a helpful AI assistant connected to a specialized Neo4j database. "
            "You govern the conversation. Always greet the user nicely if appropriate. "
            "You are provided with the conversation history, the user's message, the generated Cypher query, and the resulting database output data (JSON format). "
            "Examine if there are any errors in the DB result, and if so, apologize and explain. "
            "Otherwise, formulate a clear, readable, conversational answer directly answering the user."
            "Do NOT output plain JSON to the user unless they ask for it. Do NOT output raw Cypher to the user unless they ask for it."
        )
        
        hist_str = ""
        if history:
            for msg in history[-4:]:
                hist_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        context_str = f"Chat History:\n{hist_str}\n\nUser Message: {user_message}\nCypher Query Executed: {cypher_query}\nDatabase Output: {json.dumps(final_data)}"

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": context_str},
        ]

        res = self.generate_completion(messages, model=CHAT_MODEL, temperature=0.3)
        if res:
            return res
            
        if len(final_data) > 0 and "error" in final_data[0]:
            return f"**Database Error:**\n{final_data[0]['error']}\n\n**Generated Cypher:**\n```cypher\n{cypher_query}\n```"
        return f"**Results:**\n```json\n{json.dumps(final_data, indent=2)}\n```\n\n**Generated Cypher:**\n```cypher\n{cypher_query}\n```"

    def run(self, user_query, history=None):
        context = self.cached_context
        max_retries = 5
        history = history or []

        # Classification / Routing Layer
        decision = self.decide_and_respond(user_query, history)
        
        if not decision.startswith("SEARCH:"):
            self.log("router", "Answered directly using history/chat capabilities. Skipping DB.")
            return {
                "user_query": user_query,
                "cypher_query": "N/A",
                "final_data": [],
                "chat_reply": decision
            }
            
        # Extract the standalone query
        standalone_query = decision.replace("SEARCH:", "").replace("`", "").strip()
        self.log("router", f"Standalone DB Query: {standalone_query}")

        # Stage 1: Generate Standard Cypher intent, with validation retry
        self.log("generation", f"Stage 1: Generating standard Cypher intent...")

        standard_cypher = ""
        for attempt in range(max_retries):
            standard_cypher = self.generate_cypher_query(standalone_query, context)
            self.log(
                "generation", f"Output (Attempt {attempt+1}): {standard_cypher}"
            )

            # Syntax validation loop
            syn_score, syn_meta = self.syntax_validator.validate(
                standard_cypher, database_name=DB_NAME
            )
            prop_score, prop_meta = self.props_validator.validate(
                standard_cypher, database_name=DB_NAME, strict=False
            )
            sch_score, sch_meta = self.schema_validator.validate(
                standard_cypher, database_name=DB_NAME
            )
            
            err_ext = ""
            if syn_score is not None and syn_score != 1:
                err_ext += f"Syntax: {syn_meta}. "
            if prop_score is not None and prop_score != 1:
                err_ext += f"Props: {prop_meta}. "
            if sch_score is not None and sch_score != 1:
                err_ext += f"Schema: {sch_meta}."

            if not err_ext:
                break
            elif attempt < max_retries - 1:
                self.log(
                    "retry loop", f"Validation failed. Informing model..."
                )
                retry_msg = f"\n\nPrevious attempt failed validation. Please fix: {err_ext.strip()}"
                if retry_msg not in standalone_query:
                    standalone_query += retry_msg

        # Stage 2: Transform to Vector Search Query using External API
        cypher_query, embeddings_map = self.transform_to_vector_query(standard_cypher)

        # Fallback Logic
        if cypher_query == standard_cypher:
            self.log(
                "Fallback",
                "Vector Transformation not applied or failed. Using Standard Cypher.",
            )
            embeddings_map = {}

        # Stage 4: Inject embedding parameters
        params = {}
        if embeddings_map:
            for emb_var, semantic_term in embeddings_map.items():
                self.log(
                    "embedding",
                    f"Generating vector for semantic term: {semantic_term} into ${emb_var}",
                )
                vector = self.get_embedding(semantic_term)
                if vector:
                    params[emb_var] = vector
                else:
                    self.log("embedding", f"Failed embedding for {semantic_term}")
        elif "emb_" in cypher_query:
            # Fallback if params not parsed properly but exist in query
            self.log(
                "embedding",
                f"Fallback: Generating generic vector for query: {user_query}",
            )
            vector = self.get_embedding(user_query)
            if vector:
                params["emb_role"] = (
                    vector  # Hardcoded assumption string, hopefully avoided
                )

        # Stage 5: Final Validation and Execution
        schema_score, schema_meta = self.schema_validator.validate(
            cypher_query, database_name=DB_NAME
        )
        if schema_score != 1:
            self.log(
                "Schema Warning",
                f"Schema validation score: {schema_score}. Meta: {schema_meta}",
            )

        final_data = self.execute_query(cypher_query, params)

        # Formulate Chat Reply
        chat_reply = self.generate_chat_response(standalone_query, cypher_query, final_data, history)

        return {
            "user_query": standalone_query,
            "cypher_query": cypher_query,
            "final_data": final_data,
            "chat_reply": chat_reply,
        }
