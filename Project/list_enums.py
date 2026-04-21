from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
URI = "bolt://localhost:7687"
AUTH = ("neo4j", os.getenv("NEO4J_SECRET"))
driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    labels = session.run("CALL db.labels() YIELD label RETURN label").values()
    print("Labels:", labels)
