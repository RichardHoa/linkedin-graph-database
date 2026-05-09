# Warning
Do not run any command, this project is only meant to be run inside a VM machine, it cannot be local test as it involves running model

This code is only run on the server, attemtping to extract any info from the envrionment will fail

# AI Chat Interface

A simple Flask + Vanilla JS chat application.

## Local Setup
1. **Activate Environment:**
   `source jupyter_env/bin/activate`
2. **Install Dependencies:**
   `pip install -r requirements.txt`
3. **Run the App:**
   `python main.py`


- [x] Chat model to govern the whole chat, greet the user, and format final responses.
- [x] Coder model for converting standard Cypher queries to embedding-based vector searches.
- [] cache schema, list all the enums if possible to the schema, and cache that to a txt file
- [] check syntax and and other cyver check ater the instruct model, if oke, then pass, if not okay rerun the model until pass (max 5 times), do the same thing with embedding query


