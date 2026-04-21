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


- [] llama2:7b-chat to govern the whole chat, greed the user first, check etc, also used to answer when the output is out 
- [] fine tune the prompt of the qwen2.5-coder for better convert to embedding functions, change the code for more versatile embedding
- [] cache schema, list all the enums if possible to the schema, and cache that to a txt file
- [] check syntax and and other cyver check ater the instruct model, if oke, then pass, if not okay rerun the model until pass (max 5 times), do the same thing with embedding query


