- Python 3.8+ 
- Android Studio 

Please contact us for if you require API key and endpoint access, we will provide you with it for judging purposes! 

1) VENV SETUP
python -m venv venv

.\venv\Scripts\Activate OR source venv/bin/activate

pip install -r requirements.txt

# these have already been added to config.py and are the backend uris, note that if restarted the uri will change!
# Omni_URI = "https://your-omniparser-tunnel-url.trycloudflare.com"
# ImageEmbedding_URI = "https://your-embedding-tunnel-url.trycloudflare.com"


2) ANDROID STUDIO
install https://developer.android.com/studio
NOTE THE PATH YOU INSTALL TO YOU NEED TO LOCATE THE ADB.EXE AND ADD IT TO config.py LATER
for me example dir "C:\Users\nicho\AppData\Local\Android\Sdk\platform-tools\adb.exe"
Directory: C:\Users\nicho\AppData\Local\Android\Sdk\platform-tools

then open Android Studio and on the "Welcome to Android Studio" window, you should see
- New Project
- Open
- Clone Repository
> More Actions

Click > More Actions and then "Virtual Device Manager" 

Start the device by pressing the play button. Return back to your IDE and check if it's up (or just check the window)

adb devices  

3) set up config.py 

'''python
# LLM Settings (OpenAI)
LLM_BASE_URL = "https://api.openai.com/v1"
LLM_API_KEY = "api key here"  
LLM_MODEL = "gpt-4.1-nano-2025-04-14"  
LLM_REQUEST_TIMEOUT = 120
LLM_MAX_RETRIES = 3
LLM_MAX_TOKEN = 4096

# Neo4j Database Settings (Aura)
Neo4j_URI = "neo4j+ssc://cbd4231c.databases.neo4j.io"  # Using +ssc for self-signed cert compatibility
Neo4j_AUTH = ("neo4j", "ub7z9Hg_hCFRXjatBqE6xlKjYm8yPC7Vw8xEhNasVQ4")
Neo4j_DATABASE = "neo4j"

# Pinecone Settings
PINECONE_API_KEY = "pcsk_6FKiYq_BRuA4qCMWLTtvHozrjxC4ekAC4bdF2s8dq7jCAoYAsQEFdEL5weYMoZWcUVgwMW"
PINECONE_INDEX_NAME = "dingdongtester"
PINECONE_HOST = "https://dingdongtester-cppuygz.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_MODEL = "llama-text-embed-v2"  # Using Pinecone-hosted model
EMBEDDING_DIMENSIONS = 1024

# LangSmith Tracing
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "dingdongtester"

# Backend Services (GPU-accelerated remote server via Cloudflare Tunnel)
# Note: These URLs may change when server restarts - update as needed
Omni_URI = "https://teen-alt-clocks-athletes.trycloudflare.com"
ImageEmbedding_URI = "https://compaq-royal-elvis-vancouver.trycloudflare.com"

# OpenAI API Key (also for embeddings if needed)
OPENAI_API_KEY = "api key here"
ADB_PATH= "your adb path"
'''

cd pipeline/

python .\nich_pipeline.py "insert task here"
e,g

python .\nich_pipeline.py "go to the messaging app, identify that there is a star button above the start chat icon and that it is blue/purplish in a white box vs the blue tinted box for the start chat icon, then click the start check icon and check that the create group button is there, then click it"

or a simpler one 

python .\nich_pipeline.py "go to the app store and search for tiktok"
