from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://princev3844:Prince14%40@princecluster.zsdovyq.mongodb.net/?retryWrites=true&w=majority&appName=PrinceCluster"
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

music_db = client["musicCaptionDB"]
music_cap_collection = music_db["music_captioning_data"]