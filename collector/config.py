from dotenv import load_dotenv 
import os 

load_dotenv()

MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN")
if MAPILLARY_TOKEN is None:
    raise ValueError("API kidhr hai lol")

BASE_URL = "https://www.mapillary.com/connect?client_id=27179133271715528"