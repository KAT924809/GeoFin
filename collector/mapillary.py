from collector.config import MAPILLARY_TOKEN, BASE_URL
class MapillaryCollector:

    def __init__(self):
        self.headers = {
            "Authorization": f"OAuth {MAPILLARY_TOKEN}"
        }