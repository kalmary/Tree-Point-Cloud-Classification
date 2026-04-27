import requests
import math
from collections import Counter

SPECIES = {
    0:  ["BRZ",  "Betula_pendula",         "Brzoza brodawkowata"],
    1:  ["BK",   "Fagus_sylvatica",        "Buk zwyczajny"],
    2:  ["DB.B", "Quercus_petraea",        "Dąb bezszypułkowy"],
    3:  ["DB.C", "Quercus_rubra",          "Dąb czerwony"],
    4:  ["DB.S", "Quercus_robur",          "Dąb szypułkowy"],
    5:  ["GB",   "Carpinus_betulus",       "Grab pospolity"],
    6:  ["JS",   "Fraxinus_excelsior",     "Jesion wyniosły"],
    7:  ["JW",   "Acer_pseudoplatanus",    "Klon jawor"],
    8:  ["KL.P", "Acer_campestre",         "Klon polny"],
    9:  ["LP",   "Tilia_cordata",          "Lipa drobnolistna"],
    10: ["WZ.S", "Ulmus_laevis",           "Wiąz szypułkowy"],
    11: ["GŁG",  "Crataegus_monogyna",     "Głóg jednoszyjkowy"],
    12: ["LSZ",  "Corylus_avellana",       "Leszczyna pospolita"],
    13: ["DG",   "Pseudotsuga_menziesii",  "Daglezja zielona"],
    14: ["JD",   "Abies_alba",             "Jodła pospolita"],
    15: ["MD",   "Larix_decidua",          "Modrzew europejski"],
    16: ["SO",   "Pinus_sylvestris",       "Sosna zwyczajna"],
    17: ["ŚW",   "Picea_abies",            "Świerk pospolity"],
    18: ["TP",   "Populus_alba",           "Topola biała"],
    19: ["TP.C", "Populus_nigra",          "Topola czarna"],
    20: ["OS",   "Populus_tremula",        "Topola osika"],
    21: ["Other",                  "Inne"],
    22: ["Incorrect segmentation", "Błędna segmentacja"],
}

RDLP_TO_COLLECTION = {
    "BIAŁYSTOK":    "RDLP_Bialystok_wydzielenia",
    "KATOWICE":     "RDLP_Katowice_wydzielenia",
    "KRAKÓW":       "RDLP_Krakow_wydzielenia",
    "KROSNO":       "RDLP_Krosno_wydzielenia",
    "LUBLIN":       "RDLP_Lublin_wydzielenia",
    "ŁÓDŹ":         "RDLP_Lodz_wydzielenia",
    "OLSZTYN":      "RDLP_Olsztyn_wydzielenia",
    "PIŁA":         "RDLP_Pila_wydzielenia",
    "POZNAŃ":       "RDLP_Poznan_wydzielenia",
    "SZCZECIN":     "RDLP_Szczecin_wydzielenia",
    "SZCZECINEK":   "RDLP_Szczecinek_wydzielenia",
    "TORUŃ":        "RDLP_Torun_wydzielenia",
    "WROCŁAW":      "RDLP_Wroclaw_wydzielenia",
    "ZIELONA GÓRA": "RDLP_Zielona_Gora_wydzielenia",
    "GDAŃSK":       "RDLP_Gdansk_wydzielenia",
    "RADOM":        "RDLP_Radom_wydzielenia",
    "WARSZAWA":     "RDLP_Warszawa_wydzielenia",
}

class BDLCall():

    def __init__(self, species:dict = SPECIES):
        self.session = requests.Session()
        self.base = "https://ogcapi.bdl.lasy.gov.pl"


    def get_rdlp_collection(self, lat: float, lon: float, RDLP_dict: dict = RDLP_TO_COLLECTION):
        url = "https://ogcapi.bdl.lasy.gov.pl/collections/rdlp/items"
        params = {"bbox": f"{lon},{lat},{lon},{lat}", "f": "json"}
        feats = self.session.get(url, params=params, timeout=60).json().get("features", [])

        return RDLP_dict.get(feats[0]["properties"]["region_name"])

    def bbox_meters(self, lon: float, lat: float, size_m: int):
        half = size_m / 2
        dlat = half / 111_320
        dlon = half / (111_320 * math.cos(math.radians(lat))) 
        bbox = f"{lon-dlon},{lat-dlat},{lon+dlon},{lat+dlat}"
        return bbox
    
    def fetch(self, url, params=None):
        r = self.session.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    
    def fetch_all(self, collection, bbox):
        feats = []
        url = f"{self.base}/collections/{collection}/items"
        params = {"bbox": bbox, "limit": 1000, "f": "json"}
        while url:
            data = self.fetch(url, params)
            feats.extend(data.get("features", []))
            nxt = next(
                (l["href"] for l in data.get("links", []) if l.get("rel") == "next"),
                None,
            )
            url, params = nxt, None
        return feats

    def find_species(self, lat: float, lon: float, size_m: int = 300):

        collection = self.get_rdlp_collection(lat, lon)
        bbox  = self.bbox_meters(lon, lat, size_m)
        feats = self.fetch_all(collection, bbox)
        count_totals = Counter()
        for f in feats:
            p = f["properties"]
            sp = p.get("species_cd")
            if not sp:
                continue
            count_totals[sp] += 1
        
        print(count_totals)





BDL = BDLCall()
BDL.find_species(53.6700, 22.6000, 300)