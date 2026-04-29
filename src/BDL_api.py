import requests
import math
from collections import Counter

SPECIES_MODEL = {
    0: ['Pinus', 'sosna'],
    1: ['Picea', 'świerk'],
    2: ['Abies', 'jodła'],
    3: ['Larix', 'modrzew'],
    4: ['Pseudotsuga', 'daglezja'],
    5: ['Quercus', 'dąb'],
    6: ['Ulmus', 'wiąz'],
    7: ['Fagus', 'buk'],
    8: ['Tilia', 'lipa'],
    9: ['Carpinus', 'grab'],
    10: ['Acer', 'klon'],
    11: ['Fraxinus', 'jesion'],
    12: ['Betula', 'brzoza'],
    13: ['Corylus', 'leszczyna'],
    14: ['Crataegus', 'głóg'],
    15: ['Others', 'Inne'],
    16: ['Incorrect segmentation', 'Błędna segmentacja']
}    

SPECIES_DBL = {
    "BRZ":  ["Betula_pendula",         "Brzoza brodawkowata"],
    "BK":   ["Fagus_sylvatica",        "Buk zwyczajny"],
    "DB.S": ["Quercus_robur",          "Dąb szypułkowy"],
    "DB.B": ["Quercus_petraea",        "Dąb bezszypułkowy"],
    "DB.C": ["Quercus_rubra",          "Dąb czerwony"],
    "GB":   ["Carpinus_betulus",       "Grab pospolity"],
    "JS":   ["Fraxinus_excelsior",     "Jesion wyniosły"],
    "KL":   ["Acer_platanoides",       "Klon pospolity"],
    "JW":   ["Acer_pseudoplatanus",    "Klon jawor"],
    "KL.P": ["Acer_campestre",         "Klon polny"],
    "LP":   ["Tilia_cordata",          "Lipa drobnolistna"],
    "WZ.S": ["Ulmus_laevis",           "Wiąz szypułkowy"],
    "GŁG":  ["Crataegus_monogyna",     "Głóg jednoszyjkowy"],
    "LSZ":  ["Corylus_avellana",       "Leszczyna pospolita"],
    "DG":   ["Pseudotsuga_menziesii",  "Daglezja zielona"],
    "JD":   ["Abies_alba",             "Jodła pospolita"],
    "MD":   ["Larix_decidua",          "Modrzew europejski"],
    "SO":   ["Pinus_sylvestris",       "Sosna zwyczajna"],
    "ŚW":   ["Picea_abies",            "Świerk pospolity"],
    "OS":   ["Populus_tremula",        "Topola osika"],
    "TP":   ["Populus_alba",           "Topola biała"],
    "TP.C": ["Populus_nigra",          "Topola czarna"],
    "DB":   ["Quercus species",        "Dąb nieokreślony"]
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

    def __init__(
        self,
        species_dbl: dict = SPECIES_DBL,
        species_model: dict = SPECIES_MODEL,
        rdlp_dict: dict = RDLP_TO_COLLECTION,
        size_m: int = 600,
        model_based: bool = True,
    ):
        self.session = requests.Session()
        self.base = "https://ogcapi.bdl.lasy.gov.pl"
        self.species_dbl = species_dbl
        self.species_model = species_model
        self.rdlp_dict = rdlp_dict
        self.size_m = size_m
        self.model_based = model_based


    def get_rdlp_collection(self, lat: float, lon: float, RDLP_dict: dict = RDLP_TO_COLLECTION):
        url = "https://ogcapi.bdl.lasy.gov.pl/collections/rdlp/items"
        params = {"bbox": f"{lon},{lat},{lon},{lat}", "f": "json"}
        feats = self.session.get(url, params=params, timeout=60).json().get("features", [])

        return RDLP_dict.get(feats[0]["properties"]["region_name"])

    def bbox_meters(self, lon: float, lat: float):
        half = self.size_m / 2
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

    def find_species(self, lat: float, lon: float, input_class: int):
        # size_m powinno być w init 
        # w find_species powinna być klasa podana jako int
        # int zwrócony przez model -> 
        # -> szukasz obszaru na którym jest dane drzewo 
        # -> patrzysz jakie drzewa występują na tym terenie
        # -> robisz dopasowanie:
        # --> jeśli masz model zwrócił np. klon: szukasz jakie klony występują na tym obszarze i zwracasz najczęstszy gatunek klona
        # --> jeśli na tym obszarze nie ma np. klonu a model go zwrócił działasz zależnie od flagi: 
        # ---> model_based: bool = True: zwracasz klon (ten który w Polsce jest częstszy, dokładana nazwa)
        # ---> model_based: bool = False: zwracasz najczęstszy gatunek z obszaru
        # --> jeśli model zwrócił others 15: zwracasz najczęstsze drzewo z obszaru
        # --> jeśli model zwrócił others 15, ale nie masz danych o obszarze: zwracasz others
        # --> tak samo jak others działa błędna segmentacja
        #
        
        def _most_often(species_found: dict):
            return next(iter(species))


        input_latin = self.species_model[input_class][0]
        print(input_latin)
        
        collection = self.get_rdlp_collection(lat, lon)
        bbox  = self.bbox_meters(lon, lat)
        feats = self.fetch_all(collection, bbox)
        count_totals = Counter()
        for f in feats:
            p = f["properties"]
            sp = p.get("species_cd")

            try:
                latin = self.species_dbl[sp][0]
                count_totals[latin] += 1
            except:
                print(f"{sp} not in the DBL dict")
            
            if input_latin == "Others":
                if count_totals:
                   _most_often(count_totals)
                else:
                    return "Others"
                
            if input_latin ==  "Incorrect segmentation":
                return "Incorrect segmentation"
            
            for species in count_totals:
                if species.find(input_latin) != -1:
                    print(count_totals)
                    print(species)
                    return species
                else:
                    if self.model_based:
                        for key, value in self.species_dbl.items():
                            if value[0].find(input) != -1:
                                return value[0]
                    else:
                        return species


        print(count_totals)





BDL = BDLCall(size_m=10000)
BDL.find_species(53.614462,22.487602, 0)
