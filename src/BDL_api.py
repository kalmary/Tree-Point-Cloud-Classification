import requests

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

    def find_species(self, lat:float, lon:float, input_key: int):

        collection = self.get_rdlp_collection(lat, lon)
        print(collection)





BDL = BDLCall()
BDL.find_species(53.6472, 22.4552, 3)